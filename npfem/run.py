"""
Main script for training, evaluating, or testing NeuralPFEM.
GNN version.
"""

from npfem import learned_simulator
# === Standard Library Imports ===
import json
import os
import shutil

# === Third-Party Library Imports ===
import numpy as np
import torch
from tqdm import tqdm  # Progress bar
import hydra # Management of configuration file               
from omegaconf import DictConfig, OmegaConf
from torch_scatter import scatter_mean

# PyTorch utilities for mixed precision and optimization
from torch.cuda.amp import GradScaler

# === NeuralPFEM Library Imports ===
from npfem import utils
from npfem import data_loader
from npfem import output
from npfem.global_module import g 

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# --- Metadata initialization ---
metadata = None

# === Core Functions ===
def run_prediction(
    simulator: learned_simulator.LearnedSimulator,
    trajectory_data, 
    example_name: str,
    device: torch.device,
    cfg: DictConfig
):
    """
    Performs an autoregressive rollout prediction for a single trajectory example.
    """

    # --- Extract ground truth tensors ---
    g.position_gt = trajectory_data.position.to(device)      # [T, N, D]
    g.velocity_gt = trajectory_data.velocity.to(device)      # [T, N, D]
    g.pressure_gt = trajectory_data.pressure.to(device)      # [T, N]
    g.material_properties = trajectory_data.material_properties.to(device) 
    g.angles = trajectory_data.angles.to(device)

    g.nsteps_total = g.position_gt.shape[0]
    g.nsteps_predict = g.nsteps_total - cfg.input_sequence_length

    # --- Initialize rollout state ---
    g.current_position = g.position_gt[cfg.input_sequence_length - 1]
    g.previous_velocity = g.velocity_gt[:cfg.input_sequence_length].permute(1, 0, 2)
    g.previous_pressure = g.pressure_gt[:cfg.input_sequence_length].permute(1, 0)

    predicted_position_list = [g.position_gt[i] for i in range(cfg.input_sequence_length)]
    predicted_velocity_list = [g.velocity_gt[i] for i in range(cfg.input_sequence_length)]
    predicted_pressure_list = [g.pressure_gt[i] for i in range(cfg.input_sequence_length)]
    predicted_cells_list = []

    simulator.initialize_mesher(metadata["h"], cfg.alpha, cfg.refine, cfg.rmin_factor, cfg.rmax_factor)
    g.cells, _, g.tags = simulator.first_mesh(g.current_position.detach().cpu().numpy())
    predicted_cells_list.extend([g.cells] * cfg.input_sequence_length)
    # --- Error tracking ---
    mse_pos_list, cd_pos_list, cd_pos_press_list = [], [], []
    for g.step in tqdm(range(g.nsteps_predict), total=g.nsteps_predict):
        
        simulator.learned_update()
        # Store predictions
        predicted_position_list.append(g.current_position)
        predicted_velocity_list.append(g.current_velocity)
        predicted_pressure_list.append(g.current_pressure)
        predicted_cells_list.append(g.cells) 

        # Update state
        g.previous_velocity = torch.cat([g.previous_velocity[:, 1:], g.current_velocity.unsqueeze(1)], dim=1)
        g.previous_pressure = torch.cat([g.previous_pressure[:, 1:], g.current_pressure.unsqueeze(1)], dim=1)

        # --- Error computation ---
        if cfg.mode == "valid" or cfg.compute_errors:
            gt_idx = cfg.input_sequence_length + g.step
            mse_val, cd_val, cd_press_val = utils.compute_prediction_errors(
                pred_pos_t=g.current_position,
                pred_press_t=g.current_pressure,
                gt_pos_t=g.position_gt[gt_idx],
                gt_press_t=g.pressure_gt[gt_idx]
            )
            mse_pos_list.append(mse_val)
            cd_pos_list.append(cd_val)
            cd_pos_press_list.append(cd_press_val)
        
        g.current_position = g.next_position

    simulator.finalize()
    print("Rollout prediction finished.")

    # --- Aggregate errors ---
    mse_pos = 0#torch.stack(mse_pos_list).mean().item() if mse_pos_list else float("nan")
    cd_pos = torch.stack(cd_pos_list).mean().item() if cd_pos_list else float("nan")
    cd_pos_press = 0#torch.stack(cd_pos_press_list).mean().item() if cd_pos_press_list else float("nan")
    errors = (mse_pos, cd_pos, cd_pos_press)

    # --- Direct VTK writing ---
    if cfg.mode == "test" and cfg.vtk_output:
        positions_np = [p.cpu().numpy() for p in predicted_position_list]
        velocities_np = [v.cpu().numpy() for v in predicted_velocity_list]
        pressures_np = [p.cpu().numpy() for p in predicted_pressure_list]
        cells_np = [c for c in predicted_cells_list]

        vtk_dir = os.path.join(cfg.output_path, cfg.model_name, "output_files", f"{example_name}_vtk")
        print(f"Writing VTU files to: {vtk_dir}")
        try:
            output.write_vtu(
                positions_list=positions_np,
                velocities_list=velocities_np,
                pressures_list=pressures_np,
                cells_list=cells_np,
                output_dir=vtk_dir,
            )
        except Exception as e:
            print(f"Error writing VTU files: {e}")

    return errors


def test(device: str, cfg: DictConfig):
    """Run inference on the test dataset and optionally save results as VTK."""
    print(f"--- Starting Test Mode on {device} ---")

    test_data_path = os.path.join(cfg.data_path, "test.h5")
    print(f"Loading test data from: {test_data_path}")
    try:
        test_loader = data_loader.get_prediction_data_loader(path=test_data_path)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

    simulator = initialize(device, cfg)

    model_path = os.path.join(cfg.output_path, cfg.model_name, "models", cfg.model_file)
    print(f"Loading model from: {model_path}")
    try:
        simulator.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    simulator.to(device)
    simulator.eval()

    all_errors = []
    with torch.no_grad():
        for example_i, (trajectory_data, example_name) in enumerate(test_loader):
            print(f"\n--- Predicting Test Example {example_i}: {example_name} ---")

            with torch.autocast(device_type=str(device).split(':')[0], enabled=cfg.use_amp):
                ex_errors = run_prediction(simulator, trajectory_data, example_name, device, cfg)

            if cfg.compute_errors:
                print(f"Example {example_i} ({example_name}) Errors:")
                print(f"\tMSE (Pos): {ex_errors[0]}")
                print(f"\tCD (Pos):  {ex_errors[1]}")
                print(f"\tCD (Pos+Press): {ex_errors[2]}")
                all_errors.append(ex_errors)

    # Report mean errors
    if all_errors and cfg.compute_errors:
        mean_errors = np.mean(np.array(all_errors), axis=0)
        print("=" * os.get_terminal_size().columns)
        print("--- Mean Errors on Test Dataset ---")
        print(f"MSE (Pos): {mean_errors[0]}")
        print(f"CD (Pos):  {mean_errors[1]}")
        print(f"CD (Pos+Press): {mean_errors[2]}")
        print("=" * os.get_terminal_size().columns)


def validation(device: str, cfg: DictConfig):
    """Evaluate multiple model checkpoints on the validation dataset."""
    print(f"--- Starting Validation on {device} ---")

    valid_data_path = os.path.join(cfg.data_path, "valid.h5")
    print(f"Loading validation data from: {valid_data_path}")
    try:
        valid_loader = data_loader.get_prediction_data_loader(path=valid_data_path)
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return

    simulator = initialize(device, cfg)
    simulator.to(device)

    best_models = [[] for _ in range(3)]  # track top 10 models for each metric
    error_names = ["MSE_Pos", "CD_Pos", "CD_PosPress"]

    print(f"Evaluating models from step {cfg.start_eval_model} to {cfg.end_eval_model} (step {cfg.nsave_steps})")

    for step_num in range(cfg.start_eval_model, cfg.end_eval_model + 1, cfg.nsave_steps):
        model_file = f"model-{step_num}.pt"
        model_path = os.path.join(cfg.output_path, cfg.model_name, "models", model_file)

        if not os.path.exists(model_path):
            continue

        print(f"\n--- Evaluating Model: {model_file} ---")
        try:
            simulator.load(model_path)
        except Exception as e:
            print(f"Error loading {model_file}: {e}. Skipping.")
            continue

        simulator.eval()

        model_errors = []
        with torch.no_grad():
            for trajectory_data, example_name in valid_loader:
                with torch.autocast(device_type=str(device).split(':')[0], enabled=cfg.use_amp):
                    ex_errors = run_prediction(simulator, trajectory_data, example_name, device, cfg)
                model_errors.append(ex_errors)

        if not model_errors:
            continue

        mean_errors = np.mean(np.array(model_errors), axis=0)
        print(f"Model {step_num} Mean Errors:")
        for name, error in zip(error_names, mean_errors):
            print(f"\t{name}: {error:.6f}")

        # Save history
        try:
            with open(os.path.join(cfg.output_path, cfg.model_name, "error_history.txt"), "a") as f:
                f.write(f"{step_num}\t" + "\t".join(f"{x:.6f}" for x in mean_errors) + "\n")
        except IOError as e:
            print(f"Warning: Could not write to error_history.txt: {e}")

        # Track best models
        best_models = utils.update_best_models(best_models, error_names, mean_errors, step_num, cfg)

    print("\n--- Validation Complete ---")

def random_walk_noise(
    velocity_sequence: torch.Tensor, # [N, SeqLen, Dims]
    pressure_sequence: torch.Tensor, # [N, SeqLen]
    device: torch.device,
    batch_vector: torch.Tensor,    # [N] - identifies sample ID for each particle
    cfg: DictConfig
):
    """
    Generates random-walk noise for velocity and pressure sequences,
    calculating noise magnitude based on per-sample statistics.
    The noise accumulates over the sequence length.

    Args:
        velocity_sequence: Velocity history tensor. Shape [N, SeqLen, Dims].
        pressure_sequence: Pressure history tensor. Shape [N, SeqLen].
        device: PyTorch device.
        batch_vector: Tensor indicating which sample each particle belongs to. Shape [N].
        cfg (DictConfig): Configuration object from Hydra containing input parameters.

    Returns:
        Tuple: (position_noise, velocity_noise, pressure_noise)
               position_noise shape: [N, Dims] (derived from accumulated vel noise)
               velocity_noise shape: [N, SeqLen, Dims]
               pressure_noise shape: [N, SeqLen]
    """

    # --- Velocity Noise ---

    # Detach sequences before calculating stats to avoid impacting gradients
    # Calculate mean absolute velocity *per sample* using batch_vector
    abs_vel_per_particle = torch.mean(torch.abs(velocity_sequence.detach()), dim=[1, 2]) # Mean over SeqLen and Dims -> [N]
    #mean_abs_vel_per_sample = scatter_mean(abs_vel_per_particle, batch_vector, dim=0) # Mean over particles belonging to the same sample -> [NumSamples]

    # Get the mean abs vel back to per-particle, indexed by batch_vector
    # Shape: [N] (where each element corresponds to the mean abs vel of its sample)
    #mean_abs_vel_for_particle = mean_abs_vel_per_sample[batch_vector]

    # Calculate noise std per particle (based on its sample's mean abs vel)
    # Shape: [N]
    #vel_noise_std_per_particle = mean_abs_vel_for_particle * cfg.vel_noise_weight

    # Generate base noise steps per particle/timestep
    # Shape: [N, SeqLen-1, Dims]
    velocity_noise_steps = torch.randn_like(velocity_sequence[:, 1:, :], device=device)

    # Scale noise steps by the per-particle std (needs broadcasting over SeqLen-1 and Dims)
    # Shape: [N, SeqLen-1, Dims] * [N, 1, 1] -> [N, SeqLen-1, Dims]
    #velocity_noise_steps = velocity_noise_steps * vel_noise_std_per_particle.unsqueeze(1).unsqueeze(2)
    velocity_noise_steps = velocity_noise_steps * abs_vel_per_particle.unsqueeze(1).unsqueeze(2) * cfg.vel_noise_weight

    # Accumulate noise steps over time for each particle (independent of batch)
    accumulated_vel_noise = torch.cumsum(velocity_noise_steps, dim=1) # [N, SeqLen-1, Dims]

    # Add zero noise at the first timestep
    velocity_noise = torch.cat([
        torch.zeros_like(accumulated_vel_noise[:, 0:1, :]), # [N, 1, Dims]
        accumulated_vel_noise                              # [N, SeqLen-1, Dims]
    ], dim=1) # -> [N, SeqLen, Dims]


    # --- Pressure Noise ---

    # Calculate mean pressure *per sample* using batch_vector
    press_per_particle = torch.mean(pressure_sequence.detach(), dim=1) # Mean over SeqLen -> [N]
    mean_press_per_sample = scatter_mean(press_per_particle, batch_vector, dim=0) # Mean over particles belonging to the same sample -> [NumSamples]

    # Get the mean pressure back to per-particle, indexed by batch_vector
    # Shape: [N]
    mean_press_for_particle = mean_press_per_sample[batch_vector]

    # Calculate noise std per particle (based on its sample's mean pressure)
    # Shape: [N]
    press_noise_std_per_particle = mean_press_for_particle * cfg.press_noise_weight

    # Generate base noise steps per particle/timestep
    # Shape: [N, SeqLen-1]
    pressure_noise_steps = torch.randn_like(pressure_sequence[:, 1:], device=device)

    # Scale noise steps by the per-particle std (needs broadcasting over SeqLen-1)
    # Shape: [N, SeqLen-1] * [N, 1] -> [N, SeqLen-1]
    pressure_noise_steps = pressure_noise_steps * press_noise_std_per_particle.unsqueeze(1)
    #pressure_noise_steps = pressure_noise_steps * press_per_particle.unsqueeze(1) * cfg.press_noise_weight

    # Accumulate noise steps over time for each particle (independent of batch)
    accumulated_press_noise = torch.cumsum(pressure_noise_steps, dim=1) # [N, SeqLen-1]

    # Add zero noise at the first timestep
    pressure_noise = torch.cat([
        torch.zeros_like(accumulated_press_noise[:, 0:1]), # [N, 1]
        accumulated_press_noise                           # [N, SeqLen-1]
    ], dim=1) # -> [N, SeqLen]

    position_noise = torch.cumsum(velocity_noise, dim=1)[:,-1,:] * cfg.pos_noise_weight # [N, Dims]

    return position_noise, velocity_noise, pressure_noise

def compute_loss(pred_vel, target_vel, pred_press, target_press,
                 vel_weight=1.0, press_weight=1.0):
    """
    Compute combined velocity + pressure loss.

    Args:
        pred_vel (Tensor): [N, D]
        target_vel (Tensor): [N, D]
        pred_press (Tensor): [N] or [N, 1]
        target_press (Tensor): [N] or [N, 1]
        vel_weight (float): Weight of velocity loss
        press_weight (float): Weight of pressure loss
    """

    vel_loss = torch.nn.functional.mse_loss(pred_vel, target_vel)
    press_loss = torch.nn.functional.mse_loss(pred_press, target_press)

    return vel_weight * vel_loss + press_weight * press_loss

def train(device: str, cfg: DictConfig):
    """
    Main training function for the learned simulator.

    This function orchestrates the training process. It includes logic
    to resume training from a previous checkpoint if specified in the configuration.
    After setup, it proceeds with the training loop (loading data, performing training
    steps and saving checkpoints).

    Args:
        device (str): The computing device to train on (e.g., 'cuda:0', 'cpu').
        cfg (DictConfig): Configuration object from Hydra containing input parameters.
    """
    print(f"--- Starting Training on device {device} ---")

    # --- Initialization ---
    simulator = initialize(device, cfg)
    #optimizer = torch.optim.AdamW(simulator.parameters(), lr=cfg.lr_init, weight_decay=1e-4) 
    optimizer = torch.optim.Adam(simulator.parameters(), lr=cfg.lr_init)
    scaler = GradScaler(enabled=cfg.use_amp) 

    start_step = 0
    start_epoch = 0

    # --- Resume Training Logic ---
    if cfg.continue_training:
        print("Attempting to resume training...")
        latest_model, latest_state, latest_step = utils.find_latest_checkpoint(os.path.join(cfg.output_path, cfg.model_name))
        if latest_model and latest_state:
            print(f"Resuming from checkpoint step {latest_step}")
            try:
                simulator.load(latest_model)
                train_state = torch.load(latest_state, map_location=device)
                optimizer.load_state_dict(train_state["optimizer_state"])
                utils.optimizer_to(optimizer, device) 
                # Resume from next step
                start_step = train_state.get("global_train_state", {}).get("step", latest_step) + 1
                # Use cfg.use_amp
                if 'scaler_state' in train_state and cfg.use_amp:
                    scaler.load_state_dict(train_state['scaler_state'])
                print(f"Successfully loaded state. Resuming from step {start_step}.")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint state ({e}). Starting training from scratch.")
                start_step = 0
        else:
            print("No complete checkpoint found. Starting training from scratch.")

    simulator.to(device)
    simulator.train()

    # --- Data Loader ---
    train_data_path = os.path.join(cfg.data_path, "train.h5")
    print(f"Loading training data from: {train_data_path}")
    try:
        train_loader = data_loader.get_training_data_loader(
            path=train_data_path,
            input_length_sequence=cfg.input_sequence_length,
            batch_size=cfg.batch_size,
            shuffle=True
        )
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_data_path}")
        return
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # --- Training Loop ---
    step = start_step
    epoch = start_epoch

    print(f"Starting training loop from step {step} up to {cfg.ntraining_steps}")

    try:
        while step < cfg.ntraining_steps:
            epoch_iterator = tqdm(train_loader,
                                  desc=f"Epoch {epoch}",
                                  unit="step",
                                  leave=True)
            epoch_loss_sum = 0.0
            epoch_steps = 0

            for batch_idx, batch in enumerate(epoch_iterator):
                if step >= cfg.ntraining_steps:
                    break

                batch = batch.to(device)

                # --- Extract Data from PyG Batch ---
                position = batch.position
                velocity = batch.velocity
                pressure = batch.pressure
                material_properties = batch.material_properties  # [N, num_props]
                cells = batch.cells
                labels = batch.y
                target_vel = labels[:, :-1]
                target_press = labels[:, -1]

                
                

                
                # --- Noise Injection ---
                pos_noise, vel_noise, press_noise = random_walk_noise(
                    velocity, pressure, device, batch.batch, cfg
                )
                pos_mask = velocity[:,-1]==0
                vel_mask = velocity==0
                pos_noise[pos_mask] = 0
                vel_noise[vel_mask,] = 0
                noisy_position = position + pos_noise
                noisy_velocity = velocity + vel_noise
                noisy_pressure = pressure + press_noise
                '''
                # Create triangulation
                triang = mtri.Triangulation(noisy_position[:, 0].cpu().numpy(), noisy_position[:, 1].cpu().numpy(), cells.cpu().numpy())

                # --- Plot ---
                # Plot mesh
                plt.triplot(triang, linewidth=1)

                # Plot the two points
                #plt.scatter(position[:, 0].cpu().numpy(), position[:, 1].cpu().numpy(), color='red', s=80, zorder=5, label="Points")

                plt.savefig(f"train_step_{step}_mesh.png")
                plt.close()
                input("Press Enter to continue...")
                write_vtk.write_vtu(
                    positions_list=[noisy_position],
                    velocities_list=[noisy_velocity[:,-1,:]],
                    pressures_list=[noisy_pressure[:,-1]],
                    cells_list=[cells],
                    output_dir=f"input_{step}.vtu",
                )
                input("Press Enter to continue...")
                '''
                # --- Forward Pass ---
                optimizer.zero_grad()
                with torch.autocast(device_type=str(device).split(':')[0], enabled=cfg.use_amp):
                    pred_vel, pred_press = simulator(
                        current_position=noisy_position,
                        velocity_sequence=noisy_velocity,
                        pressure_sequence=noisy_pressure,
                        material_properties=material_properties,
                        cells=cells
                    )

                    # Normalize targets
                    vel_stats = simulator._normalization_stats["velocity"]
                    press_stats = simulator._normalization_stats["pressure"]

                    #target_norm_vel = (target_vel - vel_stats["mean"]) / vel_stats["std"]
                    #target_norm_press = (target_press.squeeze() - press_stats["mean"]) / press_stats["std"]
                    #target_vel = (target_vel- noisy_velocity[:,-1,:]) / 0.005
                    #target_press = (target_press - noisy_pressure[:,-1]) / 0.005
                    '''
                    write_vtk.write_vtu(
                        positions_list=[noisy_position],
                        velocities_list=[pred_vel * vel_stats["std"] + vel_stats["mean"]],
                        pressures_list=[pred_press * press_stats["std"] + press_stats["mean"]],
                        cells_list=[cells],
                        output_dir=f"output_{step}.vtu",
                    )
                    input("Press Enter to continue...")
                    '''
                    target_norm_vel = (target_vel - vel_stats["mean"]) / vel_stats["std"]
                    target_norm_press = (target_press - press_stats["mean"]) / press_stats["std"]
                    loss = compute_loss(pred_vel, target_norm_vel, pred_press, target_norm_press)
                #if loss.item() <= 10:
                # --- Backward Pass and Optimization ---
                scaler.scale(loss).backward()
                #scaler.unscale_(optimizer)
                # --- Gradient Clipping ---
                #torch.nn.utils.clip_grad_norm_(simulator.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()

                # --- Learning Rate Decay ---
                lr_new = cfg.lr_init * (cfg.lr_decay ** (step / cfg.lr_decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_new

                # --- Logging and Checkpointing ---
                current_loss = loss.item()
                epoch_loss_sum += current_loss
                epoch_steps += 1
                epoch_iterator.set_postfix(loss=f"{current_loss:.4f}", lr=f"{lr_new:.1e}")
                if cfg.print_loss_frequency == 'step':
                    tqdm.write(f'Step: {step}/{cfg.ntraining_steps} | Loss: {current_loss:.6f} | LR: {lr_new:.1e}')

                # --- Checkpointing ---
                if step % cfg.nsave_steps == 0 and step > 0:
                    tqdm.write(f"\nSaving checkpoint at step {step}...")
                    model_save_path = os.path.join(cfg.output_path, cfg.model_name, "models", f'model-{step}.pt')
                    train_state_save_path = os.path.join(cfg.output_path, cfg.model_name, "train_states", f'train_state-{step}.pt')
                    simulator.save(model_save_path)
                    train_state = {
                        'optimizer_state': optimizer.state_dict(),
                        'global_train_state': {'step': step},
                        'loss': current_loss,
                        'scaler_state': scaler.state_dict() if cfg.use_amp else None
                    }
                    torch.save(train_state, train_state_save_path)
                    tqdm.write(f"Checkpoint saved.\n")

                step += 1

            # --- End of Epoch ---
            epoch_iterator.close()

            if epoch_steps > 0:
                avg_epoch_loss = epoch_loss_sum / epoch_steps
                print(f'--- Epoch {epoch} Finished. Average Loss: {avg_epoch_loss:.6f}')
                try:
                    with open(os.path.join(cfg.output_path, cfg.model_name, "epoch_train_loss_history.txt"), "a") as f:
                        f.write(f"{epoch}\t{avg_epoch_loss:.6f}\n")
                except IOError as e:
                     print(f"Warning: Could not write to epoch_train_loss_history.txt: {e}")
            else:
                 print(f'--- Epoch {epoch} Finished (No steps completed this epoch). ---')

            epoch += 1
            print("-" * 50)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        print(f"Saving final model and state at step {step}...")
        final_model_path = os.path.join(cfg.output_path, cfg.model_name, "models", f'model-{step}.pt')
        final_state_path = os.path.join(cfg.output_path, cfg.model_name, "train_states", f'train_state-{step}.pt')
        simulator.save(final_model_path)
        final_train_state = {
            'optimizer_state': optimizer.state_dict(),
            'global_train_state': {'step': step},
            'loss': current_loss if 'current_loss' in locals() else float('nan'),
            'scaler_state': scaler.state_dict() if cfg.use_amp else None # Use cfg.use_amp
        }
        torch.save(final_train_state, final_state_path)
        print(f"Final checkpoint saved to {final_model_path} and {final_state_path}")


def initialize(device: torch.device, cfg: DictConfig) -> learned_simulator.LearnedSimulator:
    """
    Instantiate the LearnedSimulator model.

    Args:
        device (torch.device): Target device ('cpu' or 'cuda:X').
        cfg (DictConfig): Hydra configuration.

    Returns:
        learned_simulator.LearnedSimulator: Initialized simulator.
    """

    # --- 1. Check Metadata ---
    if metadata is None:
        raise ValueError("Metadata must be loaded before calling initialize.")

    # --- 2. Normalization Statistics ---
    normalization_stats = {
        "velocity": {
            "mean": torch.tensor(metadata["vel_mean"], dtype=torch.float32, device=device),
            "std":  torch.tensor(metadata["vel_std"], dtype=torch.float32, device=device),
        },
        "pressure": {
            "mean": metadata["press_mean"],
            "std":  metadata["press_std"],
        },
        "position": {
            "min": torch.tensor(metadata["pos_min"], dtype=torch.float32, device=device),
            "max": torch.tensor(metadata["pos_max"], dtype=torch.float32, device=device),
        },
    }

    # --- 3. Input Feature Dimension ---
    #   - velocity history: input_sequence_length * dim
    #   - pressure history: input_sequence_length * 1
    #   - material properties: n_material_properties
    nnode_in = (
        cfg.input_sequence_length * 2
        + cfg.input_sequence_length * 1
        + cfg.n_material_properties 
    )
    nedge_in = 3 #+ cfg.n_material_properties 

    # --- 4. Instantiate Simulator ---
    simulator = learned_simulator.LearnedSimulator(
        nnode_in=nnode_in,
        nedge_in=nedge_in,
        latent_dim=cfg.latent_dim,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        mp_steps=cfg.mp_steps,
        nmlp_layers=cfg.n_layers,
        normalization_stats=normalization_stats,
        dt=metadata["dt"],
        h=metadata["h"],
        spatial_norm_weight=cfg.spatial_norm_weight,
        vel_norm_weight=cfg.vel_norm_weight,
        press_norm_weight=cfg.press_norm_weight,
        remesh_freq=cfg.remesh_freq,
        device=device,
    )
    return simulator


# === Main Execution ===
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main function for training, validation, or testing."""

    # --- Device Setup ---
    if torch.cuda.is_available():
        if cfg.cuda_device_number is not None:
            device = torch.device(f'cuda:{cfg.cuda_device_number}')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Metadata Loading ---
    if not cfg.data_path:
        print("Error: 'data_path' must be specified in config or command line.")
        return

    metadata_path = os.path.join(cfg.data_path, "metadata.json")
    print(f"Loading metadata from: {metadata_path}")
    try:
        with open(metadata_path, 'rt') as fp:
            global metadata
            metadata = json.load(fp)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # --- Train, valid or test ---
    model_dir = os.path.join(cfg.output_path, cfg.model_name)
    if cfg.mode == "train":
        os.makedirs(os.path.join(model_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "train_states"), exist_ok=True)

        if not cfg.continue_training:
            # Save a copy of Hydra config
            hydra_cfg_path = os.path.join(os.getcwd(), ".hydra", "config.yaml")
            target_cfg_path = os.path.join(model_dir, "config.yaml")
            os.makedirs(model_dir, exist_ok=True)
            shutil.move(hydra_cfg_path, target_cfg_path)
            print(f"Saved config.yaml to: {target_cfg_path}")

        train(device, cfg)

    elif cfg.mode == "valid":
        os.makedirs(os.path.join(model_dir, "models"), exist_ok=True)
        validation(device, cfg)

    elif cfg.mode == "test":
        os.makedirs(cfg.output_path, exist_ok=True)
        test(device, cfg)


if __name__ == "__main__":
    main()