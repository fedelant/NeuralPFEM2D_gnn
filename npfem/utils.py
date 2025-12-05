# === Utility Functions ===
import glob
import re
import os
import torch
import bisect

def optimizer_to(optim, device):
    """
          BEFORE loading the optimizer state_dict, but can be necessary in some cases.
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def chamfer_distance(x, y, batch_reduction='mean', point_reduction='mean'):
    """
    Custom implementation of Chamfer Distance.
    Args:
        x (torch.Tensor): Predicted point cloud of shape (B, N, D).
        y (torch.Tensor): Ground truth point cloud of shape (B, M, D).
        batch_reduction (str): 'mean', 'sum', or None. How to reduce over the batch.
                               If None, returns per-batch-item distances.
        point_reduction (str): 'mean' or 'sum'. How to reduce over points in each cloud
                               for the two terms of the Chamfer distance.
    Returns:
        torch.Tensor: Chamfer distance.
        None: Placeholder for indices, to match PyTorch3D's return signature if needed by other parts.
    """
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("Inputs x and y must be 3D tensors (B, N, D)")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError("Batch size (dim 0) and feature dimension (dim 2) must match for x and y.")
    
    # Check for empty point clouds - if N or M is 0, behavior might be undefined or error-prone
    # For this specific use-case, N and M are expected to be > 0 from npt.
    if x.shape[1] == 0 and y.shape[1] == 0: # Both empty
        return torch.tensor(0.0, device=x.device, dtype=x.dtype), None
    if x.shape[1] == 0 or y.shape[1] == 0: # One empty
        # Return a large value or handle as per desired behavior for one-sided empty clouds
        # For simplicity here, returning a high value if point_reduction is 'mean'
        # If point_reduction is 'sum', it would be 0 for the empty side, which might be misleading.
        # Pytorch3D returns inf or a large number. Let's return a large number.
        # This edge case handling might need refinement based on exact desired properties.
        return torch.tensor(float('inf'), device=x.device, dtype=x.dtype), None


    B, N, D = x.shape
    _B, M, _D = y.shape

    # Expand dimensions for broadcasting:
    # x_expanded: (B, N, 1, D)
    # y_expanded: (B, 1, M, D)
    x_expanded = x.unsqueeze(2)
    y_expanded = y.unsqueeze(1)

    # Pairwise squared distances: (B, N, M)
    # (x_i - y_j)^2
    dist_matrix_sq = torch.sum((x_expanded - y_expanded) ** 2, dim=3)

    # For each point in x, find the min squared distance to any point in y
    min_dist_x_to_y_sq, _ = torch.min(dist_matrix_sq, dim=2)  # (B, N)
    # For each point in y, find the min squared distance to any point in x
    min_dist_y_to_x_sq, _ = torch.min(dist_matrix_sq, dim=1)  # (B, M)

    if point_reduction == 'mean':
        term1 = torch.mean(min_dist_x_to_y_sq, dim=1)  # (B,)
        term2 = torch.mean(min_dist_y_to_x_sq, dim=1)  # (B,)
    elif point_reduction == 'sum':
        term1 = torch.sum(min_dist_x_to_y_sq, dim=1)   # (B,)
        term2 = torch.sum(min_dist_y_to_x_sq, dim=1)   # (B,)
    else:
        raise ValueError(f"Unknown point_reduction: {point_reduction}")

    chamfer_dist_per_batch = term1 + term2 # (B,)

    if batch_reduction == 'mean':
        final_chamfer_dist = torch.mean(chamfer_dist_per_batch)
    elif batch_reduction == 'sum':
        final_chamfer_dist = torch.sum(chamfer_dist_per_batch)
    elif batch_reduction is None:
        final_chamfer_dist = chamfer_dist_per_batch # Return per-batch item distances
    else:
        raise ValueError(f"Unknown batch_reduction: {batch_reduction}")
    
    return final_chamfer_dist, None

def find_latest_checkpoint(model_path):
    """Finds the checkpoint with the highest step number in the model directory."""
    fnames = glob.glob(os.path.join(model_path, "models", 'model-*.pt'))
    max_step = -1
    latest_model_file = None
    expr = re.compile(r".*model-(\d+).pt")
    for fname in fnames:
        match = expr.search(fname)
        if match:
            step_num = int(match.groups()[0])
            if step_num > max_step:
                max_step = step_num
                latest_model_file = fname
    if max_step != -1:
        train_state_file = os.path.join(model_path, "train_states", f"train_state-{max_step}.pt")
        if not os.path.exists(train_state_file):
             print(f"Warning: Found model-{max_step}.pt but not train_state-{max_step}.pt")
             train_state_file = None # Cannot resume optimizer state etc.
        return latest_model_file, train_state_file, max_step
    else:
        return None, None, -1

def update_best_models(best_models_list, metric_names, current_errors, model_index, cfg):
    """
    Updates and logs the top 10 models based on different error metrics.

    Args:
        best_models_list: List of lists, where each inner list stores tuples
                          of (error, model_index) for a specific metric, sorted by error.
        metric_names: List of names corresponding to the error metrics.
        current_errors: List or array of the mean errors for the current model.
        model_index (int): The step number of the current model.

    Returns:
        The updated best_models_list.
    """
    num_metrics = len(metric_names)
    if len(best_models_list) != num_metrics or len(current_errors) != num_metrics:
        print("Warning: Mismatch between number of metrics and error lists in update_best_models.")
        return best_models_list

    for i, error in enumerate(current_errors):
        # Create item to insert: (error, model_index)
        item = (error, model_index)
        # Insert into the sorted list for the i-th metric
        bisect.insort(best_models_list[i], item)
        # Keep only the top 10
        if len(best_models_list[i]) > 10:
            best_models_list[i].pop() # Remove the worst (highest error)

    # Write the current top 10 best models to a file
    try:
        # Again, consider using Hydra's log directory
        with open(os.path.join(cfg.output_path, cfg.model_name, "best_models.txt"), "w") as f:
            f.write("--- Top 10 Models by Metric ---\n\n")
            for i, metric_list in enumerate(best_models_list):
                f.write(f"Metric: {metric_names[i]}\n")
                if not metric_list:
                    f.write("  (No models recorded yet)\n")
                else:
                    for rank, (err_val, model_idx) in enumerate(metric_list, start=1):
                        f.write(f"  {rank}. Model-{model_idx} \t Error = {err_val:.6f}\n")
                f.write("\n")
    except IOError as e:
        print(f"Warning: Could not write to best_models.txt: {e}")

    return best_models_list

def compute_prediction_errors(pred_pos_t, pred_press_t, gt_pos_t, gt_press_t):
    """
    Computes error metrics between predicted and ground truth states.

    Args:
        pred_pos_t (torch.Tensor): Predicted positions [N, D].
        pred_press_t (torch.Tensor): Predicted pressures [N].
        gt_pos_t (torch.Tensor): Ground truth positions [N, D].
        gt_press_t (torch.Tensor): Ground truth pressures [N].

    Returns:
        tuple: (mse_pos, chamfer_pos, chamfer_pos_press)
    """
    # 1. MSE on positions
    mse_pos = 0#torch.mean((pred_pos_t - gt_pos_t) ** 2)

    # 2. Chamfer distance on positions
    cd_pos, _ = chamfer_distance(
        pred_pos_t.unsqueeze(0), gt_pos_t.unsqueeze(0), batch_reduction=None
    )
    '''

    # 3. Chamfer distance on [position + normalized pressure] (optional)
    # You can enable normalization if needed
    epsilon = 1e-8
    max_gt_pos = torch.max(torch.abs(gt_pos_t))
    max_gt_press = torch.max(torch.abs(gt_press_t))

    pred_pos_norm = pred_pos_t / (max_gt_pos + epsilon)
    gt_pos_norm = gt_pos_t / (max_gt_pos + epsilon)
    pred_press_norm = pred_press_t.unsqueeze(-1) / (max_gt_press + epsilon)
    gt_press_norm = gt_press_t.unsqueeze(-1) / (max_gt_press + epsilon)

    pred_combined = torch.cat((pred_pos_norm, pred_press_norm), dim=-1)
    gt_combined = torch.cat((gt_pos_norm, gt_press_norm), dim=-1)
    
    cd_pos_press, _ = chamfer_distance(pred_combined.unsqueeze(0), gt_combined.unsqueeze(0), batch_reduction=None)
    '''
    cd_pos_press = 0
    return mse_pos, cd_pos, cd_pos_press