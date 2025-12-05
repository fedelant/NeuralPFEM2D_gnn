"""
DataLoader for NeuralPFEM.

Provides PyTorch Dataset and DataLoader utilities for:
- Training on single timesteps (TrainingDataset)
- Evaluation/testing on full trajectories (PredictionDataset)

Supports simulations stored in HDF5.
"""

# Dependencies
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import h5py


# --- Dataset Classes ---

class TrainingDataset(Dataset):
    """
    PyTorch Dataset for supervised training on single timesteps.

    HDF5 structure (group "outputX"):
        - position: [total_nodes, dims] concatenated across timesteps
        - velocity: [total_nodes, dims] concatenated across timesteps
        - pressure: [total_nodes, 1] concatenated across timesteps
        - offsets: [T+1] cumulative node counts for splitting timesteps
        - material_properties: [num_props] material properties
    """

    def __init__(self, path, input_length_sequence: int):
        super().__init__()
        if input_length_sequence < 1:
            raise ValueError("input_length_sequence must be >= 1")

        self._input_length_sequence = input_length_sequence
        self._path = path

        # Index all trajectories and count valid training samples
        with h5py.File(path, "r") as f:
            self._keys = list(f.keys())
            self._data_lengths = []
            for key in self._keys:
                num_steps = len(f[key]["offsets"]) - 1
                valid_samples = max(0, num_steps - input_length_sequence)
                if valid_samples == 0:
                    print(f"⚠️ Trajectory {key} skipped (only {num_steps} steps)")
                self._data_lengths.append(valid_samples)

        self._length = sum(self._data_lengths)
        if self._length == 0:
            raise ValueError(
                f"No valid samples in {path} with input_length_sequence={input_length_sequence}"
            )

        self._cumulative_lengths = np.cumsum([0] + self._data_lengths)
        print(
            f"✅ TrainingDataset initialized with {self._length} samples "
            f"from {len(self._keys)} trajectories"
        )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if not 0 <= idx < self._length:
            raise IndexError(f"Index {idx} out of bounds (length {self._length})")

        # Locate trajectory and timestep
        traj_idx = np.searchsorted(self._cumulative_lengths, idx, side="right") - 1
        local_idx = idx - self._cumulative_lengths[traj_idx]
        time_idx = self._input_length_sequence + local_idx
        key = self._keys[traj_idx]

        with h5py.File(self._path, "r") as f:
            grp = f[key]
            offsets = grp["offsets"][:]
            offsets_cells = grp["cell_offsets"][:]

            def slice_timestep(arr, t):
                return arr[offsets[t]:offsets[t + 1]]
            
            def slice_timestep_cells(arr, t):
                return arr[offsets_cells[t]:offsets_cells[t + 1]]

            # Node positions
            node_positions = torch.tensor(
                slice_timestep(grp["position"], time_idx), dtype=torch.float32
            )
            num_nodes, dims = node_positions.shape

            # Velocity history [N, seq, dims]
            vel_hist = [
                slice_timestep(grp["velocity"], t)
                for t in range(time_idx - self._input_length_sequence, time_idx)
            ]
            input_velocity_hist = torch.tensor(
                np.stack(vel_hist, axis=0), dtype=torch.float32
            ).permute(1, 0, 2)

            # Pressure history [N, seq]
            press_hist = [
                slice_timestep(grp["pressure"], t).squeeze(-1)
                for t in range(time_idx - self._input_length_sequence, time_idx)
            ]
            input_pressure_hist = torch.tensor(
                np.stack(press_hist, axis=0), dtype=torch.float32
            ).permute(1, 0)

            cells = torch.tensor(
                slice_timestep_cells(grp["cells"], time_idx), dtype=torch.int32
            )

            # Targets
            target_velocity = torch.tensor(
                slice_timestep(grp["velocity"], time_idx), dtype=torch.float32
            )
            target_pressure = torch.tensor(
                slice_timestep(grp["pressure"], time_idx), dtype=torch.float32
            )
            y_target = torch.cat([target_velocity, target_pressure], dim=-1)

            # Unified material property vector
            props = torch.tensor(grp["material_properties"][:], dtype=torch.float32)  # [num_props]
            props_star = torch.zeros(2)
            props_star[0] = props[1] / (props[0] * 9.81 * 0.3)
            props_star[1] = props[2] / 100
            material_props = props_star#.repeat(num_nodes, 1)  # [N, num_props]

        return Data(
            dims=dims,
            position=node_positions,
            velocity=input_velocity_hist,
            pressure=input_pressure_hist,
            cells=cells,
            y=y_target,
            material_properties=material_props,  # [N, num_props]
            num_nodes=num_nodes,
        )


class PredictionDataset(Dataset):
    """
    PyTorch Dataset for evaluation/testing on full trajectories.

    HDF5 structure (group "outputX"):
        - position: [total_nodes, dims] concatenated across timesteps
        - velocity: [total_nodes, dims] concatenated across timesteps
        - pressure: [total_nodes, 1] concatenated across timesteps
        - offsets: [T+1] cumulative node counts for splitting timesteps
        - material_properties: [num_props] material properties
    """

    def __init__(self, path):
        super().__init__()
        self._path = path
        with h5py.File(path, "r") as f:
            self._keys = list(f.keys())
        if not self._keys:
            raise ValueError(f"No trajectories found in {path}")
        print(f"✅ PredictionDataset initialized with {len(self._keys)} trajectories")

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        if not 0 <= idx < len(self._keys):
            raise IndexError(f"Index {idx} out of bounds")

        key = self._keys[idx]
        with h5py.File(self._path, "r") as f:
            grp = f[key]
            offsets = grp["offsets"][:]
            T = len(offsets) - 1

            positions, velocities, pressures = [], [], []
            for t in range(T):
                positions.append(grp["position"][offsets[t]:offsets[t + 1]])
                velocities.append(grp["velocity"][offsets[t]:offsets[t + 1]])
                pressures.append(grp["pressure"][offsets[t]:offsets[t + 1]].squeeze(-1))

            position_tensor = torch.tensor(np.stack(positions, axis=0), dtype=torch.float32)
            velocity_tensor = torch.tensor(np.stack(velocities, axis=0), dtype=torch.float32)
            pressure_tensor = torch.tensor(np.stack(pressures, axis=0), dtype=torch.float32)

            dims = position_tensor.shape[-1]
            num_nodes = position_tensor.shape[1]

            # Unified material property vector
            props = torch.tensor(grp["material_properties"][:], dtype=torch.float32)  # [num_props]
            props_star = torch.zeros(2)
            props_star[0] = props[1] / (props[0] * 9.81 * 0.3)
            props_star[1] = props[2] / 100
            material_props = props_star

            angles = torch.tensor(grp["angle"][()], dtype=torch.float32)
        return Data(
            dims=dims,
            position=position_tensor,
            velocity=velocity_tensor,
            pressure=pressure_tensor,
            material_properties=material_props,
            angles=angles,
        ), key

# --- DataLoader Utility Functions ---

def get_training_data_loader(
    path,
    input_length_sequence: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    **kwargs,
):
    """Builds a DataLoader for training (single timesteps)."""
    dataset = TrainingDataset(path, input_length_sequence)
    return PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def get_prediction_data_loader(
    path,
    num_workers: int = 0,
    pin_memory: bool = True,
    **kwargs,
):
    """Builds a DataLoader for prediction/evaluation (full trajectories)."""
    dataset = PredictionDataset(path)
    return TorchDataLoader(
        dataset,
        batch_size=None,  # one trajectory per iteration
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )