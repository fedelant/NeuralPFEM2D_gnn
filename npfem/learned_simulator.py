"""

"""

# === Dependencies ===
from networkx import triangles
from npfem import mesh
import torch
import torch.nn as nn
import numpy as np
from npfem.global_module import g 

# === Local Application/Library Specific Imports ===
from npfem import graph_network # Assuming this contains Encoder, Processor, Decoder

class LearnedSimulator(nn.Module):
    """
    Learned Simulator with:
        Encoder (node + edge features)
        Processor (MPNN with M message passing steps)
        Decoder (velocity + pressure)
    """

    def __init__(
        self,
        nnode_in: int,
        nedge_in: int,
        latent_dim: int,
        mlp_hidden_dim: int,
        nmlp_layers: int,
        mp_steps: int,
        normalization_stats: dict,
        dt: float,
        h: float,
        spatial_norm_weight: float,
        vel_norm_weight: float,
        press_norm_weight: float,
        remesh_freq: int,
        device: str = "cuda",
    ):
        super().__init__()

        # --- Device and Parameters ---
        self._device = torch.device(device)
        self.dim = 2
        self.dt = dt
        self._spatial_norm_weight = spatial_norm_weight
        self._vel_norm_weight = vel_norm_weight
        self._press_norm_weight = press_norm_weight
        self._remesh_freq = remesh_freq
        self._h = h

        # --- Normalization Stats ---
        self._normalization_stats = self._move_stats_to_device(
            normalization_stats, self._device
        )
        if "velocity" not in self._normalization_stats or "pressure" not in self._normalization_stats:
            raise ValueError("normalization_stats must contain 'velocity' and 'pressure' keys.")

        self._encode = graph_network.Encoder(
            n_node_in=nnode_in,
            n_edge_in=nedge_in,
            node_latent_dim=latent_dim,
            edge_latent_dim=latent_dim,  # symmetric
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        ).to(self._device)

        self._process = graph_network.Processor(
            nnode_in=latent_dim,
            nnode_out=latent_dim,
            nedge_in=latent_dim,
            nedge_out=latent_dim,
            nmessage_passing_steps=mp_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        ).to(self._device)
    # Create a stack of M Graph Networks GNs.

        self._decode = graph_network.Decoder(
            n_in_features=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            output_dim_vel=self.dim,
        ).to(self._device)

    # --- Mesher Handling ---

    def initialize_mesher(self, mesh_size, alpha, refine, rmin_factor, rmax_factor):
        """
        """
        self._mesher = mesh.GmshMesher(self.dim, mesh_size, alpha, refine, rmin_factor, rmax_factor) # Initialize the mesher
    
    def finalize(self):
        """
        """
        self._mesher.finalize_mesher()

    def first_mesh(self, positions: torch.Tensor):
        """Generates the initial mesh and returns cells and positions."""
        cells, pos, tags = self._mesher.first_mesh(positions)
        return cells, pos, tags
    
    def correct_history(
        self,
        prev_positions: torch.Tensor,       # [N_old, D]
        current_position: torch.Tensor,     # [N_new, D]
        prev_velocities: torch.Tensor,      # [N_old, C, D]
        prev_pressures: torch.Tensor,       # [N_old, C]
        cells_at_t: np.ndarray,             # [M, K]  (K=3 for triangles, K=4 for tets)
        prev_tags: np.ndarray,              # [N_old]
        new_tags: np.ndarray                # [N_new]
    ):
        device = prev_velocities.device
        N_new, D = current_position.shape
        _, C, _ = prev_velocities.shape

        corrected_velocities = torch.zeros((N_new, C, D), device=device, dtype=prev_velocities.dtype)
        corrected_pressures  = torch.zeros((N_new, C),   device=device, dtype=prev_pressures.dtype)

        # --- Fast tag lookup ---
        max_tag = max(prev_tags.max(), new_tags.max())
        tag_to_idx = -np.ones(max_tag + 1, dtype=np.int64)
        tag_to_idx[prev_tags] = np.arange(len(prev_tags))

        # --- 1. Surviving nodes (vectorized) ---
        surviving_mask = np.isin(new_tags, prev_tags)
        surviving_idx_new = np.nonzero(surviving_mask)[0]
        surviving_idx_old = tag_to_idx[new_tags[surviving_mask]]

        if len(surviving_idx_new) > 0:
            corrected_velocities[surviving_idx_new] = prev_velocities[surviving_idx_old]
            corrected_pressures[surviving_idx_new]  = prev_pressures[surviving_idx_old]

        # --- 2. New nodes (non-surviving) ---
        new_idx = np.nonzero(~surviving_mask)[0]
        if len(new_idx) > 0:
            pts = current_position[new_idx]

            # Fallback nearest neighbor (vectorized)
            nearest_idx_all = torch.argmin(torch.cdist(pts, prev_positions), dim=1)
            corrected_velocities[new_idx] = prev_velocities[nearest_idx_all]
            corrected_pressures[new_idx]  = prev_pressures[nearest_idx_all]

            elem_idx, bary = self._mesher.find_elements(
                current_position[new_idx], prev_positions, torch.as_tensor(cells_at_t, device=device)
            )

            inside_mask = elem_idx >= 0
            if inside_mask.any():
                inside_new_idx = torch.as_tensor(new_idx, device=device)[inside_mask]
                elems_hit = elem_idx[inside_mask]
                bary_hit = bary[inside_mask]

                elem_nodes = torch.as_tensor(cells_at_t, device=device)[elems_hit]  # [N_inside, K]

                # Interpolation (batched)
                vel_interp = torch.einsum("nk,nkcd->ncd", bary_hit, prev_velocities[elem_nodes])
                press_interp = torch.einsum("nk,nkc->nc", bary_hit, prev_pressures[elem_nodes])

                # Cast to match destination dtype
                vel_interp = vel_interp.to(corrected_velocities.dtype)
                press_interp = press_interp.to(corrected_pressures.dtype)

                corrected_velocities[inside_new_idx] = vel_interp
                corrected_pressures[inside_new_idx]  = press_interp

        return corrected_velocities, corrected_pressures
    
    # --- Utils ---

    def _move_stats_to_device(self, stats, device):
        """Moves normalization statistics tensors to the specified device."""
        for key in stats:
            for stat_name, tensor in stats[key].items():
                 if isinstance(tensor, torch.Tensor):
                     stats[key][stat_name] = tensor.to(device)
        return stats
    
    def _decode_predictions(
            self,
            normalized_vel: torch.Tensor,
            normalized_press: torch.Tensor
        ):
        """Denormalizes predicted velocity and pressure."""
        # Velocity Denormalization
        vel_stats = self._normalization_stats["velocity"]
        pred_vel = normalized_vel * vel_stats["std"] + vel_stats["mean"]

        # Pressure Denormalization
        press_stats = self._normalization_stats["pressure"]
        # Ensure normalized_press is [N] before denormalizing if it's [N, 1]
        pred_press = normalized_press.squeeze(-1) * press_stats["std"] + press_stats["mean"]

        return pred_vel, pred_press.to(torch.float32) # Ensure correct dtype
    
    # --- Model Saving/Loading ---

    def save(self, path: str = "model.pt"):
        """
        Saves the model's state dictionary to the specified path.

        Args:
            path: File path to save the model state.
        """
        print(f"Saving model state to {path}...")
        torch.save(self.state_dict(), path)
        print("Model state saved.")

    def load(self, path: str):
        """
        Loads the model's state dictionary from the specified path.

        Loads onto CPU first, then the model should be moved to the target device.

        Args:
            path: File path from which to load the model state.
        """
        print(f"Loading model state from {path}...")
        # Load state dict onto CPU first to avoid GPU memory issues if loading a GPU-trained model on a machine with less GPU RAM
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(state_dict)
        print("Model state loaded.")

    def cells_to_edges(self, triangles):
        """
        Convert triangle cells [ncells, 3] to edge_index [2, n_edges] efficiently.

        Args:
            triangles: LongTensor of shape [ncells, 3]

        Returns:
            edge_index: LongTensor of shape [2, n_edges] with unique edges
        """
        # Extract all edges (3 per triangle)
        e1 = triangles[:, [0, 1]]
        e2 = triangles[:, [1, 2]]
        e3 = triangles[:, [2, 0]]

        # Stack undirected edges (forward and reverse)
        edges = torch.cat([
            e1, e2, e3,
            e1.flip(1), e2.flip(1), e3.flip(1)   # reversed edges
        ], dim=0)

        # Remove duplicates
        edges = torch.unique(edges, dim=0)

        # PyG expects [2, num_edges]
        return edges.t().contiguous().long()

    # --- Feature preparation ---

    def preprocessor(
        self,
        most_recent_position: torch.Tensor,
        velocity_sequence: torch.Tensor,
        pressure_sequence: torch.Tensor,
        material_properties: torch.Tensor,
        edge_index: torch.Tensor
    ):
        """
        Construct node features for model input.

        Args:
            most_recent_position (torch.Tensor): Node positions at t, shape [N, D].
            velocity_sequence (torch.Tensor): Velocity history, shape [N, C, D].
            pressure_sequence (torch.Tensor): Pressure history, shape [N, C].
            material_properties (torch.Tensor): Material properties, shape [N, P].

        Returns:
            torch.Tensor: Node features, shape [N, F].
        """
        
        n_particles = most_recent_position.shape[0]
        node_features_list = []

        vel_stats = self._normalization_stats["velocity"]
        press_stats = self._normalization_stats["pressure"]

        # 1. Flatten velocity + pressure history
        
        scaled_vel = (velocity_sequence - vel_stats["mean"]) / vel_stats["std"] #* self._vel_norm_weight
        scaled_press = (pressure_sequence - press_stats["mean"]) / press_stats["std"] #* self._press_norm_weight
        velpress_hist = torch.cat([scaled_vel, scaled_press.unsqueeze(-1)], dim=2)
        node_features_list.append(velpress_hist.view(n_particles, -1))

        mat_features_list = []
        mat_features_list.append(material_properties.repeat(n_particles, 1))  # [N, P]

        # Initialize edge features.
        edge_features_list = []
        # Relative displacement and distances normalized to radius with shape (n_edges, 2)
        normalized_relative_displacements = (
            (most_recent_position[edge_index[0], :] - most_recent_position[edge_index[1], :])
            / self._h 
        )
        # Relative distance between 2 particles with shape (n_edges, 1)
        normalized_relative_distances = torch.norm(
            normalized_relative_displacements, dim=-1, keepdim=True)

        # Edge features has a final shape of (n_edges, 3)
        edge_features_list.append(normalized_relative_displacements)
        edge_features_list.append(normalized_relative_distances)
        # 2. Append material properties
        #edge_features_list.append(material_properties.repeat(edge_index.shape[1], 1))  # [N, P]

        return (torch.cat(node_features_list, dim=-1),
                torch.cat(mat_features_list, dim=-1),
                torch.cat(edge_features_list, dim=-1))

    
    # --- Core Methods ---

    # --- Prediction step ---
    def learned_update(
        self,
    ):

        refine_freq = 2000
        addrem_freq = 2

        bd_mask1 = g.current_position[:, -1] <= -torch.tan(g.angles)*g.current_position[:,0]
        bd_mask2 = g.current_position[:, -1] <= 0

        if g.step % addrem_freq == 0 and g.step > 0:
            g.coords = g.next_position.detach().cpu().numpy()
            g.boundv = bd_mask1 | bd_mask2
            g.boundv = g.boundv.unsqueeze(1).repeat(1, 2).cpu().detach().cpu().numpy()
            g.free_surf = np.zeros_like(g.boundv[:,0])
            g.euler = np.zeros_like(g.boundv[:,0])
        
            self._mesher.compute_node_neighbors(g.cells, g.next_position.shape[0])
            self._mesher.addrem_nodes_local()
            g.next_position = torch.from_numpy(g.coords.astype(np.float32)).to(self._device)
        
        if g.step == 0 or g.step % self._remesh_freq == 0:
            old_position = g.current_position
            old_cells = g.cells
            old_tags = g.tags
            g.cells, position_np, g.tags = self._mesher.generate_mesh(
                g.current_position.detach().cpu().numpy(),
                g.tags,
                refine=(g.step % refine_freq == 0)
            )
            g.current_position = torch.from_numpy(position_np.astype(np.float32)).to(self._device)
            g.previous_velocity, g.previous_pressure = self.correct_history(
                old_position, g.current_position,
                g.previous_velocity, g.previous_pressure,
                old_cells, old_tags, g.tags
            )
            bd_mask1 = g.current_position[:, -1] <= -torch.tan(g.angles)*g.current_position[:,0]
            bd_mask2 = g.current_position[:, -1] <= 0

        # --- 2. Build node features (pos + history + materials) ---
        edge_index = self.cells_to_edges(torch.from_numpy(g.cells).to(self._device))
        node_features, mat_features, edge_features = self.preprocessor(
            most_recent_position=g.current_position,
            velocity_sequence=g.previous_velocity,
            pressure_sequence=g.previous_pressure,
            material_properties=g.material_properties,
            edge_index=edge_index
        )

        # Encode, Process, Decode
        node_latent, edge_latent = self._encode(node_features, mat_features, edge_features)
        node_latent, _ = self._process(node_latent, edge_index, edge_latent)
        norm_pred_vel, norm_pred_press = self._decode(node_latent)

        # --- 4. Denormalize predictions ---
        g.current_velocity, g.current_pressure = self._decode_predictions(norm_pred_vel, norm_pred_press)

        # --- 5. Apply boundary conditions ---
        g.current_velocity[bd_mask1] = 0.0
        g.current_velocity[bd_mask2] = 0.0
        g.current_position[bd_mask1, -1] = -torch.tan(g.angles)*g.current_position[bd_mask1,0]
        g.current_position[bd_mask2, -1] = 0

        # --- 6. Advance to next step ---
        g.next_position = g.current_position + g.current_velocity * self.dt
    
    # --- Training Step ---

    def forward(
        self,
        current_position: torch.Tensor,    # [N, D]
        velocity_sequence: torch.Tensor,   # [N, SeqLen, D]
        pressure_sequence: torch.Tensor,   # [N, SeqLen]
        material_properties: torch.Tensor, # [N, P]
        cells: torch.Tensor          # [M, K]
    ):
        """
        Forward pass: predicts normalized velocity and pressure.

        Returns:
            pred_norm_vel (torch.Tensor): Predicted normalized velocity [N, D].
            pred_norm_press (torch.Tensor): Predicted normalized pressure [N].
        """
        edge_index = self.cells_to_edges(cells.to(current_position.device))
        # Preprocess features
        node_features, mat_features, edge_features = self.preprocessor(
            most_recent_position=current_position,
            velocity_sequence=velocity_sequence,
            pressure_sequence=pressure_sequence,
            material_properties=material_properties,
            edge_index=edge_index.to(current_position.device)
        )

        pos_stats = self._normalization_stats["position"]

        # Encode → Process → Decode
        node_latent, edge_latent = self._encode(node_features, mat_features,edge_features)
        node_latent, _ = self._process(node_latent, edge_index, edge_latent)
        pred_norm_vel, pred_norm_press = self._decode(node_latent)
    
        return pred_norm_vel, pred_norm_press.squeeze(-1)