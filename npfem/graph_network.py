from typing import List
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:
  """Build a MultiLayer Perceptron.

  Args:
    input_size: Size of input layer.
    layer_sizes: An array of input size for each hidden layer.
    output_size: Size of the output layer.
    output_activation: Activation function for the output layer.
    activation: Activation function for the hidden layers.

  Returns:
    mlp: An MLP sequential container.
  """
  # Size of each layer
  layer_sizes = [input_size] + hidden_layer_sizes
  if output_size:
    layer_sizes.append(output_size)

  # Number of layers
  nlayers = len(layer_sizes) - 1

  # Create a list of activation functions and
  # set the last element to output activation function
  act = [activation for i in range(nlayers)]
  act[-1] = output_activation

  # Create a torch sequential container
  mlp = nn.Sequential()
  for i in range(nlayers):
    mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
                                             layer_sizes[i + 1]))
    mlp.add_module("Act-" + str(i), act[i]())

  return mlp


class Encoder(nn.Module):

  def __init__(
          self,
          n_node_in: int,
          n_edge_in: int,
          node_latent_dim: int,
          edge_latent_dim: int,
          mlp_hidden_dim: int,
          nmlp_layers: int,
  ):
    super(Encoder, self).__init__()
    # Encode node features as an MLP
    self.gamman = nn.Parameter(torch.tensor(1.0))
    self.gammam = nn.Parameter(torch.tensor(1.0))
    self.gammae = nn.Parameter(torch.tensor(1.0))
    self.node_fn = nn.Sequential(*[build_mlp(n_node_in,
                                             [node_latent_dim
                                              for _ in range(nmlp_layers)],
                                             node_latent_dim),
                                   nn.LayerNorm(node_latent_dim)])
    # Encode edge features as an MLP
    self.edge_fn = nn.Sequential(*[build_mlp(n_edge_in,
                                             [edge_latent_dim
                                              for _ in range(nmlp_layers)],
                                             edge_latent_dim),
                                   nn.LayerNorm(edge_latent_dim)])

  def forward(
          self,
          x: torch.tensor,
          mat_features: torch.tensor,
          edge_features: torch.tensor):
    """The forward hook runs when the Encoder class is instantiated

    Args:
      x: Particle state representation as a torch tensor with shape
        (nparticles, nnode_input_features)
      edge_features: Edge features as a torch tensor with shape
        (nparticles, nedge_input_features)

    """
    x = self.gamman * x
    mat_features = self.gammam * mat_features
    edge_features = self.gammae * edge_features
    x_norm = torch.cat([x, mat_features], dim=-1)
    return self.node_fn(x_norm), self.edge_fn(edge_features)    


class InteractionNetwork(MessagePassing):
  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):
    """InteractionNetwork derived from torch_geometric MessagePassing class

    Args:
      nnode_in: Number of node inputs (latent dimension of size 128).
      nnode_out: Number of node outputs (latent dimension of size 128).
      nedge_in: Number of edge inputs (latent dimension of size 128).
      nedge_out: Number of edge output features (latent dimension of size 128).
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

    """
    # Aggregate features from neighbors
    super(InteractionNetwork, self).__init__(aggr='add')
    # Node MLP
    self.node_fn = nn.Sequential(*[build_mlp(nnode_in + nedge_out,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nnode_out),
                                   nn.LayerNorm(nnode_out)])
    # Edge MLP
    self.edge_fn = nn.Sequential(*[build_mlp(nnode_in + nnode_in + nedge_in,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nedge_out),
                                   nn.LayerNorm(nedge_out)])

  def forward(self,
              x: torch.tensor,
              edge_index: torch.tensor,
              edge_features: torch.tensor):
    """The forward hook runs when the InteractionNetwork class is instantiated

    Args:
      x: Particle state representation as a torch tensor with shape
        (nparticles, nnode_input_features)
      edge_index: A torch tensor list of source and target nodes with shape
        (2, nedges)
      edge_features: Edge features as a torch tensor with shape
        (nedges, nedge_in=latent_dim of 128)

    Returns:
      tuple: Updated node and edge features
    """
    # Save particle state and edge features
    x_residual = x
    edge_features_residual = edge_features
    # Start propagating messages.
    # Takes in the edge indices and all additional data which is needed to
    # construct messages and to update node embeddings.
    # Call PyG propagate() method:
    # 1. Message phase - compute messages for each edge
    # 2. Aggregate phase - aggregate messages for each node
    # 3. Update phase - updates only the node features
    # Update uses the message from step 1 and any original arguments passed to 
    # propagate() to update the node embeddings. This is why we need to store
    # the updated edge features to return them from the update() method.
    x, edge_features = self.propagate(
        edge_index=edge_index, x=x, edge_features=edge_features)

    return x + x_residual, edge_features + edge_features_residual

  def message(self,
              x_i: torch.tensor,
              x_j: torch.tensor,
              edge_features: torch.tensor) -> torch.tensor:
    """Constructs message from j to i of edge :math:`e_{i, j}`. Tensors :obj:`x`
    passed to :meth:`propagate` can be mapped to the respective nodes :math:`i`
    and :math:`j` by appending :obj:`_i` or :obj:`_j` to the variable name,
    i.e., :obj:`x_i` and :obj:`x_j`.

    Args:
      x_i: Particle state representation as a torch tensor with shape
        (nparticles, nnode_in=latent_dim of 128) at node i
      x_j: Particle state representation as a torch tensor with shape
        (nparticles, nnode_in=latent_dim of 128) at node j
      edge_features: Edge features as a torch tensor with shape
        (nedges, nedge_in=latent_dim of 128)

    """
    # Concat edge features with a final shape of [nedges, latent_dim*3]
    edge_features = torch.cat([x_i, x_j, edge_features], dim=-1)
    self._edge_features = self.edge_fn(edge_features)  # Create and store
    return self._edge_features  # This gets passed to aggregate()

  def update(self,
             x_updated: torch.tensor,
             x: torch.tensor,
             edge_features: torch.tensor):
    """Update the particle state representation

    Args:
      x: Particle state representation as a torch tensor with shape 
        (nparticles, nnode_in=latent_dim of 128)
      x_updated: Updated particle state representation as a torch tensor with 
        shape (nparticles, nnode_in=latent_dim of 128)
      edge_features: Edge features as a torch tensor with shape 
        (nedges, nedge_out=latent_dim of 128)

    Returns:
      tuple: Updated node and edge features
    """
    # Concat node features with a final shape of
    # [nparticles, latent_dim (or nnode_in) *2]
    # This gets called later, after message() and aggregate()
    # Update modified from MessagePassing takes the output of aggregation
    # as first argument and any argument which was initially passed to
    # propagate hence we need to return the stored value of edge_features
    x_updated = torch.cat([x_updated, x], dim=-1)
    x_updated = self.node_fn(x_updated)
    return x_updated, self._edge_features


class Processor(MessagePassing):
  """The Processor: :math: `\mathcal{G} \rightarrow \mathcal{G}` computes 
  interactions among nodes via :math: `M` steps of learned message-passing, to 
  generate a sequence of updated latent graphs, :math: `G = (G_1 , ..., G_M )`, 
  where :math: `G^{m+1| = GN^{m+1} (G^m )`. It returns the final graph, 
  :math: `G^M = PROCESSOR(G^0)`. Message-passing allows information to 
  propagate and constraints to be respected: the number of message-passing 
  steps required will likely scale with the complexity of the interactions.

  """

  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmessage_passing_steps: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):
    """Processor derived from torch_geometric MessagePassing class. The 
    processor uses a stack of :math: `M GNs` (where :math: `M` is a 
    hyperparameter) with identical structure, MLPs as internal edge and node 
    update functions, and either shared or unshared parameters. We use GNs 
    without global features or global updates (i.e., an interaction network), 
    and with a residual connections between the input and output latent node 
    and edge attributes.

    Args:
      nnode_in: Number of node inputs (latent dimension of size 128).
      nnode_out: Number of node outputs (latent dimension of size 128).
      nedge_in: Number of edge inputs (latent dimension of size 128).
      nedge_out: Number of edge output features (latent dimension of size 128).
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

    """
    super(Processor, self).__init__(aggr='max')
    # Create a stack of M Graph Networks GNs.
    self.gnn_stacks = nn.ModuleList([
        InteractionNetwork(
            nnode_in=nnode_in,
            nnode_out=nnode_out,
            nedge_in=nedge_in,
            nedge_out=nedge_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        ) for _ in range(nmessage_passing_steps)])

  def forward(self,
              x: torch.tensor,
              edge_index: torch.tensor,
              edge_features: torch.tensor):
    """The forward hook runs through GNN stacks when class is instantiated. 

    Args:
      x: Particle state representation as a torch tensor with shape 
        (nparticles, latent_dim)
      edge_index: A torch tensor list of source and target nodes with shape 
        (2, nedges)
      edge_features: Edge features as a torch tensor with shape 
        (nparticles, latent_dim)

    """
    for gnn in self.gnn_stacks:
      x, edge_features = gnn(x, edge_index, edge_features)
    return x, edge_features

# ===============================================================
# ========================= DECODER =============================
# ===============================================================

class Decoder(nn.Module):

    def __init__(
        self,
        n_in_features: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        output_dim_vel: int = 2,
        output_dim_press: int = 1
    ):
        super().__init__()

        self.vel_fn = build_mlp(
            input_size=n_in_features,
            hidden_layer_sizes=[mlp_hidden_dim] * nmlp_layers,
            output_size=output_dim_vel,
            output_activation=nn.Identity,
        )

        self.press_fn = build_mlp(
            input_size=n_in_features,
            hidden_layer_sizes=[mlp_hidden_dim] * nmlp_layers,
            output_size=output_dim_press,
            output_activation=nn.Identity,
        )

    def forward(self, x):
        return self.vel_fn(x), self.press_fn(x)
