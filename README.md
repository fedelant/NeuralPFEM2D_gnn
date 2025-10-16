# NeuralPFEM -- Graph Neural Network (GNN) version
NeuralPFEM is a mesh-based neural surrogate model designed for free-surface fluid flows. It combines a Lagrangian neural network model with the efficient remeshing strategy of the Particle Finite Element Method (PFEM).

During training, the model uses the mesh connectivity provided by the simulations to construct the graph. At inference time, the PFEM mesh generation algorithm is employed to build the graph dynamically.

For more details, see our paper:
F. Lanteri, M. Cremonesi, “A mesh-based Graph Neural Network approach for surrogate modeling of Lagrangian free-surface fluid flows,” Computers & Fluids, 2025. https://doi.org/10.1016/j.compfluid.2025.106773
