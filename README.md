# NeuralPFEM - Graph Neural Network (GNN) version
NeuralPFEM is a mesh-based neural surrogate model designed for free-surface fluid flows. It combines a Lagrangian neural network model with the efficient remeshing strategy of the Particle Finite Element Method (PFEM).

In this version, the core of the architecture is a GNN module.

During training, the model uses the mesh connectivity provided by the simulations to construct the graph. At inference time, the PFEM mesh generation algorithm is employed to build the graph dynamically.

---

## Cite
For more details and to cite:
```bibtex
@article{LANTERI2025106773,
title = {A mesh-based Graph Neural Network approach for surrogate modeling of Lagrangian free surface fluid flows},
journal = {Computers & Fluids},
volume = {301},
pages = {106773},
year = {2025},
issn = {0045-7930},
doi = {https://doi.org/10.1016/j.compfluid.2025.106773},
url = {https://www.sciencedirect.com/science/article/pii/S0045793025002336},
author = {Federico Lanteri and Massimiliano Cremonesi},
}
```
