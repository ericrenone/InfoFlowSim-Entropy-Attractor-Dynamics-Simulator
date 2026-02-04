# Thermodynamic Information Flow (TIF) Simulator

Minimal, self-contained Python simulator demonstrating annealing-style optimization on the probability simplex using:

- Fisher–Rao natural gradient descent  
- Time-dependent free energy functional  
- Controlled exploration–exploitation transition  
- Stochastic dynamics (Langevin-like noise)

The system starts from a uniform distribution and converges toward a low-entropy attractor while respecting thermodynamic-inspired constraints.

## Features

- Annealing of exploration (α) and exploitation (β) coefficients  
- Soft capacity constraint schedule  
- Shannon entropy, KL divergence, free energy, Fisher information tracked  
- Entropy production rate estimation  
- Clean validation suite (convergence, equilibrium, metric consistency, …)  
- Publication-quality multi-panel visualization

