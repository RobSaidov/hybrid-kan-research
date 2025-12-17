# HybridKAN: Hybrid Kolmogorov-Arnold Networks

<p align="center">
  <img src="figures/hybridkan_architecture.png" width="800" alt="HybridKAN Architecture">
</p>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**HybridKAN** is a novel neural network architecture that combines multiple mathematical basis functions into a unified framework, inspired by the Kolmogorov-Arnold representation theorem. Unlike traditional networks that rely solely on ReLU activations, HybridKAN employs parallel branches of:

- **Gabor wavelets** — Localized, orientation/frequency-selective representations
- **Legendre polynomials** — Orthogonal basis for smooth global structure
- **Chebyshev polynomials** — Optimal polynomial approximation on [-1, 1]
- **Hermite functions** — Gaussian-weighted polynomials for probabilistic modeling
- **Fourier basis** — Periodic function representation
- **ReLU** — Piecewise-linear baseline

Each branch is equipped with **learnable gates** that enable data-driven specialization, and optional **residual skip connections** with learnable weights for improved gradient flow.

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Basis Branches** | Six parallel activation families with learned importance weighting |
| **Learnable Gates** | Per-branch scalar gates (γ) for adaptive branch selection |
| **Residual Connections** | Toggle-able skip connections with learnable gates (α) |
| **Polynomial De-duplication** | Avoids redundant constant/linear terms across polynomial families |
| **CNN Preprocessing** | Optional lightweight CNN for image inputs |
| **Mixed Precision** | AMP support for efficient GPU training |
| **Gate Tracking** | Monitor branch importance evolution during training |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hybridkan.git
cd hybridkan

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA 11.x (optional, for GPU acceleration)

## Quick Start

### Basic Classification (MNIST)

```python
from hybridkan import HybridKAN, Trainer, set_seed
from hybridkan.data import get_mnist_loaders

# Set random seed for reproducibility
set_seed(42)

# Load data
train_loader, test_loader = get_mnist_loaders(
    train_size=60000,
    batch_size=128,
    use_cnn=True,  # Keep image format
)

# Create model with all branches
model = HybridKAN(
    input_dim=784,
    hidden_dims=[256, 128, 64],
    num_classes=10,
    activation_functions='all',
    use_cnn=True,
    cnn_channels=1,
    use_residual=True,
)

# Train
trainer = Trainer(model, train_loader, test_loader)
results = trainer.train()

print(f"Best Accuracy: {results['best_accuracy']:.2f}%")
```

### CIFAR-10 with Custom Branches

```python
from hybridkan import HybridKAN
from hybridkan.data import get_cifar10_loaders

train_loader, test_loader = get_cifar10_loaders(
    train_size=50000,
    batch_size=128,
    augment=True,
)

# Use specific branches only
model = HybridKAN(
    input_dim=3072,
    hidden_dims=[512, 256, 128],
    num_classes=10,
    activation_functions=['fourier', 'legendre', 'relu'],
    use_cnn=True,
    cnn_channels=3,
    cnn_output_dim=384,
)
```

### Toggle Residual Connections

```python
model = HybridKAN(
    input_dim=784,
    hidden_dims=[256, 128],
    num_classes=10,
    use_residual=True,       # Enable skip connections
    residual_every_n=1,      # Skip connection every N blocks
)

# Runtime toggle
model.set_residual_enabled(False)  # Disable residuals
model.set_residual_enabled(True)   # Re-enable residuals
```

### Extract Gate Weights

```python
# Branch gates (per layer, per branch)
branch_gates = model.get_branch_gate_weights()
# {0: {'gabor': 0.21, 'legendre': 0.45, ...}, 1: {...}}

# Residual gates
residual_gates = model.get_residual_gate_weights()
# {'residual_gate_0': 0.34, 'residual_gate_1': 0.28}

# All gates combined
all_gates = model.get_all_gate_weights()
```

## Architecture Details

### HybridKAN Block

Each block performs:

1. **Parallel Branch Computation**: All active branches process the input simultaneously
2. **Per-Branch LayerNorm**: Stabilizes scale across heterogeneous bases
3. **Learnable Gating**: Scalar multiplier (γ) per branch
4. **Concatenation**: Branch outputs are concatenated
5. **Post-Processing**: BatchNorm → GELU → Dropout → Linear Projection
6. **Residual Addition**: Optional gated skip connection (α)

### Gate Initialization

| Branch | Initial Gate (γ) | Rationale |
|--------|------------------|-----------|
| ReLU | 0.5 | Strong baseline |
| Legendre | 0.4 | Smooth polynomial |
| Chebyshev | 0.4 | Optimal approximation |
| Hermite | 0.4 | Gaussian-weighted |
| Fourier | 0.4 | Periodic |
| Gabor | 0.2 | Prevent early domination |

### Polynomial De-duplication

To avoid redundant constant (deg-0) and linear (deg-1) terms across polynomial families:
- **Legendre** keeps degrees 0, 1, 2, ..., 8
- **Chebyshev** starts at degree 2
- **Hermite** starts at degree 2

This is controlled via `dedup_poly_deg01=True` and `keep01_family='legendre'`.

## Experimental Results

### Classification Performance

| Model | MNIST | CIFAR-10 |
|-------|-------|----------|
| ReLU Only | 98.2% | 75.4% |
| Fourier + ReLU | 98.5% | 77.8% |
| **HybridKAN (All)** | **99.1%** | **82.3%** |

### Ablation Study (Leave-One-Out)

| Excluded Branch | CIFAR-10 Accuracy | Δ from All |
|-----------------|-------------------|------------|
| None (All) | 82.3% | — |
| - Gabor | 81.8% | -0.5% |
| - Legendre | 81.2% | -1.1% |
| - Fourier | 80.9% | -1.4% |
| - ReLU | 81.5% | -0.8% |

## Visualization

### Generate Architecture Diagram

```python
from hybridkan.visualization import create_architecture_diagram

fig = create_architecture_diagram(
    hidden_dims=[256, 128, 64],
    branches=['gabor', 'legendre', 'chebyshev', 'hermite', 'fourier', 'relu'],
    use_residual=True,
    show_gates=True,
    output_path='architecture.pdf',
    dpi=1200,  # Publication quality
)
```

### Plot Training Curves

```python
from hybridkan.visualization import plot_training_curves, plot_gate_evolution

# Training curves
plot_training_curves('results/logs/experiment_metrics.csv')

# Gate evolution
plot_gate_evolution('results/logs/experiment_gates.json')
```

## Project Structure

```
hybridkan/
├── hybridkan/
│   ├── __init__.py          # Package exports
│   ├── activations.py       # Basis function implementations
│   ├── model.py             # HybridKAN architecture
│   ├── trainer.py           # Training infrastructure
│   ├── data.py              # Data loading utilities
│   ├── utils.py             # General utilities
│   └── visualization.py     # Plotting functions
├── notebooks/
│   └── HybridKAN_Demo.ipynb # Interactive tutorial
├── figures/
│   └── *.pdf               # Architecture diagrams
├── setup.py
├── requirements.txt
└── README.md
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{hybridkan2024,
  title={HybridKAN: Hybrid Kolmogorov-Arnold Networks with Multi-Basis Activation Functions},
  author={Rob},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Research supervised by Dr. Bandari (San Francisco Bay University)
- Inspired by the Kolmogorov-Arnold representation theorem
- PyTorch team for the deep learning framework
