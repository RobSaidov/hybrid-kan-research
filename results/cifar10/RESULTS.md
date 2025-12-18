# HybridKAN Experiment Results - CIFAR-10

Generated: 2025-12-17

Device: NVIDIA GeForce RTX 3060 Laptop GPU

## Summary

### Main Comparison

| Model | Accuracy | Parameters | Time |
|-------|----------|------------|------|
| **relu_only** | **86.15%** | 364,240 | ~40 min |
| all_branches | 85.63% | 1,355,947 | ~2 hr |
| all_no_residual | 85.17% | 1,314,792 | ~2 hr |

### Ablation Study (Leave-One-Out)

| Excluded Branch | Accuracy | Δ from All |
|-----------------|----------|------------|
| None (All) | 85.63% | — |
| - Gabor | 85.76% | +0.13% |
| - Legendre | 85.56% | -0.07% |
| - Chebyshev | 85.37% | -0.26% |
| - Hermite | 85.44% | -0.19% |
| - Fourier | 85.39% | -0.24% |
| - ReLU | 85.50% | -0.13% |

## Key Findings

1. **ReLU-only baseline is surprisingly strong** at 86.15%, showing the CNN preprocessor is doing heavy lifting
2. **Residual connections help**: All branches (85.63%) vs No residual (85.17%) = +0.46%
3. **Gabor branch may be redundant**: Removing it actually *improves* accuracy slightly (+0.13%)
4. **Most impactful branches**: Chebyshev (-0.26%) and Fourier (-0.24%) contribute most

## Gate Analysis (All Branches Model)

The learned gate weights after training show interesting patterns:

**Layer 0 (Early)**: Gabor dominates (1.10), others suppressed
**Layer 1 (Middle)**: Gabor (1.07) and ReLU (0.83) active
**Layer 2 (Late)**: More balanced - ReLU (0.81), Hermite (0.79), Gabor (0.75)

This suggests the network learns to use Gabor wavelets early for feature detection, then transitions to ReLU and polynomial bases for classification.

## Best Configuration

- **Model**: relu_only (simple CNN + ReLU MLP)
- **Accuracy**: 86.15%
- **Best Epoch**: 56
- **Training Time**: ~40 minutes
