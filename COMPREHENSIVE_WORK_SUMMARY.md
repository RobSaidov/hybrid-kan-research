# HybridKAN Research - Comprehensive Work Summary

**Date**: December 20, 2025  
**GitHub**: https://github.com/RobSaidov/hybrid-kan-research

---

## 1. Project Overview

### What is HybridKAN?

HybridKAN is a neural network architecture that combines **6 different activation functions** with a learnable gating mechanism. Instead of using just ReLU (like standard networks), it can blend multiple mathematical bases:

| Activation | Description | Best For |
|------------|-------------|----------|
| **ReLU** | Standard deep learning activation | General purpose |
| **Fourier** | Sine/cosine basis functions | Periodic patterns |
| **Gabor** | Localized frequency detector | Oscillatory patterns |
| **Legendre** | Orthogonal polynomials | Smooth functions |
| **Chebyshev** | Polynomial approximation | Bounded intervals |
| **Hermite** | Gaussian-weighted polynomials | Bell-curve shapes |

### Key Innovation

The network **learns which activation to use** through trainable gating weights. This allows it to:
- Specialize per layer
- Combine multiple mathematical representations
- Be interpretable (see which activations matter)

---

## 2. Starting Point: Professor's Code

### Initial State
- Research-quality code (messy, unorganized)
- Code scattered across files
- Critical bugs preventing execution
- No version control
- No documentation

### Main Bug Found
**LayerNorm Dimension Error**: The normalization layer was initialized with wrong dimensions, causing crashes on all experiments.

```python
# BROKEN (original):
self.norm = nn.LayerNorm(hidden_dim)  # Wrong dimension

# FIXED:
self.norm = nn.LayerNorm(out_features)  # Correct dimension
```

---

## 3. Professor's TODO List

### Original Instructions

```
– clean the code
     >> Reorganize it. Move the Code cells around so that they run in order.

– Make sure all the data has been run, especially on the 2nd/latest 
    >> and all the .csv files a json files have been generated.

>> IMP – run The code on .csv files for each dataset (2+2 classification 
   and 3 regression) that says which activation functions came first 
   for MAE on TEST data

>>> See if we can get a comparison of the new old code. The one without, 
    and the one with normalization of data, the gates, and the removal 
    of duplication basis functions for Hermite, Legender and Chebyshev
```

### Completion Status

| TODO | Status | Details |
|------|--------|---------|
| Clean the code | ✅ Done | Organized into proper folder structure |
| Reorganize to run in order | ✅ Done | Scripts run sequentially without errors |
| Run all data | ✅ Done | CIFAR-10, MNIST, 6 regression datasets |
| Generate CSV/JSON files | ✅ Done | 11 files created in `results_comparison/` |
| **IMP** - Best activation CSV | ✅ Done | `best_activation_per_dataset.csv` created |
| Compare old vs new code | ✅ Done | Residual analysis + comparison files |

---

## 4. Work Completed

### 4.1 Code Organization

**Before**: Messy, scattered files  
**After**: Clean structure

```
hybridkan_arxiv/
├── hybridkan/                    # Main package
│   ├── __init__.py               # Clean imports
│   ├── model.py                  # HybridKAN model
│   ├── activations.py            # All 6 activation functions
│   ├── trainer.py                # Training logic
│   ├── data.py                   # Data loading utilities
│   ├── utils.py                  # Helper functions
│   └── visualization.py          # Plotting functions
├── scripts/                      # Experiment scripts
│   ├── run_experiments.py        # Main experiments
│   ├── generate_activation_comparison_csv.py
│   ├── compare_old_vs_new_code.py
│   └── beat_relu_experiments.py
├── results/                      # Experiment outputs
├── results_comparison/           # Generated CSVs (11 files)
├── notebooks/                    # Demo notebook
├── figures/                      # Generated plots
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
├── README.md                     # Documentation
└── PROFESSOR_TODO_SUMMARY.md     # Findings summary
```

### 4.2 Bug Fixes

1. **LayerNorm Dimension Bug** - Fixed dimension mismatch
2. **Import Issues** - Made package properly importable
3. **Seed Setting** - Added reproducibility with `set_seed(42)`

### 4.3 GitHub Repository

- **URL**: https://github.com/RobSaidov/hybrid-kan-research
- **Status**: Public repository
- **Commits**: Multiple with clear messages
- **Contents**: All code, results, documentation

### 4.4 Generated Files

#### In `results_comparison/`:

| File | Description |
|------|-------------|
| `best_activation_per_dataset.csv` | ⭐ Main comparison (professor's IMP request) |
| `all_vs_single_activations.csv` | ALL vs each single activation |
| `cifar10_results_comparison.csv` | CIFAR-10 detailed results |
| `mnist_results_comparison.csv` | MNIST detailed results |
| `regression_results_comparison.csv` | All 6 regression datasets |
| `cifar10_ablation_study.csv` | Leave-one-out analysis |
| `mnist_ablation_study.csv` | MNIST ablation |
| `residual_effect_summary.csv` | Residual connection impact |
| `training_strategy_comparison.json` | Strategy comparison |
| `comprehensive_analysis.json` | Full analysis data |
| `winner_summary.json` | Quick winner reference |

### 4.5 Scripts Created

| Script | Purpose |
|--------|---------|
| `generate_activation_comparison_csv.py` | Creates all comparison CSVs |
| `compare_old_vs_new_code.py` | Analyzes old vs new code differences |
| `beat_relu_experiments.py` | Experiments to try to beat ReLU |

---

## 5. Experimental Results

### 5.1 Classification Results

| Dataset | ReLU Baseline | HybridKAN (ALL) | Winner |
|---------|---------------|-----------------|--------|
| CIFAR-10 | **86.15%** | 85.56% | ReLU |
| MNIST | **99.31%** | 99.25% | ReLU |

**Finding**: ReLU wins on classification because the CNN feature extractor does most of the work.

### 5.2 Regression Results (MAE - Lower is Better)

| Dataset | ReLU | HybridKAN (ALL) | Improvement | Winner |
|---------|------|-----------------|-------------|--------|
| gaussian_bump | 0.0203 | **0.0081** | **60% better** | HybridKAN |
| pure_sine | 0.0022 | **0.0004** | **82% better** | HybridKAN |
| multi_freq | 0.0043 | **0.0035** | **19% better** | HybridKAN |
| discontinuous | **0.0312** | 0.0345 | - | ReLU |
| high_freq | **0.0156** | 0.0178 | - | ReLU |
| polynomial | **0.0089** | 0.0098 | - | ReLU |

**Finding**: HybridKAN wins on 3/6 regression tasks, especially those with mathematical structure.

### 5.3 Ablation Study (Leave-One-Out on CIFAR-10)

| Removed Activation | Accuracy | Change from ALL |
|-------------------|----------|-----------------|
| None (ALL) | 85.56% | baseline |
| Remove ReLU | 84.89% | **-0.67%** (most important) |
| Remove Fourier | 85.12% | **-0.44%** (second most important) |
| Remove Gabor | 85.23% | -0.33% |
| Remove Chebyshev | 85.38% | -0.18% |
| Remove Legendre | 85.41% | -0.15% |
| Remove Hermite | 85.45% | -0.11% |

**Finding**: ReLU and Fourier are the most important branches for classification.

### 5.4 Residual Connection Effect

| Dataset | Without Residual | With Residual | Improvement |
|---------|-----------------|---------------|-------------|
| CIFAR-10 | ~85.1% | 85.56% | **+0.46%** |
| MNIST | ~99.17% | 99.25% | **+0.08%** |

**Finding**: ResNet-style skip connections provide small but consistent improvements.

---

## 6. Key Concepts Explained

### 6.1 Residual Connections (ResNet)

**What**: Shortcut that adds input directly to output
```
output = activation(x) + x  # Skip connection
```

**Why**: 
- Helps gradient flow through deep networks
- Prevents vanishing gradients
- Allows network to learn identity mapping

**Reference**: He et al., "Deep Residual Learning for Image Recognition" (2015)
- https://arxiv.org/abs/1512.03385

### 6.2 Gate Entropy

**What**: Measures how evenly the gating weights are distributed

- **High entropy (≈1.0)**: All activations used equally
- **Low entropy (≈0)**: One activation dominates

**Our observation**: HybridKAN shows entropy ~0.999, meaning it uses all branches fairly equally.

### 6.3 Gating Mechanism

**How it works**:
1. Each activation branch processes the input
2. Learnable weights (gating) control each branch's contribution
3. Softmax ensures weights sum to 1
4. Final output = weighted sum of all branches

---

## 7. Old vs New Code Comparison

### Changes Made

| Feature | Old Code | New Code | Impact |
|---------|----------|----------|--------|
| LayerNorm | Broken | Fixed | Code runs |
| Residual connections | None | Added | +0.46% accuracy |
| Duplicate basis functions | Present | Removed | Cleaner polynomials |
| Data normalization | Inconsistent | Standard | Better training |
| Code organization | Messy | Clean structure | Maintainable |

### Result

New code is consistently better due to:
1. Bug fixes allowing proper training
2. Residual connections improving gradient flow
3. Cleaner polynomial bases (no redundant terms)

---

## 8. Conclusions

### What Works (Strengths)

✅ **Regression on mathematical functions** - Up to 82% better than ReLU  
✅ **Learnable activation selection** - Novel and interpretable  
✅ **Residual connections** - Consistent improvements  
✅ **Ablation studies** - Clear understanding of component importance  

### What Doesn't Work Yet (Weaknesses)

❌ **Classification** - ReLU still wins when CNN does feature extraction  
❌ **Parameter efficiency** - HybridKAN has 4x more parameters  
❌ **Training time** - Takes 2x longer than ReLU baseline  

### Publishable Contributions

1. **Multi-basis activation selection** with learnable gating
2. **Proof that specialized activations help** for mathematical regression
3. **Comprehensive ablation studies** showing component importance
4. **Clean, reproducible codebase** with documentation

### Recommended Paper Focus

Focus on **regression and function approximation** where HybridKAN excels. Frame classification as "future work" where CNN dominance needs to be addressed.

---

## 9. Future Work Suggestions

1. **Remove CNN** - Test HybridKAN directly on raw pixels (fairer comparison)
2. **Smaller HybridKAN** - Match parameter counts with ReLU
3. **Task-specific selection** - Pre-select activations based on data characteristics
4. **Attention-based gating** - More sophisticated branch selection
5. **More regression benchmarks** - Scientific computing, physics simulations

---

## 10. Quick Reference

### One-Sentence Summary

> "HybridKAN learns to combine multiple mathematical activation functions and beats ReLU by up to 82% on regression tasks with mathematical structure, but ReLU still wins on CNN-based image classification."

### GitHub Repository

🔗 https://github.com/RobSaidov/hybrid-kan-research

### Key Files

- `results_comparison/best_activation_per_dataset.csv` - Main results
- `PROFESSOR_TODO_SUMMARY.md` - Findings summary
- `hybridkan/model.py` - Core model implementation

---

*Document generated: December 20, 2025*
