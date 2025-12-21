# HybridKAN: Complete Architecture Documentation

**Date**: December 21, 2025  
**Authors**: Rob Saidov & Professor's Lab  
**GitHub**: https://github.com/RobSaidov/hybrid-kan-research

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Architecture Overview](#3-architecture-overview)
4. [Activation Functions (6 Bases)](#4-activation-functions-6-bases)
5. [Key Architectural Features](#5-key-architectural-features)
6. [Ablation Study Methodology](#6-ablation-study-methodology)
7. [Experimental Results Summary](#7-experimental-results-summary)
8. [Visual Architecture Diagrams](#8-visual-architecture-diagrams)

---

## 1. Executive Summary

### What is HybridKAN?

HybridKAN is a **Multi-Basis Neural Network** that extends standard MLPs by replacing single activation functions with **6 parallel activation branches** that are combined through **learnable gates**. This design is inspired by the **Kolmogorov-Arnold Representation Theorem**.

### Key Features at a Glance

| Feature | Description | Purpose |
|---------|-------------|---------|
| **6 Parallel Branches** | Gabor, Legendre, Chebyshev, Hermite, Fourier, ReLU | Capture diverse mathematical patterns |
| **Learnable Gates** | Softplus-scaled weights per branch | Adaptive branch importance |
| **Per-Branch LayerNorm** | Normalize each branch independently | Stabilize multi-scale outputs |
| **Residual Connections** | Skip connections with learnable gates | Improve gradient flow |
| **Polynomial De-duplication** | Avoid redundant degree-0/1 terms | Reduce parameter redundancy |
| **Optional CNN Preprocessor** | For image inputs (MNIST, CIFAR-10) | Spatial feature extraction |

---

## 2. Theoretical Foundation

### 2.1 Kolmogorov-Arnold Representation Theorem (KA Theorem)

The mathematical foundation comes from Andrey Kolmogorov's 1957 theorem:

**Theorem**: Any continuous function f: [0,1]^n → ℝ can be written as:

$$f(x_1, x_2, ..., x_n) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

**Interpretation**:
- **Inner functions** φ_{q,p}: Map individual inputs through univariate functions
- **Outer functions** Φ_q: Combine the sums into final output
- **Total functions needed**: (n+1)(2n+1) = 2n² + 3n + 1

### 2.2 Why Multiple Bases?

Different mathematical bases excel at different patterns:

| Pattern Type | Best Basis | Mathematical Reason |
|--------------|-----------|---------------------|
| Periodic (sin/cos) | **Fourier** | Inherently periodic basis |
| Localized oscillations | **Gabor** | Windowed sinusoid |
| Smooth polynomials | **Legendre** | Orthogonal on [-1,1] |
| Bounded intervals | **Chebyshev** | Minimize max error |
| Gaussian-weighted | **Hermite** | Orthogonal under e^(-x²) |
| Piecewise linear | **ReLU** | Universal approximator |

### 2.3 Connection to Original KAN Papers

Traditional KANs (Liu et al., 2024) use **B-splines** for the φ functions. HybridKAN extends this by:

1. Using **multiple orthogonal bases** instead of just splines
2. Adding **learnable gates** to weight branch importance
3. Incorporating **modern techniques** (LayerNorm, residual connections)

---

## 3. Architecture Overview

### 3.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            HybridKAN Forward Pass                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT (e.g., 32×32×3 image)                                               │
│      │                                                                      │
│      ▼                                                                      │
│   ┌──────────────────────────────┐                                          │
│   │  CNN Preprocessor (Optional) │ ← 3 conv blocks → 256-dim embedding     │
│   └──────────────────────────────┘                                          │
│      │                                                                      │
│      ▼                                                                      │
│   ┌──────────────────────────────┐                                          │
│   │     Input LayerNorm          │ ← Normalize input features               │
│   └──────────────────────────────┘                                          │
│      │                                                                      │
│      ▼                                                                      │
│   ┌──────────────────────────────┐                                          │
│   │   HybridKANBlock (Layer 1)   │ ← 6 parallel branches + gating           │
│   └──────────────────────────────┘                                          │
│      │ + skip connection (optional)                                         │
│      ▼                                                                      │
│   ┌──────────────────────────────┐                                          │
│   │   HybridKANBlock (Layer 2)   │ ← 6 parallel branches + gating           │
│   └──────────────────────────────┘                                          │
│      │ + skip connection (optional)                                         │
│      ▼                                                                      │
│   ┌──────────────────────────────┐                                          │
│   │   HybridKANBlock (Layer 3)   │ ← 6 parallel branches + gating           │
│   └──────────────────────────────┘                                          │
│      │                                                                      │
│      ▼                                                                      │
│   ┌──────────────────────────────┐                                          │
│   │      Output Head             │ ← Linear → softmax (classification)      │
│   │      (Linear Layer)          │ ← Linear (regression)                    │
│   └──────────────────────────────┘                                          │
│      │                                                                      │
│      ▼                                                                      │
│   OUTPUT (10 classes or 1 value)                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Single HybridKANBlock Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HybridKANBlock Detail                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT x ∈ ℝ^(B × D_in)                                                    │
│      │                                                                      │
│      ├────────┬────────┬────────┬────────┬────────┬────────┐                │
│      ▼        ▼        ▼        ▼        ▼        ▼        │                │
│   ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐         │                │
│   │Gabor ││Legen-││Cheby-││Hermite││Fourier││ReLU  │         │                │
│   │      ││dre   ││shev  ││      ││      ││      │         │                │
│   └───┬──┘└───┬──┘└───┬──┘└───┬──┘└───┬──┘└───┬──┘         │                │
│       │       │       │       │       │       │            │                │
│       ▼       ▼       ▼       ▼       ▼       ▼            │                │
│   ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐         │                │
│   │Layer ││Layer ││Layer ││Layer ││Layer ││Layer │ ←Per-Branch              │
│   │Norm  ││Norm  ││Norm  ││Norm  ││Norm  ││Norm  │  Normalization           │
│   └───┬──┘└───┬──┘└───┬──┘└───┬──┘└───┬──┘└───┬──┘         │                │
│       │       │       │       │       │       │            │                │
│       ▼       ▼       ▼       ▼       ▼       ▼            │                │
│   ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐         │                │
│   │Gate  ││Gate  ││Gate  ││Gate  ││Gate  ││Gate  │ ←Learnable               │
│   │α_gab ││α_leg ││α_che ││α_her ││α_fou ││α_rel │  Weights                 │
│   └───┬──┘└───┬──┘└───┬──┘└───┬──┘└───┬──┘└───┬──┘         │                │
│       │       │       │       │       │       │            │                │
│       └───────┴───────┴───────┴───────┴───────┘            │                │
│                       │                                     │ Skip          │
│                       ▼                                     │ Connection    │
│               ┌───────────────┐                             │                │
│               │ Concatenate   │ → ℝ^(B × 6×D_out)          │                │
│               └───────┬───────┘                             │                │
│                       │                                     │                │
│                       ▼                                     │                │
│               ┌───────────────┐                             │                │
│               │  BatchNorm1d  │                             │                │
│               └───────┬───────┘                             │                │
│                       │                                     │                │
│                       ▼                                     │                │
│               ┌───────────────┐                             │                │
│               │     GELU      │                             │                │
│               └───────┬───────┘                             │                │
│                       │                                     │                │
│                       ▼                                     │                │
│               ┌───────────────┐                             │                │
│               │   Dropout     │                             │                │
│               └───────┬───────┘                             │                │
│                       │                                     │                │
│                       ▼                                     │                │
│               ┌───────────────┐                             │                │
│               │  Projection   │ → ℝ^(B × D_out)            │                │
│               │   (Linear)    │                             │                │
│               └───────┬───────┘                             │                │
│                       │                                     │                │
│                       ▼                                     ▼                │
│               ┌───────────────────────────────────────────────┐             │
│               │          x + α_res × identity                 │ ←Residual   │
│               │        (if residual enabled)                  │  Gate       │
│               └───────────────────────────────────────────────┘             │
│                                   │                                         │
│                                   ▼                                         │
│                              OUTPUT ∈ ℝ^(B × D_out)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Activation Functions (6 Bases)

### 4.1 Gabor Wavelets

**Mathematical Form**:
$$\text{Gabor}(x) = A \cdot \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \cdot \cos(\pi \cdot f \cdot x + \phi)$$

**Learnable Parameters** (per output × per input):
- μ (center): Position of the wavelet
- σ (scale): Width of Gaussian envelope
- f (frequency): Oscillation frequency
- φ (phase): Phase shift
- A (amplitude): Output scaling

**Best For**: Localized oscillatory patterns, texture detection, edge detection

**Implementation Notes**:
- Parameters clamped for stability: σ ∈ [0.05, 5.0], f ∈ [0.2, 5.0], A ∈ [0.0, 1.0]
- Gaussian term clamped: exp(-50) minimum to avoid underflow

---

### 4.2 Legendre Polynomials

**Mathematical Form** (Bonnet's recursion):
$$P_0(x) = 1, \quad P_1(x) = x$$
$$P_n(x) = \frac{(2n-1) \cdot x \cdot P_{n-1}(x) - (n-1) \cdot P_{n-2}(x)}{n}$$

**Output**:
$$\text{Legendre}(x) = \sum_{k=0}^{d} c_k \cdot P_k(\tanh(s \cdot x))$$

**Learnable Parameters**:
- c_k: Mixing coefficients for each degree
- s: Input scaling factor

**Properties**:
- Orthogonal on [-1, 1] with weight w(x) = 1
- Bounded: |P_n(x)| ≤ 1 on [-1, 1]
- Default degree: 8 (9 basis functions)

**Best For**: Smooth global approximation, polynomial features

---

### 4.3 Chebyshev Polynomials (First Kind)

**Mathematical Form**:
$$T_0(x) = 1, \quad T_1(x) = x$$
$$T_n(x) = 2x \cdot T_{n-1}(x) - T_{n-2}(x)$$

**Properties**:
- Satisfies: T_n(cos θ) = cos(nθ)
- **Minimax property**: Minimizes maximum error on [-1, 1]
- Orthogonal with weight w(x) = 1/√(1-x²)

**Best For**: Uniform approximation, minimizing worst-case error

---

### 4.4 Hermite Polynomials (Probabilist's)

**Mathematical Form**:
$$H_0(x) = 1, \quad H_1(x) = 2x$$
$$H_n(x) = 2x \cdot H_{n-1}(x) - 2(n-1) \cdot H_{n-2}(x)$$

**Output** (with Gaussian envelope):
$$\text{Hermite}(x) = \sum_{k=0}^{d} c_k \cdot H_k(x/\sigma) \cdot \exp(-x^2/(2\sigma^2))$$

**Properties**:
- Orthogonal under Gaussian measure: ∫ H_m H_n e^(-x²) dx = √π · 2^n · n! · δ_{mn}
- Natural for probability distributions

**Best For**: Gaussian-shaped data, probability density estimation, bell curves

---

### 4.5 Fourier Basis

**Mathematical Form**:
$$\text{Fourier}(x) = \sum_{k=1}^{K} A_k \cdot \sin(f_k \cdot x + \phi_k)$$

**Learnable Parameters**:
- f_k: Frequencies (initialized ~ N(0, 2))
- φ_k: Phases (initialized ~ U(-π, π))
- A_k: Amplitudes (initialized = 0.5)

**Properties**:
- Complete basis for L²[0, 2π]
- Naturally captures periodicity
- Default: K = 8 frequency components

**Best For**: Periodic functions, oscillatory signals, Fourier analysis

---

### 4.6 ReLU (Baseline)

**Mathematical Form**:
$$\text{ReLU}(x) = \max(0, W \cdot x + b)$$

**Properties**:
- Universal approximator (with sufficient width)
- Piecewise linear → computationally efficient
- Default activation in modern deep learning

**Best For**: General purpose, fast computation, baseline comparison

---

## 5. Key Architectural Features

### 5.1 Learnable Branch Gates

**Purpose**: Let the network learn which activation functions matter for each layer.

**Implementation**:
```python
class BranchGate(nn.Module):
    def __init__(self, init_value=0.5):
        self.alpha = nn.Parameter(torch.tensor(init_value))
    
    def forward(self, x):
        return F.softplus(self.alpha) * x  # Non-negative scaling
```

**Why Softplus?**
- Ensures gate weight ≥ 0 (can't have negative importance)
- Smooth gradient everywhere (unlike ReLU)
- Allows gates to go arbitrarily large if needed

**Typical Learned Values** (CIFAR-10, Layer 0):
| Branch | Gate Weight | Interpretation |
|--------|-------------|----------------|
| Gabor | 1.10 | Dominant early |
| Legendre | 0.65 | Moderate |
| Chebyshev | 0.62 | Moderate |
| Hermite | 0.60 | Moderate |
| Fourier | 0.58 | Moderate |
| ReLU | 0.70 | Important |

---

### 5.2 Per-Branch Layer Normalization

**Problem**: Different activation functions output at vastly different scales:
- ReLU: unbounded positive
- Gabor: bounded by amplitude ≈ 0.1
- Polynomials: can grow to degree^n

**Solution**: Apply LayerNorm **after each branch, before gating**:

```python
for name in self.branch_names:
    out = self.branches[name](x)           # Branch output
    out = self.branch_norms[name](out)     # Normalize to mean=0, var=1
    out = self.gates[name](out)            # Then apply gate
```

**Effect**: All branches compete on equal footing in the concatenation.

---

### 5.3 Residual (Skip) Connections

**Purpose**: Improve gradient flow, enable deeper networks.

**Implementation**:
```python
# With learnable gating (not just x + F(x))
class ResidualGate(nn.Module):
    def __init__(self, init_value=0.1):
        self.alpha = nn.Parameter(torch.tensor(init_value))
    
    def forward(self, x):
        return torch.sigmoid(self.alpha) * x  # Output in [0, 1]
```

**Forward Pass**:
```python
identity = x
x = block(x)
if use_residual:
    if identity.shape != x.shape:
        identity = self.projection(identity)  # Match dimensions
    x = x + residual_gate(identity)  # Learnable skip strength
```

**Results** (CIFAR-10):
| Model | Residual | Accuracy |
|-------|----------|----------|
| all_branches | ✅ Yes | 85.63% |
| all_no_residual | ❌ No | 85.17% |
| **Improvement** | — | **+0.46%** |

---

### 5.4 Polynomial Degree De-duplication

**Problem**: Legendre, Chebyshev, and Hermite ALL have:
- Degree 0: constant term (≈ 1)
- Degree 1: linear term (≈ x)

This creates **redundant parameters** learning the same thing.

**Solution**: Keep deg-0/1 in ONE family only (default: Legendre), skip in others:

```python
def _compute_start_degrees(self, dedup, keep_family):
    if dedup:
        for family in ["legendre", "chebyshev", "hermite"]:
            if family != keep_family:
                start_degrees[family] = 2  # Skip deg 0 and 1
            else:
                start_degrees[family] = 0  # Keep all
    return start_degrees
```

**Result**:
| Family | Degrees Used (with dedup) | Degrees Used (without) |
|--------|---------------------------|------------------------|
| Legendre | 0, 1, 2, 3, ..., 8 | 0, 1, 2, ..., 8 |
| Chebyshev | 2, 3, ..., 8 | 0, 1, 2, ..., 8 |
| Hermite | 2, 3, ..., 6 | 0, 1, 2, ..., 6 |

**Parameter Savings**: ~15% fewer polynomial parameters

---

### 5.5 CNN Preprocessor (Optional)

**Purpose**: Extract spatial features from images before the HybridKAN MLP.

**Architecture**:
```
Input: [B, C, H, W]  (e.g., [64, 3, 32, 32] for CIFAR-10)
    │
    ▼
Conv2d(C→32) + BatchNorm + GELU + MaxPool(2)
    │
    ▼
Conv2d(32→64) + BatchNorm + GELU + MaxPool(2)
    │
    ▼
Conv2d(64→128) + BatchNorm + GELU + AdaptiveAvgPool(1)
    │
    ▼
Flatten + Linear(128→256) + GELU
    │
    ▼
Output: [B, 256]  (flattened features for HybridKAN)
```

**Effect on Accuracy** (CIFAR-10):
| Model | CNN | Accuracy |
|-------|-----|----------|
| HybridKAN + CNN | ✅ Yes | 85.63% |
| HybridKAN (no CNN) | ❌ No | ~80% |

---

## 6. Ablation Study Methodology

### 6.1 Leave-One-Out Analysis

**Question**: How much does each branch contribute?

**Method**: Train with ALL 6 branches, then train with each branch removed one at a time.

**CIFAR-10 Results**:
| Configuration | Accuracy | Δ from Full |
|---------------|----------|-------------|
| ALL (baseline) | 85.63% | — |
| - Gabor | 85.76% | **+0.13%** (helps to remove!) |
| - Legendre | 85.56% | -0.07% |
| - Chebyshev | 85.37% | **-0.26%** (important!) |
| - Hermite | 85.44% | -0.19% |
| - Fourier | 85.39% | **-0.24%** (important!) |
| - ReLU | 85.50% | -0.13% |

**Key Findings**:
1. **Chebyshev and Fourier** are most important (biggest drops when removed)
2. **Gabor** may be redundant for CIFAR-10 (actually improves when removed)
3. **No single branch is catastrophic** to remove

---

### 6.2 Single Branch Analysis

**Question**: How good is each activation alone?

**CIFAR-10 Results** (with CNN preprocessor):
| Configuration | Accuracy |
|---------------|----------|
| ReLU only | **86.15%** |
| Fourier only | 83.2% |
| Legendre only | 82.8% |
| Chebyshev only | 82.5% |
| Gabor only | 81.9% |
| Hermite only | 81.5% |

**Key Finding**: ReLU alone + CNN beats all HybridKAN variants on CIFAR-10!

---

### 6.3 Residual Connection Analysis

| Model | With Residual | Without Residual | Δ |
|-------|---------------|------------------|---|
| ALL branches | 85.63% | 85.17% | **+0.46%** |

---

## 7. Experimental Results Summary

### 7.1 Classification Tasks

| Dataset | ReLU Only | HybridKAN (ALL) | Winner |
|---------|-----------|-----------------|--------|
| **CIFAR-10** | **86.15%** | 85.63% | ReLU |
| **MNIST** | **99.50%** | 99.44% | ReLU |
| **Wine** | 100% | 100% | Tie |
| **Iris** | 90% | 80% | ReLU |
| **Breast Cancer** | **95.61%** | 92.11% | ReLU |

**Conclusion**: ReLU wins or ties on all classification tasks.

---

### 7.2 Regression Tasks (HybridKAN Wins!)

| Dataset | ReLU MSE | HybridKAN MSE | Improvement |
|---------|----------|---------------|-------------|
| **pure_sine** | 0.450 | **0.080** | **82% better** |
| **gaussian_bump** | 0.200 | **0.080** | **60% better** |
| **multi_freq** | 0.160 | **0.130** | **19% better** |

**Conclusion**: HybridKAN excels when data has periodic or localized structure.

---

### 7.3 When to Use HybridKAN?

| Data Characteristics | Recommendation |
|---------------------|----------------|
| Images (CNN needed) | Use **ReLU** (simpler, faster, same/better accuracy) |
| Periodic functions (sin, cos) | Use **HybridKAN with Fourier** |
| Localized oscillations | Use **HybridKAN with Gabor** |
| Polynomial/smooth functions | Use **HybridKAN with Legendre/Chebyshev** |
| General tabular data | Try both, compare |

---

## 8. Visual Architecture Diagrams

### 8.1 Complete Network Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HybridKAN Complete Architecture                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Input Layer                                │   │
│  │  Image: [B, 3, 32, 32]    OR    Tabular: [B, D_features]    │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │            CNN Preprocessor (if use_cnn=True)                │   │
│  │                                                              │   │
│  │   Conv2d(3→32) → BN → GELU → MaxPool                        │   │
│  │   Conv2d(32→64) → BN → GELU → MaxPool                       │   │
│  │   Conv2d(64→128) → BN → GELU → AdaptivePool                 │   │
│  │   Flatten → Linear(128→256) → GELU                          │   │
│  │                                                              │   │
│  │   Output: [B, 256]                                           │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Input LayerNorm                           │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│         ┌───────────────────┼───────────────────┐                  │
│         │                   │                   │                  │
│         ▼                   ▼                   ▼                  │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐             │
│  │ HybridKAN  │     │ HybridKAN  │     │ HybridKAN  │             │
│  │  Block 1   │ ──► │  Block 2   │ ──► │  Block 3   │             │
│  │ (256→256)  │     │ (256→128)  │     │ (128→64)   │             │
│  └─────┬──────┘     └─────┬──────┘     └─────┬──────┘             │
│        │                  │                  │                     │
│        └──────────────────┴──────────────────┘                     │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Output Head                               │   │
│  │                                                              │   │
│  │   Classification: Linear(64→num_classes) → LogSoftmax       │   │
│  │   Regression:     Linear(64→1)                              │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                       Output                                 │   │
│  │   Classification: [B, num_classes] (log probabilities)      │   │
│  │   Regression:     [B, 1] (predicted value)                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Branch Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│              Activation Function Comparison                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  GABOR (Localized Oscillation)        FOURIER (Global Periodic)    │
│  ~~~~                                 ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿        │
│ /    \                                                              │
│       ~~~~                            Best for: sin(x), cos(2πx)   │
│                                                                     │
│  Best for: edge detection,            Parameters: f_k, φ_k, A_k    │
│            localized features         (8 frequencies default)       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LEGENDRE (Smooth Polynomial)         CHEBYSHEV (Minimax Poly)     │
│      ___                                  ___                       │
│    _/   \_                              _/   \_                     │
│  _/       \_                          _/       \_                   │
│ /           \                        /           \                  │
│                                                                     │
│  Best for: smooth interpolation       Best for: uniform approx     │
│  Orthogonal on [-1,1], w(x)=1         Orthogonal, w=1/√(1-x²)      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  HERMITE (Gaussian-weighted)          RELU (Piecewise Linear)      │
│      _∩_                                    /                       │
│    _/   \_                                 /                        │
│  _/       \_                         _____/                         │
│ /           \                                                       │
│ ∿∿∿∿∿∿∿∿∿∿∿∿∿                        Best for: general purpose     │
│                                       Fast, simple, proven          │
│  Best for: probability densities,                                   │
│            Gaussian data                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: File Structure

```
hybridkan_arxiv/
├── hybridkan/                      # Core library
│   ├── __init__.py                 # Package exports
│   ├── model.py                    # HybridKAN, HybridKANBlock, Gates
│   ├── activations.py              # All 6 activation classes
│   ├── trainer.py                  # Training loop
│   ├── data.py                     # Data utilities
│   ├── utils.py                    # Helpers
│   └── visualization.py            # Plotting
│
├── scripts/                        # Experiment scripts
│   ├── run_experiments.py          # Main experiments (CIFAR-10, MNIST)
│   ├── run_additional_datasets.py  # Wine, Iris, Breast Cancer, etc.
│   ├── beat_relu_experiments.py    # Strategies to beat ReLU
│   └── generate_*.py               # CSV/visualization generation
│
├── results/                        # Experiment outputs
│   ├── cifar10/RESULTS.md
│   └── mnist/RESULTS.md
│
├── results_comparison/             # CSV comparisons
│   ├── classification_comparison.csv
│   ├── regression_comparison.csv
│   └── ... (11 files)
│
├── figures/                        # Generated visualizations
│   ├── hybridkan_architecture.png
│   ├── gate_heatmap.png
│   └── ... 
│
├── notebooks/
│   └── HybridKAN_Demo.ipynb       # Interactive demo
│
└── Documentation
    ├── README.md
    ├── RESEARCH_SUMMARY.md
    ├── COMPREHENSIVE_WORK_SUMMARY.md
    └── HYBRIDKAN_ARCHITECTURE_DETAILED.md (this file)
```

---

## Appendix B: Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | [256, 128, 64] | Hidden layer widths |
| `dropout_rate` | 0.3 | Dropout probability |
| `per_branch_norm` | True | LayerNorm per branch |
| `branch_gates` | True | Learnable gate weights |
| `use_residual` | True | Skip connections |
| `residual_every_n` | 1 | Add skip every N blocks |
| `dedup_poly_deg01` | True | Remove redundant deg-0/1 |
| `keep01_family` | "legendre" | Which family keeps deg-0/1 |
| Gabor `amp_init` | 0.1 | Initial amplitude |
| Gabor `sigma_init` | 1.0 | Initial Gaussian width |
| Legendre `degree` | 8 | Max polynomial degree |
| Chebyshev `degree` | 8 | Max polynomial degree |
| Hermite `degree` | 6 | Max polynomial degree |
| Fourier `n_frequencies` | 8 | Number of sin components |

---

## Appendix C: Training Configuration

| Setting | CIFAR-10/MNIST | Tabular |
|---------|----------------|---------|
| Optimizer | AdamW | Adam |
| Learning Rate | 1e-3 | 1e-3 |
| Weight Decay | 1e-4 | 0 |
| Scheduler | CosineAnnealing | None |
| Batch Size | 128 | 32 |
| Epochs | 50 | 100-150 |
| Early Stopping | Patience=20 | Patience=20 |

---

*Document generated December 21, 2025*
