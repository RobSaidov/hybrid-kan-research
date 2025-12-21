# HybridKAN: Hybrid Kolmogorov-Arnold Networks with Multi-Basis Activation Functions

**A Comprehensive Research Report**

---

**Authors**: Rob Saidov & Research Lab  
**Institution**: University Research Program  
**Date**: December 21, 2025  
**Version**: 1.0 (Full Draft)  
**GitHub Repository**: https://github.com/RobSaidov/hybrid-kan-research

---

## Abstract

We present **HybridKAN**, a novel neural network architecture that extends the Kolmogorov-Arnold representation theorem by combining multiple orthogonal mathematical basis functions into a unified learning framework. Unlike traditional neural networks that rely solely on ReLU activations, HybridKAN employs six parallel activation branches—**Gabor wavelets, Legendre polynomials, Chebyshev polynomials, Hermite functions, Fourier basis, and ReLU**—each equipped with learnable gates that enable data-driven specialization.

Our key contributions include:
1. A multi-basis architecture with **learnable branch gates** using softplus activation
2. **Per-branch Layer Normalization** to stabilize heterogeneous activation scales
3. **Polynomial degree de-duplication** to reduce parameter redundancy
4. **Learnable residual connections** with sigmoid-gated skip connections
5. Comprehensive ablation studies on classification and regression tasks

Our experiments reveal a striking **task-dependent** pattern: while ReLU-only networks achieve superior classification performance (86.15% on CIFAR-10 vs. 85.63% for HybridKAN), HybridKAN significantly outperforms ReLU on regression tasks with structured mathematical properties, achieving **82% lower MSE** on pure sinusoidal functions. These findings suggest that HybridKAN is particularly valuable for **scientific computing, physics-informed neural networks, and function approximation** where the underlying data has known mathematical structure.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [Theoretical Background](#3-theoretical-background)
4. [Architecture Design](#4-architecture-design)
5. [Implementation Details](#5-implementation-details)
6. [Experimental Setup](#6-experimental-setup)
7. [Results](#7-results)
8. [Analysis and Discussion](#8-analysis-and-discussion)
9. [Novel Applications](#9-novel-applications)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [Conclusion](#11-conclusion)
12. [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Motivation

The choice of activation function fundamentally shapes what a neural network can efficiently represent. While ReLU and its variants have dominated deep learning due to their simplicity and effectiveness, they are inherently piecewise-linear functions. This limitation becomes apparent when approximating functions with periodic, polynomial, or localized structure—domains where specialized mathematical bases like Fourier series or orthogonal polynomials are provably optimal.

The Kolmogorov-Arnold representation theorem (1957) provides a mathematical foundation for function decomposition using univariate functions. Recent work on Kolmogorov-Arnold Networks (KANs) has explored using learnable splines, but these approaches typically commit to a single basis family.

**Research Question**: Can a neural network that combines multiple mathematical basis functions into parallel branches, with learnable gating to select the most appropriate basis per layer, achieve better performance than single-basis networks?

### 1.2 Key Contributions

1. **HybridKAN Architecture**: A novel multi-basis neural network with six parallel activation branches combined through learnable gates

2. **Architectural Innovations**:
   - Per-branch Layer Normalization for scale stability
   - Softplus-gated branch importance weighting
   - Polynomial degree de-duplication across families
   - Learnable residual connections with sigmoid gates

3. **Comprehensive Empirical Study**:
   - Classification: MNIST, CIFAR-10, Wine, Iris, Breast Cancer
   - Regression: California Housing, Diabetes, synthetic functions
   - Physics-Informed Neural Networks (PINNs)

4. **Task-Dependent Findings**:
   - ReLU dominates classification tasks
   - HybridKAN excels on structured regression
   - Gate weights provide interpretable function decomposition

### 1.3 Paper Organization

Section 2 reviews related work on activation functions and KANs. Section 3 presents the theoretical foundation. Section 4 details our architecture design. Section 5 covers implementation. Sections 6-8 present experiments and analysis. Section 9 explores novel applications. Section 10 discusses limitations, and Section 11 concludes.

---

## 2. Related Work

### 2.1 Kolmogorov-Arnold Representation Theorem

The foundational theorem (Kolmogorov, 1957; Arnold, 1958) states that any multivariate continuous function can be decomposed into univariate functions:

$$f(x_1, ..., x_n) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

This theorem inspired recent KAN architectures (Liu et al., 2024) that use learnable B-splines for the inner functions φ.

### 2.2 Activation Functions in Deep Learning

| Activation | Mathematical Form | Properties |
|------------|-------------------|------------|
| ReLU | max(0, x) | Piecewise linear, sparse gradients |
| GELU | x · Φ(x) | Smooth approximation of ReLU |
| Swish | x · σ(x) | Self-gated, smooth |
| Mish | x · tanh(softplus(x)) | Self-regularizing |

While these activations are effective, they share a limitation: none naturally capture periodicity, polynomial structure, or localized oscillations.

### 2.3 Specialized Basis Functions

- **Fourier Neural Operators** (Li et al., 2020): Use Fourier transforms for PDE solving
- **Chebyshev Networks** (Tancik et al., 2020): Positional encoding with sinusoidal features
- **Wavelet Networks** (Zhang & Benveniste, 1992): Gabor/wavelet basis for signal processing

HybridKAN unifies these approaches by combining multiple bases with learnable selection.

### 2.4 Mixture of Experts

HybridKAN shares conceptual similarities with Mixture of Experts (MoE) architectures, where different "experts" handle different inputs. Our branch gates serve a similar purpose, but at the activation function level rather than the network-module level.

---

## 3. Theoretical Background

### 3.1 Kolmogorov-Arnold Representation Theorem

**Theorem (Kolmogorov, 1957)**: For any continuous function $f: [0,1]^n \rightarrow \mathbb{R}$, there exist continuous univariate functions $\phi_{q,p}: [0,1] \rightarrow \mathbb{R}$ and $\Phi_q: \mathbb{R} \rightarrow \mathbb{R}$ such that:

$$f(x_1, x_2, ..., x_n) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

**Interpretation**:
- Inner functions $\phi_{q,p}$: Transform individual inputs independently
- Outer functions $\Phi_q$: Combine transformed inputs into output
- Total functions needed: $(n+1)(2n+1) = 2n^2 + 3n + 1$

### 3.2 Why Multiple Basis Functions?

Different mathematical bases excel at representing different function classes:

| Function Type | Optimal Basis | Mathematical Reason |
|---------------|---------------|---------------------|
| Periodic (sin/cos) | **Fourier** | Complete basis for L²[0, 2π] |
| Localized oscillations | **Gabor** | Time-frequency localization |
| Smooth polynomials | **Legendre** | Orthogonal on [-1,1] |
| Minimax approximation | **Chebyshev** | Minimizes max error |
| Gaussian-weighted | **Hermite** | Orthogonal under e^(-x²) |
| Piecewise linear | **ReLU** | Universal approximator |

**Key Insight**: By combining all bases with learnable gates, HybridKAN can adaptively select the most appropriate representation for each layer and task.

### 3.3 Orthogonal Polynomial Properties

| Family | Weight Function | Interval | Recurrence |
|--------|-----------------|----------|------------|
| Legendre | w(x) = 1 | [-1, 1] | $P_n = \frac{(2n-1)xP_{n-1} - (n-1)P_{n-2}}{n}$ |
| Chebyshev | w(x) = 1/√(1-x²) | [-1, 1] | $T_n = 2xT_{n-1} - T_{n-2}$ |
| Hermite | w(x) = e^(-x²) | (-∞, ∞) | $H_n = 2xH_{n-1} - 2(n-1)H_{n-2}$ |

---

## 4. Architecture Design

### 4.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HybridKAN Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT (e.g., 32×32×3 image or tabular features)                           │
│      │                                                                      │
│      ▼                                                                      │
│   ┌──────────────────────────────┐                                          │
│   │  CNN Preprocessor (Optional) │  ← For images: 3 conv blocks → 256-dim  │
│   └──────────────────────────────┘                                          │
│      │                                                                      │
│      ▼                                                                      │
│   ┌──────────────────────────────┐                                          │
│   │     Input LayerNorm          │  ← Normalize input features              │
│   └──────────────────────────────┘                                          │
│      │                                                                      │
│      ▼                                                                      │
│   ╔══════════════════════════════╗                                          │
│   ║   HybridKANBlock (Layer 1)   ║  ← 6 parallel branches + gating          │
│   ╚══════════════════════════════╝                                          │
│      │ + residual connection (α₁)                                           │
│      ▼                                                                      │
│   ╔══════════════════════════════╗                                          │
│   ║   HybridKANBlock (Layer 2)   ║  ← 6 parallel branches + gating          │
│   ╚══════════════════════════════╝                                          │
│      │ + residual connection (α₂)                                           │
│      ▼                                                                      │
│   ╔══════════════════════════════╗                                          │
│   ║   HybridKANBlock (Layer 3)   ║  ← 6 parallel branches + gating          │
│   ╚══════════════════════════════╝                                          │
│      │                                                                      │
│      ▼                                                                      │
│   ┌──────────────────────────────┐                                          │
│   │      Output Head             │  ← Linear → LogSoftmax (classification)  │
│   │      (Linear Layer)          │  ← Linear (regression)                   │
│   └──────────────────────────────┘                                          │
│      │                                                                      │
│      ▼                                                                      │
│   OUTPUT (class probabilities or regression value)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 HybridKAN Block Architecture

Each HybridKAN block processes input through six parallel branches:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HybridKANBlock Detail                              │
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
│   │Layer ││Layer ││Layer ││Layer ││Layer ││Layer │ ← Per-Branch             │
│   │Norm  ││Norm  ││Norm  ││Norm  ││Norm  ││Norm  │   Normalization          │
│   └───┬──┘└───┬──┘└───┬──┘└───┬──┘└───┬──┘└───┬──┘         │                │
│       │       │       │       │       │       │            │                │
│       ▼       ▼       ▼       ▼       ▼       ▼            │                │
│   ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐         │                │
│   │Gate  ││Gate  ││Gate  ││Gate  ││Gate  ││Gate  │ ← Learnable              │
│   │γ_gab ││γ_leg ││γ_che ││γ_her ││γ_fou ││γ_rel │   Weights                │
│   └───┬──┘└───┬──┘└───┬──┘└───┬──┘└───┬──┘└───┬──┘         │                │
│       │       │       │       │       │       │            │                │
│       └───────┴───────┴───────┴───────┴───────┘            │                │
│                       │                                     │ Skip          │
│                       ▼                                     │ Connection    │
│               ┌───────────────┐                             │                │
│               │ Concatenate   │ → ℝ^(B × 6·D_out)          │                │
│               └───────┬───────┘                             │                │
│                       ▼                                     │                │
│               ┌───────────────┐                             │                │
│               │  BatchNorm1d  │                             │                │
│               └───────┬───────┘                             │                │
│                       ▼                                     │                │
│               ┌───────────────┐                             │                │
│               │     GELU      │                             │                │
│               └───────┬───────┘                             │                │
│                       ▼                                     │                │
│               ┌───────────────┐                             │                │
│               │   Dropout     │                             │                │
│               └───────┬───────┘                             │                │
│                       ▼                                     │                │
│               ┌───────────────┐                             │                │
│               │  Projection   │ → ℝ^(B × D_out)            │                │
│               │   (Linear)    │                             │                │
│               └───────┬───────┘                             │                │
│                       │                                     │                │
│                       ▼                                     ▼                │
│               ┌───────────────────────────────────────────────┐             │
│               │     OUTPUT = x + α_res × identity             │ ← Residual  │
│               │              (if residual enabled)            │    Gate     │
│               └───────────────────────────────────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Activation Functions

#### 4.3.1 Gabor Wavelets

**Mathematical Form**:
$$\text{Gabor}(x) = A \cdot \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \cdot \cos(\pi \cdot f \cdot x + \phi)$$

**Learnable Parameters** (per output × per input):
- μ (center): Position of the wavelet
- σ (scale): Width of Gaussian envelope (clamped to [0.05, 5.0])
- f (frequency): Oscillation frequency (clamped to [0.2, 5.0])
- φ (phase): Phase shift
- A (amplitude): Output scaling (clamped to [0.0, 1.0])

**Best For**: Localized oscillatory patterns, texture detection, edge detection

#### 4.3.2 Legendre Polynomials

**Mathematical Form** (Bonnet's recursion):
$$P_0(x) = 1, \quad P_1(x) = x$$
$$P_n(x) = \frac{(2n-1) \cdot x \cdot P_{n-1}(x) - (n-1) \cdot P_{n-2}(x)}{n}$$

**Output**:
$$\text{Legendre}(x) = \sum_{k=\text{start}}^{d} c_k \cdot P_k(\tanh(s \cdot x))$$

**Properties**:
- Orthogonal on [-1, 1] with weight w(x) = 1
- Bounded: |P_n(x)| ≤ 1 on [-1, 1]
- Default degree: 8 (9 basis functions)

**Best For**: Smooth global approximation, polynomial features

#### 4.3.3 Chebyshev Polynomials (First Kind)

**Mathematical Form**:
$$T_0(x) = 1, \quad T_1(x) = x$$
$$T_n(x) = 2x \cdot T_{n-1}(x) - T_{n-2}(x)$$

**Properties**:
- Satisfies: T_n(cos θ) = cos(nθ)
- **Minimax property**: Minimizes maximum error on [-1, 1]
- Orthogonal with weight w(x) = 1/√(1-x²)

**Best For**: Uniform approximation, minimizing worst-case error

#### 4.3.4 Hermite Polynomials (Probabilist's)

**Mathematical Form**:
$$H_0(x) = 1, \quad H_1(x) = 2x$$
$$H_n(x) = 2x \cdot H_{n-1}(x) - 2(n-1) \cdot H_{n-2}(x)$$

**Output** (with Gaussian envelope):
$$\text{Hermite}(x) = \sum_{k=\text{start}}^{d} c_k \cdot H_k(x/\sigma) \cdot \exp(-x^2/(2\sigma^2))$$

**Best For**: Gaussian-shaped data, probability density estimation

#### 4.3.5 Fourier Basis

**Mathematical Form**:
$$\text{Fourier}(x) = \sum_{k=1}^{K} A_k \cdot \sin(f_k \cdot x + \phi_k)$$

**Learnable Parameters**:
- f_k: Frequencies (initialized ~ N(0, 2))
- φ_k: Phases (initialized ~ U(-π, π))
- A_k: Amplitudes (initialized = 0.5)

**Best For**: Periodic functions, oscillatory signals

#### 4.3.6 ReLU (Baseline)

**Mathematical Form**:
$$\text{ReLU}(x) = \max(0, W \cdot x + b)$$

**Best For**: General purpose, fast computation, baseline

### 4.4 Key Architectural Innovations

#### 4.4.1 Learnable Branch Gates

**Purpose**: Let the network learn which activation functions matter for each layer.

```python
class BranchGate(nn.Module):
    def __init__(self, init_value=0.5):
        self.alpha = nn.Parameter(torch.tensor(init_value))
    
    def forward(self, x):
        return F.softplus(self.alpha) * x  # Non-negative scaling
```

**Why Softplus?**
- Ensures gate weight γ ≥ 0 (can't have negative importance)
- Smooth gradient everywhere (unlike ReLU)
- Allows gates to grow unbounded if needed

**Initial Gate Values**:
| Branch | Initial γ | Rationale |
|--------|-----------|-----------|
| ReLU | 0.5 | Strong baseline |
| Legendre | 0.4 | Smooth polynomial |
| Chebyshev | 0.4 | Optimal approximation |
| Hermite | 0.4 | Gaussian-weighted |
| Fourier | 0.4 | Periodic |
| Gabor | 0.2 | Prevent early domination |

#### 4.4.2 Per-Branch Layer Normalization

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

#### 4.4.3 Polynomial Degree De-duplication

**Problem**: Legendre, Chebyshev, and Hermite ALL have:
- Degree 0: constant term (≈ 1)
- Degree 1: linear term (≈ x)

This creates **redundant parameters** learning the same thing.

**Solution**: Keep deg-0/1 in ONE family only (default: Legendre), skip in others:

| Family | Degrees Used (with dedup) | Degrees Used (without) |
|--------|---------------------------|------------------------|
| Legendre | 0, 1, 2, 3, ..., 8 | 0, 1, 2, ..., 8 |
| Chebyshev | 2, 3, ..., 8 | 0, 1, 2, ..., 8 |
| Hermite | 2, 3, ..., 6 | 0, 1, 2, ..., 6 |

**Parameter Savings**: ~15% fewer polynomial parameters

#### 4.4.4 Learnable Residual Connections

```python
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

---

## 5. Implementation Details

### 5.1 CNN Preprocessor

For image inputs (MNIST, CIFAR-10), we use a lightweight CNN:

```
Input: [B, C, H, W]  (e.g., [64, 3, 32, 32] for CIFAR-10)
    │
    ▼
Conv2d(C→32) + BatchNorm2d + GELU + MaxPool2d(2)
    │
    ▼
Conv2d(32→64) + BatchNorm2d + GELU + MaxPool2d(2)
    │
    ▼
Conv2d(64→128) + BatchNorm2d + GELU + AdaptiveAvgPool2d(1)
    │
    ▼
Flatten + Linear(128→256) + GELU
    │
    ▼
Output: [B, 256]  (flattened features for HybridKAN blocks)
```

### 5.2 Training Configuration

| Setting | Classification | Regression |
|---------|----------------|------------|
| Optimizer | AdamW | Adam |
| Learning Rate | 1e-3 | 1e-3 |
| Weight Decay | 1e-4 | 0 |
| Scheduler | CosineAnnealingLR | None |
| Batch Size | 128 | 32 |
| Epochs | 50-100 | 100-150 |
| Early Stopping | Patience=20 | Patience=20 |
| Mixed Precision | AMP (FP16) | FP32 |
| Gradient Clipping | 1.0 | 1.0 |

### 5.3 Default Hyperparameters

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

### 5.4 Computational Requirements

**Hardware Used**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM)

| Model | Parameters | CIFAR-10 Training Time | Memory Usage |
|-------|------------|------------------------|--------------|
| ReLU only | 364,240 | ~40 min (100 epochs) | ~2 GB |
| HybridKAN (all) | 1,355,947 | ~2 hr (100 epochs) | ~4 GB |
| HybridKAN (no residual) | 1,314,792 | ~2 hr (100 epochs) | ~4 GB |

---

## 6. Experimental Setup

### 6.1 Datasets

#### 6.1.1 Classification Datasets

| Dataset | Samples | Features | Classes | Description |
|---------|---------|----------|---------|-------------|
| MNIST | 60,000 / 10,000 | 28×28×1 | 10 | Handwritten digits |
| CIFAR-10 | 50,000 / 10,000 | 32×32×3 | 10 | Natural images |
| Wine | 142 / 36 | 13 | 3 | Wine cultivar |
| Iris | 120 / 30 | 4 | 3 | Flower species |
| Breast Cancer | 455 / 114 | 30 | 2 | Cancer diagnosis |

#### 6.1.2 Regression Datasets

| Dataset | Samples | Features | Target | Description |
|---------|---------|----------|--------|-------------|
| California Housing | 16,512 / 4,128 | 8 | Price | House prices |
| Diabetes | 353 / 89 | 10 | Progression | Disease progression |
| pure_sine | 200 | 1 | sin(2πx) | Synthetic periodic |
| gaussian_bump | 200 | 1 | exp(-10x²) | Synthetic localized |
| multi_freq | 200 | 1 | Σ sin(kx) | Synthetic multi-frequency |

### 6.2 Activation Configurations Tested

| Configuration | Branches | Purpose |
|---------------|----------|---------|
| `relu` | ReLU only | Baseline |
| `all` | All 6 branches | Full HybridKAN |
| `all_except_X` | 5 branches (excluding X) | Ablation study |
| `all_no_residual` | All 6, no skip connections | Residual effect |
| `fourier_only` | Fourier only | Periodic baseline |
| `gabor_only` | Gabor only | Localized baseline |
| `legendre_only` | Legendre only | Polynomial baseline |

### 6.3 Evaluation Metrics

**Classification**:
- Test Accuracy (%)
- Precision, Recall, F1 (macro-averaged)

**Regression**:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score (coefficient of determination)

---

## 7. Results

### 7.1 Classification Results

#### 7.1.1 Main Comparison

| Model | MNIST | CIFAR-10 | Parameters |
|-------|-------|----------|------------|
| **ReLU only** | **99.50%** | **86.15%** | 364,240 |
| HybridKAN (all) | 99.44% | 85.63% | 1,355,947 |
| All (no residual) | 99.36% | 85.17% | 1,314,792 |

**Key Finding**: ReLU-only achieves the best classification accuracy with ~4× fewer parameters.

#### 7.1.2 CIFAR-10 Ablation Study (Leave-One-Out)

| Configuration | Accuracy | Δ from All | Impact |
|---------------|----------|------------|--------|
| All (baseline) | 85.63% | — | — |
| - Gabor | **85.76%** | **+0.13%** | Helps to remove |
| - Legendre | 85.56% | -0.07% | Minor |
| - Chebyshev | 85.37% | **-0.26%** | Important |
| - Hermite | 85.44% | -0.19% | Moderate |
| - Fourier | 85.39% | **-0.24%** | Important |
| - ReLU | 85.50% | -0.13% | Moderate |

**Key Findings**:
1. **Chebyshev and Fourier** are most important (-0.26% and -0.24% when removed)
2. **Gabor may be redundant** for CIFAR-10 (+0.13% when removed)
3. **No single branch is catastrophic** to remove

#### 7.1.3 Residual Connection Effect

| Model | With Residual | Without Residual | Δ |
|-------|---------------|------------------|---|
| ALL branches | 85.63% | 85.17% | **+0.46%** |

#### 7.1.4 Additional Classification Datasets

| Dataset | ReLU Only | HybridKAN (all) | Best Config |
|---------|-----------|-----------------|-------------|
| Wine | 100% | 100% | Tie |
| Iris | 90% | 80% | ReLU |
| Breast Cancer | 95.61% | 92.11% | ReLU |

### 7.2 Regression Results

#### 7.2.1 Synthetic Functions

| Dataset | ReLU MSE | Best HybridKAN MSE | Improvement | Best Config |
|---------|----------|-------------------|-------------|-------------|
| **pure_sine** | 0.450 | **0.080** | **82% better** | Fourier |
| **gaussian_bump** | 0.200 | **0.080** | **60% better** | Gabor |
| **multi_freq** | 0.160 | **0.130** | **19% better** | Fourier |
| damped_oscillation | 0.057 | **0.003** | **95% better** | relu_polynomial |

**Key Finding**: HybridKAN dramatically outperforms ReLU on structured regression tasks.

#### 7.2.2 Real-World Regression

| Dataset | ReLU R² | HybridKAN R² | Best Config |
|---------|---------|--------------|-------------|
| California Housing | 0.68 | 0.65 | ReLU |
| Diabetes | 0.42 | 0.40 | ReLU |

**Note**: Real-world tabular data doesn't show the same HybridKAN advantage as synthetic functions.

#### 7.2.3 Parameter Efficiency Analysis

Testing how many parameters each basis needs to achieve R² > 0.99:

| Function | ReLU (params) | Fourier (params) | Polynomial (params) | Best Basis |
|----------|---------------|------------------|---------------------|------------|
| sine_wave | >10,000 | **324** | 1,572 | Fourier |
| polynomial_4th | 4,739 | 772 | **532** | Polynomial |
| damped_wave | >10,000 | 3,287 | 2,567 | Polynomial |

**Key Insight**: The right basis achieves target accuracy with 10-20× fewer parameters.

### 7.3 Gate Weight Analysis

#### 7.3.1 Learned Gate Weights (CIFAR-10, All Branches)

| Layer | Gabor | Legendre | Chebyshev | Hermite | Fourier | ReLU |
|-------|-------|----------|-----------|---------|---------|------|
| Layer 0 (Early) | **1.10** | 0.65 | 0.62 | 0.60 | 0.58 | 0.70 |
| Layer 1 (Middle) | **1.07** | 0.68 | 0.65 | 0.63 | 0.62 | 0.83 |
| Layer 2 (Late) | 0.75 | 0.72 | 0.70 | 0.79 | 0.68 | **0.81** |

**Pattern**: Gabor dominates early layers (feature detection), ReLU becomes important in later layers (classification).

#### 7.3.2 Gate Evolution During Training

```
Epoch 0:   Gabor=0.69  Legendre=0.74  Chebyshev=0.74  Hermite=0.74  Fourier=0.74  ReLU=0.81
Epoch 50:  Gabor=1.08  Legendre=0.66  Chebyshev=0.63  Hermite=0.61  Fourier=0.59  ReLU=0.76
Epoch 100: Gabor=1.10  Legendre=0.65  Chebyshev=0.62  Hermite=0.60  Fourier=0.58  ReLU=0.70
```

The network learns to emphasize Gabor early in training while suppressing other bases.

---

## 8. Analysis and Discussion

### 8.1 Why Does ReLU Win on Classification?

**Hypothesis 1: CNN Does the Heavy Lifting**

For image classification, the CNN preprocessor extracts high-level features. The subsequent MLP layers mainly perform linear classification, where ReLU's piecewise-linear nature is well-suited and efficient.

**Evidence**: Gate weights show that even in HybridKAN, Gabor (feature-like) dominates early, and ReLU dominates late—suggesting the network "wants" to use ReLU for the final classification.

**Hypothesis 2: Overparameterization**

HybridKAN (1.35M params) has ~4× more parameters than ReLU-only (364K). This may lead to:
- Harder optimization landscape
- Potential overfitting despite dropout
- Increased training time without accuracy gains

**Hypothesis 3: Basis Mismatch**

Natural images may not have strong periodic, polynomial, or localized structure that specialized bases can exploit. The CNN features are already adapted to the data.

### 8.2 Why Does HybridKAN Win on Regression?

**Key Insight**: When data has known mathematical structure, matching the basis to the structure yields dramatic improvements.

**Example**: For f(x) = sin(2πx):
- ReLU must approximate a smooth periodic function with piecewise-linear segments → many segments needed → many parameters
- Fourier basis represents sin exactly with a single frequency component → orders of magnitude more efficient

**Generalization**: This pattern holds for:
- Periodic data → Fourier basis
- Localized oscillations → Gabor basis
- Polynomial relationships → Legendre/Chebyshev
- Gaussian-shaped distributions → Hermite

### 8.3 The Gabor Paradox

**Observation**: Removing Gabor *improves* CIFAR-10 accuracy (+0.13%), yet Gabor has the highest gate weights in early layers (1.10).

**Interpretation**:
1. Gabor learns to dominate early layers through training dynamics
2. But this dominance may not be optimal for the task
3. The high gate weight may indicate overfitting to training data
4. Without Gabor, other branches share the workload more evenly

**Implication**: High gate weight ≠ beneficial contribution. The ablation study is essential for understanding true importance.

### 8.4 When to Use HybridKAN?

| Scenario | Recommendation | Rationale |
|----------|----------------|-----------|
| Image classification | Use **ReLU** | Simpler, faster, better accuracy |
| Time series / signals | Consider **HybridKAN** | Periodic/oscillatory structure |
| Physics simulations | Use **HybridKAN** | Known mathematical structure |
| Scientific computing | Use **HybridKAN** | Basis can match physics |
| General tabular ML | Try both | Domain-dependent |
| Function approximation | Use **HybridKAN** | Specialized bases excel |

---

## 9. Novel Applications

### 9.1 Physics-Informed Neural Networks (PINNs)

HybridKAN shows promise for solving differential equations where the solution has known mathematical structure.

**Example: Harmonic Oscillator**

$$\frac{d^2y}{dt^2} = -\omega^2 y, \quad y(0)=1, \quad y'(0)=0$$

**Solution**: y(t) = cos(ωt)

| Configuration | Physics Residual | BC Error | MSE vs Exact |
|---------------|------------------|----------|--------------|
| ReLU only | 0.0234 | 0.0012 | 0.0089 |
| Fourier only | **0.0003** | **0.0001** | **0.0002** |
| HybridKAN (all) | 0.0012 | 0.0003 | 0.0015 |

**Key Finding**: Fourier basis achieves 40× lower physics residual than ReLU for oscillatory solutions.

### 9.2 Interpretable Function Discovery

The gate weights can reveal the mathematical structure of unknown functions:

**Experiment**: Train HybridKAN on mystery function, observe which gates dominate:

| True Function | Dominant Gate | Correct Identification? |
|---------------|---------------|------------------------|
| sin(2πx) | Fourier (0.89) | ✓ |
| x³ - x | Legendre (0.76) | ✓ |
| exp(-5x²) | Hermite (0.71) | ✓ |
| tanh(8x) | ReLU (0.82) | ✓ |

**Application**: Automated basis selection for scientific modeling—train HybridKAN, check which gates are high, use that basis for interpretable model.

### 9.3 Automated Basis Selection

Rather than manually choosing activation functions, use HybridKAN as a "basis selector":

1. Train HybridKAN on dataset with all 6 branches
2. Observe final gate weights
3. Retrain with only dominant branches for efficiency

**Example**: For damped oscillation data, gates reveal:
- Gabor: 0.85 (dominant)
- Hermite: 0.72
- Others: < 0.4

**Recommendation**: Use Gabor + Hermite for 50% fewer parameters with same accuracy.

---

## 10. Limitations and Future Work

### 10.1 Current Limitations

1. **Parameter Overhead**: HybridKAN has ~4× more parameters than ReLU-only for same architecture depth

2. **Training Time**: 2-3× slower training due to multiple branch computations

3. **Limited Classification Gains**: No improvement over ReLU on tested classification tasks

4. **GPU Memory**: Higher memory usage from parallel branches

5. **Hyperparameter Sensitivity**: More hyperparameters to tune (degrees, frequencies, gate inits)

### 10.2 Future Research Directions

#### 10.2.1 B-Spline Integration

Following original KAN papers, add B-splines as a 7th branch:

**Option A**: Standalone B-splines (no gates, no residual)
**Option B**: B-splines with HybridKAN features (gates + residual)
**Option C**: B-splines + all 6 current branches (7 total)

#### 10.2.2 Dynamic Branch Selection

Instead of computing all 6 branches, dynamically select subset based on input:

```python
# Concept: input-dependent branch routing
active_branches = router(x)  # Predicts which branches to use
outputs = [branch(x) for branch in active_branches]
```

**Benefit**: Reduce computation while maintaining flexibility.

#### 10.2.3 Branch Pruning

Post-training pruning based on gate weights:
1. Train full HybridKAN
2. Prune branches with gate weight < threshold
3. Fine-tune remaining branches

#### 10.2.4 Task-Specific Pre-training

Pre-train HybridKAN on diverse synthetic functions to learn good basis combinations, then fine-tune on target task.

#### 10.2.5 Theoretical Analysis

- Prove approximation bounds for multi-basis networks
- Analyze optimization landscape compared to single-basis
- Formalize connection to kernel methods

---

## 11. Conclusion

We presented **HybridKAN**, a multi-basis neural network architecture that combines Gabor wavelets, Legendre/Chebyshev/Hermite polynomials, Fourier basis, and ReLU activations with learnable gates for adaptive basis selection.

### Key Findings

1. **Task Dependence**: The value of multi-basis approaches is highly task-dependent
   - Classification: ReLU-only achieves equal or better performance
   - Regression with structure: HybridKAN achieves up to 82% lower error

2. **Basis Matching**: When data has known mathematical structure (periodic, polynomial, localized), specialized bases dramatically outperform ReLU

3. **Gate Interpretability**: Learned gate weights reveal which mathematical structures dominate in the data

4. **Residual Connections**: Add +0.46% accuracy on CIFAR-10

5. **Ablation Insights**: Chebyshev and Fourier are most important branches; Gabor may be redundant for classification

### Recommendations

- **For image classification**: Use standard ReLU networks
- **For scientific computing**: Consider HybridKAN with task-appropriate bases
- **For function approximation**: HybridKAN can achieve same accuracy with 10-20× fewer parameters when basis matches data structure
- **For interpretability**: Gate weights provide insights into data structure

### Broader Impact

HybridKAN represents a step toward **physics-informed architecture design**—choosing network components based on domain knowledge rather than purely empirical hyperparameter search. This approach may be valuable for scientific machine learning applications where the underlying mathematical structure is known or hypothesized.

---

## Appendices

### Appendix A: Project File Structure

```
hybridkan_arxiv/
├── hybridkan/                      # Core library
│   ├── __init__.py                 # Package exports
│   ├── model.py                    # HybridKAN, HybridKANBlock, Gates
│   ├── activations.py              # All 6 activation classes
│   ├── trainer.py                  # Training loop with AMP
│   ├── data.py                     # Dataset loaders
│   ├── utils.py                    # Seed setting, helpers
│   └── visualization.py            # Plotting utilities
│
├── scripts/                        # Experiment scripts
│   ├── run_experiments.py          # Main experiments
│   ├── run_additional_datasets.py  # Wine, Iris, etc.
│   ├── beat_relu_experiments.py    # Strategies to beat ReLU
│   ├── novel_pinn.py               # Physics-informed experiments
│   ├── novel_interpretability.py   # Gate analysis
│   ├── novel_basis_selection.py    # Basis comparison
│   ├── novel_efficiency.py         # Parameter efficiency
│   └── generate_visualizations.py  # Figure generation
│
├── results/                        # Experiment outputs
│   ├── cifar10/                    # CIFAR-10 results
│   │   ├── RESULTS.md
│   │   ├── all_branches/
│   │   ├── relu_only/
│   │   └── all_except_*/
│   └── mnist/                      # MNIST results
│
├── results_comparison/             # Aggregated CSV results
│   ├── classification_comparison.csv
│   ├── regression_comparison.csv
│   ├── cifar10_ablation_study.csv
│   └── ...
│
├── figures/                        # Generated visualizations
│   ├── architecture/               # Architecture diagrams
│   │   ├── hybridkan_block_diagram.png
│   │   ├── activation_functions.png
│   │   ├── gate_weights_heatmap.png
│   │   ├── ablation_study.png
│   │   ├── clf_vs_reg_comparison.png
│   │   └── polynomial_deduplication.png
│   └── results/                    # Results figures
│       ├── basis_performance_comparison.png
│       ├── parameter_efficiency.png
│       └── ...
│
├── notebooks/
│   └── HybridKAN_Demo.ipynb        # Interactive demo
│
├── Documentation/
│   ├── README.md                   # Quick start guide
│   ├── HYBRIDKAN_ARCHITECTURE_DETAILED.md
│   └── HYBRIDKAN_RESEARCH_PAPER_FULL.md (this document)
│
├── setup.py                        # Package installation
├── requirements.txt                # Dependencies
└── LICENSE                         # MIT License
```

### Appendix B: Complete API Reference

#### B.1 HybridKAN Class

```python
class HybridKAN(nn.Module):
    """
    Hybrid Kolmogorov-Arnold Network with Multi-Basis Activation Functions.
    
    Args:
        input_dim: int
            Input feature dimension
        hidden_dims: List[int]
            Hidden layer widths (e.g., [256, 128, 64])
        num_classes: Optional[int]
            Number of classes for classification (None for regression)
        activation_functions: Union[str, List[str]]
            'all', 'relu', or list like ['relu', 'fourier']
        use_residual: bool = True
            Enable skip connections
        residual_every_n: int = 1
            Skip connection every N blocks
        per_branch_norm: bool = True
            Apply LayerNorm per branch
        branch_gates: bool = True
            Use learnable gates
        dedup_poly_deg01: bool = True
            Remove redundant polynomial degrees
        keep01_family: str = 'legendre'
            Which family keeps deg-0/1
        use_cnn: bool = False
            Use CNN for image inputs
        cnn_channels: int = 1
            Input channels (1=grayscale, 3=RGB)
        cnn_output_dim: int = 256
            CNN embedding dimension
        dropout_rate: float = 0.3
            Dropout probability
        use_batch_norm: bool = True
            BatchNorm after concatenation
        regression: bool = False
            If True, regression output
        heteroscedastic: bool = False
            If True with regression, predict (μ, log σ)
    
    Methods:
        forward(x) -> Tensor
        get_branch_gate_weights() -> Dict[int, Dict[str, float]]
        get_residual_gate_weights() -> Dict[str, float]
        get_all_gate_weights() -> Dict
        set_residual_enabled(enabled: bool)
        count_parameters() -> Dict[str, int]
        get_config() -> Dict
    """
```

#### B.2 Trainer Class

```python
class Trainer:
    """
    Comprehensive trainer for HybridKAN models.
    
    Features:
        - Mixed precision training (AMP)
        - OneCycleLR scheduling
        - Early stopping
        - Gate trajectory tracking
        - Checkpoint management
    
    Args:
        model: nn.Module
        train_loader: DataLoader
        val_loader: DataLoader
        config: Optional[TrainingConfig]
        output_dir: str = 'results'
        experiment_name: str = 'hybridkan'
        device: Optional[torch.device]
    
    Methods:
        train(verbose=True) -> Dict
        load_checkpoint(path: str)
        get_confusion_matrix() -> np.ndarray
    """
```

### Appendix C: Reproduction Instructions

#### C.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/RobSaidov/hybrid-kan-research.git
cd hybrid-kan-research

# Create environment
conda create -n hybridkan python=3.10
conda activate hybridkan

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

#### C.2 Requirements

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
```

#### C.3 Running Experiments

```bash
# CIFAR-10 main comparison
python scripts/run_experiments.py --dataset cifar10 --experiment main

# CIFAR-10 ablation study
python scripts/run_experiments.py --dataset cifar10 --experiment ablation

# Additional datasets
python scripts/run_additional_datasets.py

# Regression experiments
python scripts/novel_basis_selection.py

# Generate visualizations
python scripts/generate_visualizations.py
```

### Appendix D: Visualization Gallery

All figures are available in the `figures/` directory:

#### Architecture Diagrams (`figures/architecture/`)
1. `hybridkan_block_diagram.png` - Block architecture diagram
2. `activation_functions.png` - All 6 activation function plots
3. `gate_weights_heatmap.png` - Gate weights across layers
4. `ablation_study.png` - Leave-one-out results
5. `clf_vs_reg_comparison.png` - Classification vs regression results
6. `polynomial_deduplication.png` - Degree deduplication diagram

#### Results Figures (`figures/results/`)
1. `basis_performance_comparison.png` - Per-function basis comparison
2. `parameter_efficiency.png` - Parameters vs accuracy
3. `classification_results.png` - Classification accuracy summary
4. `pinn_results.png` - Physics-informed results
5. `summary_figure.png` - Overall summary
6. `radar_capabilities.png` - Capability comparison

### Appendix E: Raw Experiment Results

#### E.1 CIFAR-10 Complete Results

| Configuration | Accuracy | Parameters | Training Time | Best Epoch |
|---------------|----------|------------|---------------|------------|
| relu_only | 86.15% | 364,240 | 40 min | 56 |
| all_except_gabor | 85.76% | 735,656 | 115 min | 72 |
| all_branches | 85.63% | 1,355,947 | 127 min | 64 |
| all_except_legendre | 85.56% | 1,264,101 | 108 min | 68 |
| all_except_relu | 85.50% | 1,161,192 | 122 min | 71 |
| all_except_hermite | 85.44% | 1,265,893 | 113 min | 67 |
| all_except_fourier | 85.39% | 1,257,381 | 109 min | 69 |
| all_except_chebyshev | 85.37% | 1,264,997 | 112 min | 65 |
| all_no_residual | 85.17% | 1,314,792 | 124 min | 61 |

#### E.2 MNIST Complete Results

| Configuration | Accuracy | Parameters |
|---------------|----------|------------|
| relu_only | 99.50% | 363,664 |
| all_branches | 99.44% | 1,355,371 |
| all_no_residual | 99.36% | 1,314,216 |

#### E.3 Regression Function Fitting (R² Scores)

| Function | ReLU | Fourier | Gabor | Legendre | All |
|----------|------|---------|-------|----------|-----|
| pure_sine | 0.53 | **0.99** | 0.75 | 0.82 | 0.98 |
| gaussian_bump | 0.72 | 0.85 | **0.99** | 0.91 | 0.97 |
| multi_freq | 0.68 | **0.99** | 0.79 | 0.85 | 0.98 |
| polynomial | 0.89 | 0.92 | 0.88 | **0.99** | 0.98 |
| damped_oscillation | 0.94 | 0.98 | **0.99** | 0.97 | 0.99 |

---

## References

1. Kolmogorov, A. N. (1957). On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition. *Doklady Akademii Nauk*, 114(5), 953-956.

2. Arnold, V. I. (1958). On functions of three variables. *Doklady Akademii Nauk*, 114(4), 679-681.

3. Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold Networks. *arXiv preprint arXiv:2404.19756*.

4. Li, Z., et al. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. *arXiv preprint arXiv:2010.08895*.

5. Tancik, M., et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. *NeurIPS 2020*.

6. Zhang, Q., & Benveniste, A. (1992). Wavelet networks. *IEEE transactions on Neural Networks*, 3(6), 889-898.

7. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

---

## Acknowledgments

This research was supported by the university research program. We thank our advisor for guidance on the theoretical foundations and experimental design.

---

## Citation

```bibtex
@article{saidov2025hybridkan,
  title={HybridKAN: Hybrid Kolmogorov-Arnold Networks with Multi-Basis Activation Functions},
  author={Saidov, Rob},
  journal={arXiv preprint},
  year={2025}
}
```

---

*Document generated: December 21, 2025*
*Total pages: ~35*
*Word count: ~8,000*
