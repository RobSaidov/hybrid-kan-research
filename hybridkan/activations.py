# -*- coding: utf-8 -*-
"""
Activation Functions for HybridKAN

This module implements the multi-basis activation functions:
- Gabor: Localized, orientation/frequency-selective atoms
- Legendre/Chebyshev: Orthogonal polynomial bases for smooth global structure
- Hermite: Polynomial × Gaussian for localized curvature under Gaussian measure
- Fourier: Sinusoidal bank for periodic structure
- ReLU: Piecewise-linear baseline

Each branch maps R^{D_in} → R^{H} via learned parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GaborActivation(nn.Module):
    """
    Gabor wavelet activation with per-output × per-input parameters.
    
    Computes: amplitude × exp(-0.5 × ((x - μ)/σ)²) × cos(π × freq × x + phase)
    Output is summed over input dimension.
    
    Safe initialization ensures stable early training:
    - Small amplitudes (default 0.1)
    - Moderate frequencies (clamped 0.2-5.0)
    - Bounded sigma (clamped 0.05-5.0)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension (number of Gabor units)
        amp_init: Initial amplitude (default 0.1)
        sigma_init: Initial standard deviation (default 1.0)
        freq_init: Initial frequency (default 1.0)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        amp_init: float = 0.10,
        sigma_init: float = 1.0,
        freq_init: float = 1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters: [out_features, in_features]
        self.mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.freq = nn.Parameter(torch.full((out_features, in_features), freq_init))
        self.phase = nn.Parameter(torch.zeros(out_features, in_features))
        self.amplitude = nn.Parameter(torch.full((out_features, in_features), amp_init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, in_features]
        Returns:
            Output tensor [batch_size, out_features]
        """
        # Expand for broadcasting: [B, 1, D_in]
        x_exp = x.unsqueeze(1)
        
        # Clamp parameters for numerical stability
        mu = self.mu.unsqueeze(0)  # [1, D_out, D_in]
        sigma = torch.clamp(self.sigma.unsqueeze(0), 0.05, 5.0)
        freq = torch.clamp(self.freq.unsqueeze(0), 0.2, 5.0)
        phase = self.phase.unsqueeze(0)
        amp = torch.clamp(self.amplitude.unsqueeze(0), 0.0, 1.0)
        
        # Gaussian envelope
        x_norm = (x_exp - mu) / (sigma + 1e-6)
        gaussian = torch.exp(-0.5 * torch.clamp(x_norm ** 2, max=50.0))
        
        # Oscillatory component
        oscillation = torch.cos(math.pi * freq * x_exp + phase)
        
        # Combine and sum over input dimension
        output = (amp * gaussian * oscillation).sum(dim=2)  # [B, D_out]
        return output
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class LegendreActivation(nn.Module):
    """
    Legendre polynomial basis with trainable mixing coefficients.
    
    Uses stable Bonnet's recursion:
        P_0(x) = 1
        P_1(x) = x
        P_n(x) = ((2n-1) × x × P_{n-1}(x) - (n-1) × P_{n-2}(x)) / n
    
    Supports `start_degree` to skip lower degrees (for de-duplication across families).
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        degree: Maximum polynomial degree (inclusive)
        start_degree: First degree to include (0=constant, 1=linear, 2=quadratic, ...)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 8,
        start_degree: int = 0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = int(degree)
        self.start_degree = int(start_degree)
        
        width = self.degree - self.start_degree + 1
        self.coeffs = nn.Parameter(torch.randn(out_features, width) * 0.1)
        self.input_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale and compress input to [-1, 1]
        scale = torch.clamp(self.input_scale, 0.1, 2.0)
        x_scaled = torch.tanh(x * scale)
        
        # Build polynomial stack via stable recursion
        P0 = torch.ones_like(x_scaled)
        polys = [P0]
        
        if self.degree >= 1:
            P1 = x_scaled
            polys.append(P1)
            for n in range(2, self.degree + 1):
                Pn = ((2 * n - 1) * x_scaled * polys[-1] - (n - 1) * polys[-2]) / n
                polys.append(torch.clamp(Pn, -100.0, 100.0))
        
        # Stack and slice from start_degree: [B, width, D_in]
        poly_stack = torch.stack(polys[self.start_degree:self.degree + 1], dim=1)
        
        # Weighted combination
        c = self.coeffs.unsqueeze(0)  # [1, D_out, width]
        p = poly_stack.unsqueeze(1)   # [B, 1, width, D_in]
        weighted = (p * c.unsqueeze(-1)).sum(dim=2)  # [B, D_out, D_in]
        
        return weighted.sum(dim=-1)  # [B, D_out]
    
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"degree={self.degree}, start_degree={self.start_degree}")


class ChebyshevActivation(nn.Module):
    """
    Chebyshev polynomial (first kind) basis with trainable mixing.
    
    Uses stable recursion:
        T_0(x) = 1
        T_1(x) = x
        T_n(x) = 2x × T_{n-1}(x) - T_{n-2}(x)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        degree: Maximum polynomial degree (inclusive)
        start_degree: First degree to include
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 8,
        start_degree: int = 0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = int(degree)
        self.start_degree = int(start_degree)
        
        width = self.degree - self.start_degree + 1
        self.coeffs = nn.Parameter(torch.randn(out_features, width) * 0.1)
        self.input_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.clamp(self.input_scale, 0.1, 2.0)
        x_scaled = torch.tanh(x * scale)
        
        # Chebyshev recursion
        T0 = torch.ones_like(x_scaled)
        polys = [T0]
        
        if self.degree >= 1:
            T1 = x_scaled
            polys.append(T1)
            T_prev, T_cur = T0, T1
            for _ in range(2, self.degree + 1):
                T_next = 2 * x_scaled * T_cur - T_prev
                polys.append(torch.clamp(T_next, -100.0, 100.0))
                T_prev, T_cur = T_cur, T_next
        
        poly_stack = torch.stack(polys[self.start_degree:self.degree + 1], dim=1)
        c = self.coeffs.unsqueeze(0)
        p = poly_stack.unsqueeze(1)
        weighted = (p * c.unsqueeze(-1)).sum(dim=2)
        
        return weighted.sum(dim=-1)
    
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"degree={self.degree}, start_degree={self.start_degree}")


class HermiteActivation(nn.Module):
    """
    Probabilists' Hermite polynomial basis with Gaussian envelope.
    
    Computes: H_n(x/σ) × exp(-x²/(2σ²))
    
    Uses recursion:
        H_0(x) = 1
        H_1(x) = 2x
        H_n(x) = 2x × H_{n-1}(x) - 2(n-1) × H_{n-2}(x)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        degree: Maximum polynomial degree (inclusive)
        start_degree: First degree to include
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 6,
        start_degree: int = 0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = int(degree)
        self.start_degree = int(start_degree)
        
        width = self.degree - self.start_degree + 1
        self.coeffs = nn.Parameter(torch.randn(out_features, width) * 0.1)
        self.sigma = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.clamp(self.sigma, 0.1, 5.0)
        x_scaled = x / sigma
        
        # Hermite recursion
        H0 = torch.ones_like(x_scaled)
        polys = [H0]
        
        if self.degree >= 1:
            H1 = 2 * x_scaled
            polys.append(H1)
            for n in range(2, self.degree + 1):
                Hn = 2 * x_scaled * polys[-1] - 2 * (n - 1) * polys[-2]
                polys.append(torch.clamp(Hn, -100.0, 100.0))
        
        poly_stack = torch.stack(polys[self.start_degree:self.degree + 1], dim=1)
        
        # Apply Gaussian envelope
        gaussian = torch.exp(-torch.clamp(x_scaled ** 2, max=50.0))
        poly_stack = poly_stack * gaussian.unsqueeze(1)
        
        c = self.coeffs.unsqueeze(0)
        p = poly_stack.unsqueeze(1)
        weighted = (p * c.unsqueeze(-1)).sum(dim=2)
        
        return weighted.sum(dim=-1)
    
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"degree={self.degree}, start_degree={self.start_degree}")


class FourierActivation(nn.Module):
    """
    Fourier basis with learnable frequencies, phases, and amplitudes.
    
    Computes: Σ_k amplitude_k × sin(frequency_k × x + phase_k)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        n_frequencies: Number of frequency components
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_frequencies: int = 8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_frequencies = int(n_frequencies)
        
        # Parameters: [out_features, n_frequencies]
        self.frequencies = nn.Parameter(torch.randn(out_features, n_frequencies) * 2.0)
        self.phases = nn.Parameter(torch.randn(out_features, n_frequencies) * math.pi)
        self.amplitudes = nn.Parameter(torch.ones(out_features, n_frequencies) * 0.5)
        self.input_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x * torch.clamp(self.input_scale, 0.1, 2.0)
        x_scaled = x_scaled.unsqueeze(1)  # [B, 1, D_in]
        
        freq = self.frequencies.unsqueeze(0)   # [1, D_out, n_f]
        phase = self.phases.unsqueeze(0)
        amp = self.amplitudes.unsqueeze(0)
        
        # Sinusoidal computation: [B, D_out, n_f, D_in]
        sin_term = torch.sin(freq.unsqueeze(-1) * x_scaled.unsqueeze(2) + phase.unsqueeze(-1))
        components = amp.unsqueeze(-1) * sin_term
        
        # Sum over frequencies and input dimensions
        return components.sum(dim=2).sum(dim=-1)  # [B, D_out]
    
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"n_frequencies={self.n_frequencies}")


class ReLUActivation(nn.Module):
    """
    Simple Linear → ReLU branch for baseline comparison.
    
    Maintains signature parity with other activation families.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# Registry of activation classes
ACTIVATION_REGISTRY = {
    "gabor": GaborActivation,
    "legendre": LegendreActivation,
    "chebyshev": ChebyshevActivation,
    "hermite": HermiteActivation,
    "fourier": FourierActivation,
    "relu": ReLUActivation,
}

# Default configurations for each branch
BRANCH_DEFAULTS = {
    "gabor": {
        "gate_init": 0.2,
        "amp_init": 0.10,
        "sigma_init": 1.0,
        "freq_init": 1.0,
    },
    "legendre": {
        "gate_init": 0.4,
        "degree": 8,
    },
    "chebyshev": {
        "gate_init": 0.4,
        "degree": 8,
    },
    "hermite": {
        "gate_init": 0.4,
        "degree": 6,
    },
    "fourier": {
        "gate_init": 0.4,
        "n_frequencies": 8,
    },
    "relu": {
        "gate_init": 0.5,
    },
}

# Canonical order for consistent reporting
CANONICAL_BRANCHES = ["gabor", "legendre", "chebyshev", "hermite", "fourier", "relu"]
