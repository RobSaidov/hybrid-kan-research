# -*- coding: utf-8 -*-
"""
HybridKAN Model Architecture

Core implementation of the Hybrid Kolmogorov-Arnold Network with:
- Multi-basis parallel branches (Gabor, Legendre, Chebyshev, Hermite, Fourier, ReLU)
- Per-branch LayerNorm and learnable gates
- Optional residual (skip) connections with learnable weights
- Optional CNN preprocessing for image inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Union
from collections import OrderedDict

from .activations import (
    GaborActivation,
    LegendreActivation,
    ChebyshevActivation,
    HermiteActivation,
    FourierActivation,
    ReLUActivation,
    ACTIVATION_REGISTRY,
    BRANCH_DEFAULTS,
    CANONICAL_BRANCHES,
)


class BranchGate(nn.Module):
    """
    Learnable scalar gate for branch importance weighting.
    
    Uses softplus to ensure non-negativity while maintaining gradient flow.
    """
    
    def __init__(self, init_value: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_value)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.alpha) * x
    
    @property
    def weight(self) -> float:
        """Returns the effective gate weight (after softplus)."""
        with torch.no_grad():
            return F.softplus(self.alpha).item()
    
    @property
    def raw_weight(self) -> float:
        """Returns the raw parameter value."""
        return self.alpha.item()


class ResidualGate(nn.Module):
    """
    Learnable gate for residual/skip connection strength.
    
    Controls the contribution of skip connections in the network.
    Can be monitored to understand information flow patterns.
    """
    
    def __init__(self, init_value: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_value)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.alpha) * x
    
    @property
    def weight(self) -> float:
        """Returns the effective gate weight (after sigmoid, range 0-1)."""
        with torch.no_grad():
            return torch.sigmoid(self.alpha).item()


class CNNPreprocessor(nn.Module):
    """
    Lightweight CNN feature extractor for image inputs.
    
    Architecture: 3 conv blocks → adaptive pooling → projection
    Suitable for MNIST (1 channel) and CIFAR-10 (3 channels).
    """
    
    def __init__(self, in_channels: int = 1, output_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim
        
        # Progressive channel expansion: in → 32 → 64 → 128
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        return self.projection(x)


class HybridKANBlock(nn.Module):
    """
    Single HybridKAN layer block.
    
    Computes parallel branch outputs → optional per-branch LayerNorm →
    optional gates → concatenate → BatchNorm → GELU → Dropout → projection
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        branches: List[str],
        branch_configs: Dict,
        start_degrees: Dict[str, int],
        per_branch_norm: bool = True,
        branch_gates: bool = True,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.branch_names = branches
        self.per_branch_norm = per_branch_norm
        self.branch_gates = branch_gates
        
        # Build branches
        self.branches = nn.ModuleDict()
        self.branch_norms = nn.ModuleDict() if per_branch_norm else None
        self.gates = nn.ModuleDict() if branch_gates else None
        
        for name in branches:
            config = branch_configs.get(name, {})
            
            # Create activation module
            if name == "gabor":
                module = GaborActivation(
                    in_features, out_features,
                    amp_init=config.get("amp_init", 0.1),
                    sigma_init=config.get("sigma_init", 1.0),
                    freq_init=config.get("freq_init", 1.0),
                )
            elif name == "legendre":
                module = LegendreActivation(
                    in_features, out_features,
                    degree=config.get("degree", 8),
                    start_degree=start_degrees.get("legendre", 0),
                )
            elif name == "chebyshev":
                module = ChebyshevActivation(
                    in_features, out_features,
                    degree=config.get("degree", 8),
                    start_degree=start_degrees.get("chebyshev", 0),
                )
            elif name == "hermite":
                module = HermiteActivation(
                    in_features, out_features,
                    degree=config.get("degree", 6),
                    start_degree=start_degrees.get("hermite", 0),
                )
            elif name == "fourier":
                module = FourierActivation(
                    in_features, out_features,
                    n_frequencies=config.get("n_frequencies", 8),
                )
            elif name == "relu":
                module = ReLUActivation(in_features, out_features)
            else:
                raise ValueError(f"Unknown branch type: {name}")
            
            self.branches[name] = module
            
            if per_branch_norm:
                self.branch_norms[name] = nn.LayerNorm(out_features)
            
            if branch_gates:
                gate_init = config.get("gate_init", 0.4)
                self.gates[name] = BranchGate(init_value=gate_init)
        
        # Post-concatenation layers
        total_features = out_features * len(branches)
        self.batch_norm = nn.BatchNorm1d(total_features) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(total_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        for name in self.branch_names:
            out = self.branches[name](x)
            
            if self.per_branch_norm and self.branch_norms is not None:
                out = self.branch_norms[name](out)
            
            if self.branch_gates and self.gates is not None:
                out = self.gates[name](out)
            
            outputs.append(out)
        
        # Concatenate branch outputs
        combined = torch.cat(outputs, dim=1)
        
        if self.batch_norm is not None:
            combined = self.batch_norm(combined)
        
        combined = F.gelu(combined)
        combined = self.dropout(combined)
        
        return F.gelu(self.projection(combined))
    
    def get_gate_weights(self) -> Dict[str, float]:
        """Returns current gate weights for all branches."""
        if not self.branch_gates or self.gates is None:
            return {}
        return {name: self.gates[name].weight for name in self.branch_names}


class HybridKAN(nn.Module):
    """
    Hybrid Kolmogorov-Arnold Network with Multi-Basis Activation Functions.
    
    Key Features:
    - Parallel multi-basis branches (Gabor, Legendre, Chebyshev, Hermite, Fourier, ReLU)
    - Per-branch LayerNorm and learnable gates for adaptive branch importance
    - Optional residual skip connections with learnable gates
    - Polynomial degree de-duplication across families
    - Optional CNN preprocessing for image inputs
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer widths (e.g., [256, 128, 64])
        num_classes: Number of output classes (classification) or None for regression
        activation_functions: Branch selection - 'all', 'relu', or list like ['relu', 'fourier']
        
        # ResNet configuration
        use_residual: Enable skip connections between blocks
        residual_every_n: Add skip connection every N blocks (default: 1 = every block)
        
        # Normalization and gating
        per_branch_norm: Apply LayerNorm per branch before gating
        branch_gates: Use learnable scalar gates per branch
        
        # Polynomial de-duplication
        dedup_poly_deg01: Remove redundant deg-0/1 terms across polynomial families
        keep01_family: Which family keeps deg-0/1 ('legendre', 'chebyshev', or 'hermite')
        
        # CNN preprocessing
        use_cnn: Use CNN feature extractor for image inputs
        cnn_channels: Input channels for CNN (1 for grayscale, 3 for RGB)
        cnn_output_dim: CNN output embedding dimension
        
        # Training
        dropout_rate: Dropout probability
        use_batch_norm: Use BatchNorm after branch concatenation
        
        # Regression
        regression: If True, output is regression value(s)
        heteroscedastic: If True with regression, predict (μ, log σ) for uncertainty
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: Optional[int] = None,
        activation_functions: Union[str, List[str]] = "all",
        
        # ResNet configuration
        use_residual: bool = True,
        residual_every_n: int = 1,
        
        # Normalization and gating
        per_branch_norm: bool = True,
        branch_gates: bool = True,
        
        # Polynomial de-duplication
        dedup_poly_deg01: bool = True,
        keep01_family: str = "legendre",
        
        # CNN preprocessing
        use_cnn: bool = False,
        cnn_channels: int = 1,
        cnn_output_dim: int = 256,
        
        # Training
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        
        # Regression
        regression: bool = False,
        heteroscedastic: bool = False,
    ):
        super().__init__()
        
        # Store configuration
        self.regression = regression
        self.heteroscedastic = heteroscedastic
        self.use_residual = use_residual
        self.residual_every_n = residual_every_n
        self.per_branch_norm = per_branch_norm
        self.branch_gates = branch_gates
        
        # Resolve activation functions
        self.active_branches = self._resolve_branches(activation_functions)
        if not self.active_branches:
            raise ValueError(f"No valid branches from: {activation_functions}")
        
        # Compute start degrees for polynomial de-duplication
        self.start_degrees = self._compute_start_degrees(dedup_poly_deg01, keep01_family)
        
        # CNN preprocessing
        self.use_cnn = use_cnn
        if use_cnn:
            self.cnn = CNNPreprocessor(in_channels=cnn_channels, output_dim=cnn_output_dim)
            actual_input_dim = cnn_output_dim
        else:
            self.cnn = None
            actual_input_dim = input_dim
        
        # Input normalization
        self.input_norm = nn.LayerNorm(actual_input_dim)
        
        # Build hidden blocks
        self.blocks = nn.ModuleList()
        self.residual_gates = nn.ModuleDict()
        self.residual_projections = nn.ModuleDict()
        
        prev_dim = actual_input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            block = HybridKANBlock(
                in_features=prev_dim,
                out_features=hidden_dim,
                branches=self.active_branches,
                branch_configs=BRANCH_DEFAULTS,
                start_degrees=self.start_degrees,
                per_branch_norm=per_branch_norm,
                branch_gates=branch_gates,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
            )
            self.blocks.append(block)
            
            # Residual connections
            if use_residual and (i + 1) % residual_every_n == 0:
                gate_key = f"residual_gate_{i}"
                self.residual_gates[gate_key] = ResidualGate(init_value=0.1)
                
                # Projection if dimensions don't match
                if prev_dim != hidden_dim:
                    proj_key = f"residual_proj_{i}"
                    self.residual_projections[proj_key] = nn.Linear(prev_dim, hidden_dim)
            
            prev_dim = hidden_dim
        
        # Output head
        if regression:
            out_dim = 2 if heteroscedastic else 1
            self.output_head = nn.Linear(prev_dim, out_dim)
        else:
            if num_classes is None:
                raise ValueError("num_classes required for classification")
            self.output_head = nn.Linear(prev_dim, num_classes)
    
    def _resolve_branches(self, spec: Union[str, List[str]]) -> List[str]:
        """Resolve branch specification to list of canonical branch names."""
        if isinstance(spec, str):
            spec = spec.lower().strip()
            if spec == "all":
                return CANONICAL_BRANCHES.copy()
            elif spec.startswith("all_except_"):
                exclude = spec.replace("all_except_", "")
                return [b for b in CANONICAL_BRANCHES if b != exclude]
            else:
                # Single branch
                if spec in CANONICAL_BRANCHES:
                    return [spec]
                raise ValueError(f"Unknown branch: {spec}")
        
        # List of branches
        branches = []
        for b in spec:
            b_lower = b.lower().strip()
            if b_lower in CANONICAL_BRANCHES:
                branches.append(b_lower)
        return branches if branches else ["relu"]
    
    def _compute_start_degrees(self, dedup: bool, keep_family: str) -> Dict[str, int]:
        """Compute polynomial start degrees for de-duplication."""
        poly_families = ["legendre", "chebyshev", "hermite"]
        start_degrees = {f: 0 for f in poly_families}
        
        if dedup:
            keep_family = keep_family.lower()
            if keep_family not in poly_families:
                keep_family = "legendre"
            for f in poly_families:
                if f != keep_family:
                    start_degrees[f] = 2  # Skip deg-0 (constant) and deg-1 (linear)
        
        return start_degrees
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN preprocessing
        if self.use_cnn and self.cnn is not None:
            x = self.cnn(x)
        
        # Input normalization
        x = self.input_norm(x)
        
        # Process through blocks with optional residual connections
        for i, block in enumerate(self.blocks):
            identity = x
            x = block(x)
            
            # Apply residual connection
            if self.use_residual and (i + 1) % self.residual_every_n == 0:
                gate_key = f"residual_gate_{i}"
                proj_key = f"residual_proj_{i}"
                
                if gate_key in self.residual_gates:
                    # Project identity if needed
                    if proj_key in self.residual_projections:
                        identity = self.residual_projections[proj_key](identity)
                    
                    # Gated residual addition
                    x = x + self.residual_gates[gate_key](identity)
        
        # Output head
        out = self.output_head(x)
        
        if not self.regression:
            return F.log_softmax(out, dim=1)
        elif self.heteroscedastic:
            return out  # [B, 2] for (μ, log σ)
        else:
            return out.squeeze(-1)
    
    def get_branch_gate_weights(self) -> Dict[int, Dict[str, float]]:
        """
        Returns gate weights for all branches in all blocks.
        
        Returns:
            Dict mapping block_index → {branch_name: gate_weight}
        """
        weights = {}
        for i, block in enumerate(self.blocks):
            weights[i] = block.get_gate_weights()
        return weights
    
    def get_residual_gate_weights(self) -> Dict[str, float]:
        """
        Returns weights for all residual gates.
        
        Returns:
            Dict mapping gate_key → weight (0-1 range)
        """
        weights = {}
        for key, gate in self.residual_gates.items():
            weights[key] = gate.weight
        return weights
    
    def get_all_gate_weights(self) -> Dict[str, any]:
        """
        Returns comprehensive gate information.
        
        Returns:
            Dict with 'branch_gates' and 'residual_gates' keys
        """
        return {
            "branch_gates": self.get_branch_gate_weights(),
            "residual_gates": self.get_residual_gate_weights(),
        }
    
    def set_residual_enabled(self, enabled: bool):
        """Toggle residual connections on/off at runtime."""
        self.use_residual = enabled
    
    def count_parameters(self) -> Dict[str, int]:
        """Returns parameter counts by component."""
        counts = {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        
        if self.cnn is not None:
            counts["cnn"] = sum(p.numel() for p in self.cnn.parameters())
        
        counts["blocks"] = sum(p.numel() for p in self.blocks.parameters())
        counts["output_head"] = sum(p.numel() for p in self.output_head.parameters())
        
        return counts
    
    def get_config(self) -> Dict:
        """Returns model configuration for reproducibility."""
        return {
            "active_branches": self.active_branches,
            "use_residual": self.use_residual,
            "residual_every_n": self.residual_every_n,
            "per_branch_norm": self.per_branch_norm,
            "branch_gates": self.branch_gates,
            "start_degrees": self.start_degrees,
            "use_cnn": self.use_cnn,
            "regression": self.regression,
            "heteroscedastic": self.heteroscedastic,
            "num_blocks": len(self.blocks),
        }
