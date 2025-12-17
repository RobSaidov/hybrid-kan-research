# -*- coding: utf-8 -*-
"""
Utility Functions for HybridKAN

General-purpose utilities for:
- Reproducibility (seeding)
- Device management
- Data loading helpers
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: If True, use CUDA when available
        
    Returns:
        torch.device for computation
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameter_count(count: int) -> str:
    """Format parameter count with appropriate suffix."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    return str(count)
