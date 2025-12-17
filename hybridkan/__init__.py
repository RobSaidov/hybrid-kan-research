# -*- coding: utf-8 -*-
"""
HybridKAN: Hybrid Kolmogorov-Arnold Networks with Multi-Basis Activation Functions

A research implementation combining localized (Gabor, Hermite), polynomial (Legendre, Chebyshev),
periodic (Fourier), and piecewise-linear (ReLU) representations with optional residual connections.

Author: Rob (San Francisco Bay University)
Research Supervisor: Dr. Bandari
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Rob"

from .model import HybridKAN
from .activations import (
    GaborActivation,
    LegendreActivation,
    ChebyshevActivation,
    HermiteActivation,
    FourierActivation,
    ReLUActivation,
)
from .trainer import Trainer
from .utils import set_seed, get_device
from .data import (
    get_data_loaders,
    get_mnist_loaders,
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_wine_loaders,
    get_iris_loaders,
    get_california_housing_loaders,
    get_diabetes_loaders,
    get_custom_loaders,
    get_dataset_info,
    list_datasets,
    DATASET_INFO,
)

__all__ = [
    # Model
    "HybridKAN",
    # Activations
    "GaborActivation",
    "LegendreActivation",
    "ChebyshevActivation",
    "HermiteActivation",
    "FourierActivation",
    "ReLUActivation",
    # Training
    "Trainer",
    # Utilities
    "set_seed",
    "get_device",
    # Data Loaders
    "get_data_loaders",
    "get_mnist_loaders",
    "get_cifar10_loaders",
    "get_cifar100_loaders",
    "get_wine_loaders",
    "get_iris_loaders",
    "get_california_housing_loaders",
    "get_diabetes_loaders",
    "get_custom_loaders",
    "get_dataset_info",
    "list_datasets",
    "DATASET_INFO",
]
