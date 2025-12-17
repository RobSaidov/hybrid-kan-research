# -*- coding: utf-8 -*-
"""
HybridKAN Multi-Dataset Data Loaders

Supports:
- Classification: MNIST, CIFAR-10, CIFAR-100, Wine, Iris, Satellite (EuroSAT)
- Regression: California Housing, Diabetes

Each loader returns (train_loader, test_loader) with consistent interface.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from typing import Tuple, Optional, Dict, Any
import warnings

# Try importing torchvision
try:
    import torchvision
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, CIFAR100
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn("torchvision not available. Image datasets will not work.")

# Try importing sklearn for tabular datasets
try:
    from sklearn.datasets import (
        load_wine, load_iris, load_diabetes, 
        fetch_california_housing
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Tabular datasets will not work.")


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

DATASET_INFO = {
    # Classification datasets
    "mnist": {
        "type": "classification",
        "num_classes": 10,
        "input_dim": 784,
        "channels": 1,
        "image_size": 28,
        "description": "Handwritten digits 0-9",
    },
    "cifar10": {
        "type": "classification", 
        "num_classes": 10,
        "input_dim": 3072,
        "channels": 3,
        "image_size": 32,
        "description": "10-class natural images",
    },
    "cifar100": {
        "type": "classification",
        "num_classes": 100,
        "input_dim": 3072,
        "channels": 3,
        "image_size": 32,
        "description": "100-class natural images",
    },
    "wine": {
        "type": "classification",
        "num_classes": 3,
        "input_dim": 13,
        "channels": None,
        "description": "Wine cultivar classification (13 chemical features)",
    },
    "iris": {
        "type": "classification",
        "num_classes": 3,
        "input_dim": 4,
        "channels": None,
        "description": "Iris flower species (4 features)",
    },
    # Regression datasets
    "california_housing": {
        "type": "regression",
        "num_classes": None,
        "input_dim": 8,
        "channels": None,
        "description": "California house prices (8 features)",
    },
    "diabetes": {
        "type": "regression",
        "num_classes": None,
        "input_dim": 10,
        "channels": None,
        "description": "Diabetes progression (10 features)",
    },
}


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Get metadata about a dataset."""
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_INFO.keys())}")
    return DATASET_INFO[name]


# =============================================================================
# TRANSFORM UTILITIES
# =============================================================================

class FlattenTransform:
    """Flatten image tensor to 1D vector."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)


class AddChannelTransform:
    """Add channel dimension if missing."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 2 else x


# =============================================================================
# IMAGE DATASETS (Classification)
# =============================================================================

def get_mnist_loaders(
    train_size: int = 60000,
    batch_size: int = 128,
    augment: bool = True,
    use_cnn: bool = True,
    num_workers: int = 2,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset.
    
    Args:
        train_size: Number of training samples (max 60000)
        batch_size: Batch size for DataLoader
        augment: Apply data augmentation
        use_cnn: If True, return [1, 28, 28] images; else [784] vectors
        num_workers: DataLoader workers
        data_dir: Directory to store/load data
        
    Returns:
        (train_loader, test_loader)
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision required for MNIST")
    
    test_size = min(10000, int(train_size * 0.2))
    
    if use_cnn:
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        flatten = FlattenTransform()
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                flatten,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                flatten,
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            flatten,
        ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, transform=train_transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, transform=test_transform, download=True
    )
    
    if train_size < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:train_size]
        train_dataset = Subset(train_dataset, indices)
    
    if test_size < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:test_size]
        test_dataset = Subset(test_dataset, indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, test_loader


def get_cifar10_loaders(
    train_size: int = 50000,
    batch_size: int = 128,
    augment: bool = True,
    use_cnn: bool = True,
    num_workers: int = 2,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset.
    
    Args:
        train_size: Number of training samples (max 50000)
        batch_size: Batch size
        augment: Apply augmentation (RandomCrop, HorizontalFlip)
        use_cnn: If True, return [3, 32, 32]; else [3072] vectors
        num_workers: DataLoader workers
        data_dir: Data directory
        
    Returns:
        (train_loader, test_loader)
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision required for CIFAR-10")
    
    test_size = min(10000, int(train_size * 0.2))
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    if use_cnn:
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        flatten = FlattenTransform()
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            flatten,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            flatten,
        ])
    
    train_dataset = CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)
    
    if train_size < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:train_size]
        train_dataset = Subset(train_dataset, indices)
    
    if test_size < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:test_size]
        test_dataset = Subset(test_dataset, indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, test_loader


def get_cifar100_loaders(
    train_size: int = 50000,
    batch_size: int = 128,
    augment: bool = True,
    use_cnn: bool = True,
    num_workers: int = 2,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-100 dataset (100 classes).
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision required for CIFAR-100")
    
    test_size = min(10000, int(train_size * 0.2))
    
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    if use_cnn:
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        flatten = FlattenTransform()
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            flatten,
        ])
        test_transform = train_transform
    
    train_dataset = CIFAR100(root=data_dir, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR100(root=data_dir, train=False, transform=test_transform, download=True)
    
    if train_size < len(train_dataset):
        train_dataset = Subset(train_dataset, torch.randperm(len(train_dataset))[:train_size])
    if test_size < len(test_dataset):
        test_dataset = Subset(test_dataset, torch.randperm(len(test_dataset))[:test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


# =============================================================================
# TABULAR CLASSIFICATION DATASETS
# =============================================================================

def get_wine_loaders(
    batch_size: int = 32,
    test_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load Wine dataset (3-class classification, 13 features).
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required for Wine dataset")
    
    data = load_wine()
    X, y = data.data, data.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state, stratify=y
    )
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_iris_loaders(
    batch_size: int = 16,
    test_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load Iris dataset (3-class classification, 4 features).
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required for Iris dataset")
    
    data = load_iris()
    X, y = data.data, data.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state, stratify=y
    )
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# =============================================================================
# REGRESSION DATASETS
# =============================================================================

def get_california_housing_loaders(
    batch_size: int = 64,
    test_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load California Housing dataset (regression, 8 features).
    Target: Median house value (in $100,000s)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required for California Housing")
    
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state
    )
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Store scalers for inverse transform
    train_loader.scaler_X = scaler_X
    train_loader.scaler_y = scaler_y
    test_loader.scaler_X = scaler_X
    test_loader.scaler_y = scaler_y
    
    return train_loader, test_loader


def get_diabetes_loaders(
    batch_size: int = 32,
    test_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load Diabetes dataset (regression, 10 features).
    Target: Quantitative measure of disease progression.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required for Diabetes dataset")
    
    data = load_diabetes()
    X, y = data.data, data.target
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state
    )
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    train_loader.scaler_X = scaler_X
    train_loader.scaler_y = scaler_y
    test_loader.scaler_X = scaler_X
    test_loader.scaler_y = scaler_y
    
    return train_loader, test_loader


# =============================================================================
# CUSTOM DATASET LOADER
# =============================================================================

def get_custom_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    batch_size: int = 64,
    task: str = "classification",
    normalize: bool = True,
    test_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders from custom numpy arrays.
    
    Args:
        X_train: Training features [N, D] or [N, C, H, W]
        y_train: Training targets [N] or [N, 1]
        X_test: Test features (if None, split from train)
        y_test: Test targets
        batch_size: Batch size
        task: "classification" or "regression"
        normalize: Apply StandardScaler to features
        test_split: Fraction for test if X_test is None
        
    Returns:
        (train_loader, test_loader)
    """
    if X_test is None:
        stratify = y_train if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_split, stratify=stratify, random_state=42
        )
    
    if normalize and X_train.ndim == 2:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)
    
    if task == "classification":
        y_train_t = torch.LongTensor(y_train)
        y_test_t = torch.LongTensor(y_test)
    else:
        y_train_t = torch.FloatTensor(y_train).flatten()
        y_test_t = torch.FloatTensor(y_test).flatten()
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# =============================================================================
# UNIFIED LOADER INTERFACE
# =============================================================================

def get_data_loaders(
    dataset: str,
    batch_size: int = 128,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Unified interface to load any supported dataset.
    
    Args:
        dataset: Dataset name (case-insensitive)
        batch_size: Batch size
        **kwargs: Dataset-specific arguments
        
    Returns:
        (train_loader, test_loader)
        
    Example:
        >>> train_loader, test_loader = get_data_loaders("cifar10", batch_size=128)
        >>> train_loader, test_loader = get_data_loaders("wine", batch_size=32)
        >>> train_loader, test_loader = get_data_loaders("california_housing")
    """
    dataset = dataset.lower().replace("-", "_").replace(" ", "_")
    
    loaders = {
        "mnist": get_mnist_loaders,
        "cifar10": get_cifar10_loaders,
        "cifar_10": get_cifar10_loaders,
        "cifar100": get_cifar100_loaders,
        "cifar_100": get_cifar100_loaders,
        "wine": get_wine_loaders,
        "iris": get_iris_loaders,
        "california_housing": get_california_housing_loaders,
        "california": get_california_housing_loaders,
        "housing": get_california_housing_loaders,
        "diabetes": get_diabetes_loaders,
    }
    
    if dataset not in loaders:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {sorted(set(loaders.keys()))}"
        )
    
    return loaders[dataset](batch_size=batch_size, **kwargs)


def list_datasets():
    """Print information about all available datasets."""
    print("\n" + "=" * 70)
    print("HybridKAN Supported Datasets")
    print("=" * 70)
    
    print("\n📊 CLASSIFICATION DATASETS:")
    print("-" * 70)
    for name, info in DATASET_INFO.items():
        if info["type"] == "classification":
            print(f"  {name:20s} | Classes: {info['num_classes']:3d} | "
                  f"Features: {info['input_dim']:5d} | {info['description']}")
    
    print("\n📈 REGRESSION DATASETS:")
    print("-" * 70)
    for name, info in DATASET_INFO.items():
        if info["type"] == "regression":
            print(f"  {name:20s} | Features: {info['input_dim']:5d} | {info['description']}")
    
    print("\n" + "=" * 70)
    print("Usage: train_loader, test_loader = get_data_loaders('dataset_name')")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    list_datasets()
