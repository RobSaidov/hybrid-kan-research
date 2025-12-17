#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HybridKAN Training Examples

This script demonstrates how to train HybridKAN on various datasets:
1. CIFAR-10 (main image classification)
2. Wine (tabular classification)  
3. California Housing (regression)
4. Custom dataset

Run with: python examples/train_examples.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybridkan import (
    HybridKAN, Trainer, set_seed,
    get_data_loaders, get_dataset_info, list_datasets
)


def train_cifar10():
    """
    Example 1: CIFAR-10 Image Classification (your main dataset)
    
    10 classes, 32x32 RGB images
    Recommended: use_cnn=True, augment=True
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: CIFAR-10 Classification")
    print("=" * 60)
    
    set_seed(42)
    
    # Load CIFAR-10 data
    train_loader, test_loader = get_data_loaders(
        "cifar10",
        batch_size=128,
        train_size=50000,  # Full dataset
        augment=True,
        use_cnn=True,
    )
    
    # Get dataset info
    info = get_dataset_info("cifar10")
    print(f"Dataset: CIFAR-10")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Image size: {info['channels']} x {info['image_size']} x {info['image_size']}")
    
    # Create model with ALL branches + ResNet connections
    model = HybridKAN(
        input_dim=info['input_dim'],
        hidden_dims=[512, 256, 128],  # Deeper for CIFAR-10
        num_classes=info['num_classes'],
        activation_functions='all',   # All 6 branches
        use_cnn=True,
        cnn_channels=info['channels'],
        cnn_output_dim=384,
        use_residual=True,
        residual_every_n=1,
        per_branch_norm=True,
        branch_gates=True,
        dropout_rate=0.3,
    )
    
    print(f"\nModel Configuration:")
    print(f"  Branches: {model.active_branches}")
    print(f"  ResNet: {model.use_residual}")
    print(f"  Parameters: {model.count_parameters()['total']:,}")
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=100,
        learning_rate=1e-3,
        patience=15,
        use_amp=True,
    )
    
    results = trainer.train()
    
    print(f"\nResults:")
    print(f"  Best Accuracy: {results['best_accuracy']:.2f}%")
    print(f"  Best Epoch: {results['best_epoch']}")
    
    # Show final gate weights
    print(f"\nFinal Branch Gates (Block 0):")
    for branch, weight in model.get_branch_gate_weights()[0].items():
        print(f"  {branch}: {weight:.4f}")
    
    return model, results


def train_wine():
    """
    Example 2: Wine Classification (tabular data)
    
    3 classes, 13 chemical features
    No CNN needed (tabular)
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Wine Classification")
    print("=" * 60)
    
    set_seed(42)
    
    train_loader, test_loader = get_data_loaders(
        "wine",
        batch_size=32,
    )
    
    info = get_dataset_info("wine")
    print(f"Dataset: Wine")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Features: {info['input_dim']}")
    
    # Smaller model for tabular data
    model = HybridKAN(
        input_dim=info['input_dim'],
        hidden_dims=[64, 32],
        num_classes=info['num_classes'],
        activation_functions='all',
        use_cnn=False,  # No CNN for tabular
        use_residual=True,
        dropout_rate=0.2,
    )
    
    print(f"\nModel Parameters: {model.count_parameters()['total']:,}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=100,
        learning_rate=1e-3,
        patience=20,
    )
    
    results = trainer.train()
    print(f"\nBest Accuracy: {results['best_accuracy']:.2f}%")
    
    return model, results


def train_california_housing():
    """
    Example 3: California Housing Regression
    
    Regression task, 8 features
    Target: Median house value
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: California Housing Regression")
    print("=" * 60)
    
    set_seed(42)
    
    train_loader, test_loader = get_data_loaders(
        "california_housing",
        batch_size=64,
    )
    
    info = get_dataset_info("california_housing")
    print(f"Dataset: California Housing")
    print(f"  Task: {info['type']}")
    print(f"  Features: {info['input_dim']}")
    
    # Regression model
    model = HybridKAN(
        input_dim=info['input_dim'],
        hidden_dims=[128, 64, 32],
        num_classes=None,       # Not needed for regression
        regression=True,        # Enable regression mode
        heteroscedastic=False,  # Homoscedastic (single output)
        activation_functions='all',
        use_cnn=False,
        use_residual=True,
        dropout_rate=0.1,
    )
    
    print(f"\nModel Parameters: {model.count_parameters()['total']:,}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=100,
        learning_rate=1e-3,
        patience=20,
        task='regression',
    )
    
    results = trainer.train()
    print(f"\nBest MSE: {results.get('best_mse', 'N/A')}")
    print(f"Best R²: {results.get('best_r2', 'N/A')}")
    
    return model, results


def train_custom_dataset():
    """
    Example 4: Custom Dataset
    
    Load your own data from numpy arrays
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Dataset")
    print("=" * 60)
    
    import numpy as np
    from hybridkan import get_custom_loaders
    
    set_seed(42)
    
    # Generate synthetic data (replace with your data)
    n_samples = 1000
    n_features = 20
    n_classes = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    print(f"Custom Dataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")
    
    # Create loaders
    train_loader, test_loader = get_custom_loaders(
        X_train=X,
        y_train=y,
        batch_size=64,
        task="classification",
        normalize=True,
        test_split=0.2,
    )
    
    # Build model
    model = HybridKAN(
        input_dim=n_features,
        hidden_dims=[64, 32],
        num_classes=n_classes,
        activation_functions='all',
        use_residual=True,
    )
    
    print(f"\nModel Parameters: {model.count_parameters()['total']:,}")
    
    # Note: In practice, you would train this
    print("(Custom model created - ready for training)")
    
    return model


def run_branch_comparison():
    """
    Example 5: Compare different branch configurations
    
    Useful for ablation studies
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Branch Configuration Comparison")
    print("=" * 60)
    
    set_seed(42)
    
    # Quick test on MNIST
    train_loader, test_loader = get_data_loaders(
        "mnist",
        batch_size=128,
        train_size=5000,  # Small subset for quick comparison
        use_cnn=True,
    )
    
    info = get_dataset_info("mnist")
    
    configurations = [
        ('ReLU only', ['relu']),
        ('Polynomial', ['legendre', 'chebyshev']),
        ('Fourier + ReLU', ['fourier', 'relu']),
        ('All branches', 'all'),
    ]
    
    results_table = []
    
    for name, branches in configurations:
        print(f"\nTraining: {name}...")
        
        model = HybridKAN(
            input_dim=info['input_dim'],
            hidden_dims=[128, 64],
            num_classes=info['num_classes'],
            activation_functions=branches,
            use_cnn=True,
            cnn_channels=info['channels'],
            use_residual=True,
        )
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=20,  # Quick comparison
            learning_rate=1e-3,
            patience=5,
        )
        
        res = trainer.train()
        results_table.append({
            'config': name,
            'branches': branches if isinstance(branches, str) else ', '.join(branches),
            'accuracy': res['best_accuracy'],
            'params': model.count_parameters()['total'],
        })
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Configuration':<20} | {'Accuracy':>10} | {'Parameters':>12}")
    print("-" * 60)
    for r in results_table:
        print(f"{r['config']:<20} | {r['accuracy']:>9.2f}% | {r['params']:>12,}")


if __name__ == "__main__":
    # Show available datasets
    list_datasets()
    
    # Uncomment the examples you want to run:
    
    # train_cifar10()        # Main image classification
    # train_wine()           # Tabular classification
    # train_california_housing()  # Regression
    # train_custom_dataset()  # Custom data
    # run_branch_comparison()  # Ablation study
    
    print("\n✓ Edit this file to uncomment the experiments you want to run!")
