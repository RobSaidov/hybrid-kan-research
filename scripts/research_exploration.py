#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HybridKAN Research Exploration - Finding Where Multi-Basis Shines

This script explores MULTIPLE angles to find where HybridKAN provides value:

1. PURE MLP (No CNN) - Let the branches do the work, not convolutions
2. SYNTHETIC TASKS - Functions that NEED different bases (polynomials, waves, etc.)
3. TABULAR DATA - Where neural nets traditionally struggle
4. INTERPRETABILITY - What do the gate weights tell us?
5. ROBUSTNESS - Is HybridKAN more robust to noise?
6. FEW-SHOT LEARNING - Does it generalize better with less data?
7. CONVERGENCE ANALYSIS - Does it learn faster?
8. BRANCH DISCOVERY - Which branches matter for which tasks?

The hypothesis: ReLU wins on images because CNNs already extract features.
HybridKAN should shine when the ACTIVATION FUNCTIONS need to model complex patterns.
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hybridkan import HybridKAN, set_seed

try:
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import (
        load_wine, load_iris, load_breast_cancer,
        make_classification, make_regression,
        fetch_california_housing,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# SYNTHETIC FUNCTION DATASETS - WHERE BASES SHOULD MATTER!
# =============================================================================

class SyntheticDataset:
    """Generate synthetic datasets that NEED specific basis functions."""
    
    @staticmethod
    def polynomial_function(n_samples: int = 5000, noise: float = 0.1):
        """
        Target: y = 0.5*x^3 - 2*x^2 + x + 0.3
        Legendre/Chebyshev/Hermite should excel here!
        """
        X = np.random.uniform(-2, 2, (n_samples, 1)).astype(np.float32)
        y = 0.5 * X**3 - 2 * X**2 + X + 0.3
        y += np.random.normal(0, noise, y.shape).astype(np.float32)
        return X, y.flatten(), "Polynomial (cubic)"
    
    @staticmethod
    def sinusoidal_function(n_samples: int = 5000, noise: float = 0.1):
        """
        Target: y = sin(2πx) + 0.5*sin(4πx)
        Fourier should absolutely dominate here!
        """
        X = np.random.uniform(0, 2, (n_samples, 1)).astype(np.float32)
        y = np.sin(2 * np.pi * X) + 0.5 * np.sin(4 * np.pi * X)
        y += np.random.normal(0, noise, y.shape).astype(np.float32)
        return X, y.flatten(), "Sinusoidal (multi-frequency)"
    
    @staticmethod
    def gaussian_mixture(n_samples: int = 5000, noise: float = 0.1):
        """
        Target: sum of Gaussian bumps
        Gabor (localized oscillations) should help!
        """
        X = np.random.uniform(-3, 3, (n_samples, 1)).astype(np.float32)
        y = (np.exp(-X**2) + 0.5*np.exp(-(X-1.5)**2) + 0.5*np.exp(-(X+1.5)**2))
        y += np.random.normal(0, noise, y.shape).astype(np.float32)
        return X, y.flatten(), "Gaussian Mixture"
    
    @staticmethod
    def mixed_function(n_samples: int = 5000, noise: float = 0.1):
        """
        Target: polynomial + sinusoid + Gaussian
        This NEEDS multiple basis types!
        """
        X = np.random.uniform(-2, 2, (n_samples, 1)).astype(np.float32)
        poly = 0.3 * X**2
        wave = 0.5 * np.sin(3 * np.pi * X)
        gaussian = np.exp(-2 * X**2)
        y = poly + wave + gaussian
        y += np.random.normal(0, noise, y.shape).astype(np.float32)
        return X, y.flatten(), "Mixed (poly+sin+gaussian)"
    
    @staticmethod
    def high_frequency(n_samples: int = 5000, noise: float = 0.05):
        """
        Target: High frequency oscillations
        Tests if Fourier can capture rapid changes
        """
        X = np.random.uniform(0, 1, (n_samples, 1)).astype(np.float32)
        y = np.sin(20 * np.pi * X) * np.exp(-3 * X)
        y += np.random.normal(0, noise, y.shape).astype(np.float32)
        return X, y.flatten(), "High Frequency Damped"
    
    @staticmethod
    def step_function(n_samples: int = 5000, noise: float = 0.1):
        """
        Discontinuous function - hard for smooth bases
        """
        X = np.random.uniform(-2, 2, (n_samples, 1)).astype(np.float32)
        y = np.where(X > 0, 1.0, -1.0) + 0.3 * X
        y += np.random.normal(0, noise, y.shape).astype(np.float32)
        return X, y.flatten().astype(np.float32), "Step + Linear"
    
    @staticmethod
    def multivariate_complex(n_samples: int = 5000, n_features: int = 5, noise: float = 0.1):
        """
        Multi-dimensional function with interactions
        y = sin(x1) + x2^2 + exp(-x3^2) + x4*x5
        """
        X = np.random.uniform(-2, 2, (n_samples, n_features)).astype(np.float32)
        y = (np.sin(2 * np.pi * X[:, 0]) + 
             0.5 * X[:, 1]**2 + 
             np.exp(-X[:, 2]**2) + 
             0.3 * X[:, 3] * X[:, 4])
        y += np.random.normal(0, noise, y.shape).astype(np.float32)
        return X, y, "Multivariate Complex"


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

class RegressionExperiment:
    """Run regression experiments comparing branches."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results = []
    
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_name: str,
        branches_list: List,
        hidden_dims: List[int] = [64, 32],
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> Dict:
        """Run regression experiment with multiple branch configurations."""
        
        # Normalize
        X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
        y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
        
        X_train_norm = (X_train - X_mean) / X_std
        X_test_norm = (X_test - X_mean) / X_std
        y_train_norm = (y_train - y_mean) / y_std
        y_test_norm = (y_test - y_mean) / y_std
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train_norm).unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_norm),
            torch.FloatTensor(y_test_norm).unsqueeze(1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256)
        
        input_dim = X_train.shape[1]
        task_results = {'task': task_name, 'experiments': []}
        
        for branches in branches_list:
            set_seed(42)
            
            branch_name = branches if isinstance(branches, str) else '_'.join(branches)
            
            # Create model
            model = HybridKAN(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=1,
                activation_functions=branches,
                use_cnn=False,
                use_residual=True,
                dropout_rate=0.1,
                regression=True,
            )
            model = model.to(self.device)
            
            # Train
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            
            best_mse = float('inf')
            train_losses = []
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = F.mse_loss(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                scheduler.step()
                train_losses.append(epoch_loss / len(train_loader))
                
                # Evaluate
                model.eval()
                test_preds = []
                test_targets = []
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.to(self.device)
                        pred = model(X_batch)
                        test_preds.append(pred.cpu().numpy())
                        test_targets.append(y_batch.numpy())
                
                test_preds = np.concatenate(test_preds).flatten()
                test_targets = np.concatenate(test_targets).flatten()
                
                # Denormalize for true MSE
                test_preds_denorm = test_preds * y_std + y_mean
                test_targets_denorm = test_targets * y_std + y_mean
                mse = mean_squared_error(test_targets_denorm, test_preds_denorm)
                
                if mse < best_mse:
                    best_mse = mse
                    best_r2 = r2_score(test_targets_denorm, test_preds_denorm)
            
            # Get gate weights
            gate_weights = model.get_branch_gate_weights()
            
            result = {
                'branches': branch_name,
                'best_mse': float(best_mse),
                'best_r2': float(best_r2),
                'params': sum(p.numel() for p in model.parameters()),
                'gate_weights': gate_weights,
                'final_train_loss': train_losses[-1],
            }
            
            task_results['experiments'].append(result)
            print(f"  {branch_name:30s} MSE: {best_mse:.6f}  R2: {best_r2:.4f}")
        
        return task_results


class ClassificationExperiment:
    """Run classification experiments."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_name: str,
        branches_list: List,
        num_classes: int,
        hidden_dims: List[int] = [64, 32],
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> Dict:
        """Run classification experiment."""
        
        # Normalize
        X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
        X_train_norm = (X_train - X_mean) / X_std
        X_test_norm = (X_test - X_mean) / X_std
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_norm),
            torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256)
        
        input_dim = X_train.shape[1]
        task_results = {'task': task_name, 'experiments': []}
        
        for branches in branches_list:
            set_seed(42)
            
            branch_name = branches if isinstance(branches, str) else '_'.join(branches)
            
            model = HybridKAN(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                activation_functions=branches,
                use_cnn=False,
                use_residual=True,
                dropout_rate=0.2,
            )
            model = model.to(self.device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            
            best_acc = 0
            
            for epoch in range(epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = F.nll_loss(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                # Evaluate
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.to(self.device)
                        pred = model(X_batch).argmax(dim=1)
                        correct += (pred.cpu() == y_batch).sum().item()
                        total += y_batch.size(0)
                
                acc = 100 * correct / total
                best_acc = max(best_acc, acc)
            
            gate_weights = model.get_branch_gate_weights()
            
            result = {
                'branches': branch_name,
                'best_accuracy': float(best_acc),
                'params': sum(p.numel() for p in model.parameters()),
                'gate_weights': gate_weights,
            }
            
            task_results['experiments'].append(result)
            print(f"  {branch_name:30s} Accuracy: {best_acc:.2f}%")
        
        return task_results


class RobustnessExperiment:
    """Test robustness to input noise."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_name: str,
        branches_list: List,
        noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
        hidden_dims: List[int] = [64, 32],
        epochs: int = 150,
    ) -> Dict:
        """Train on clean data, test on noisy data."""
        
        X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
        y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
        
        X_train_norm = (X_train - X_mean) / X_std
        y_train_norm = (y_train - y_mean) / y_std
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train_norm).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        input_dim = X_train.shape[1]
        task_results = {'task': task_name, 'experiments': []}
        
        for branches in branches_list:
            set_seed(42)
            branch_name = branches if isinstance(branches, str) else '_'.join(branches)
            
            model = HybridKAN(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=1,
                activation_functions=branches,
                use_cnn=False,
                use_residual=True,
                dropout_rate=0.1,
                regression=True,
            )
            model = model.to(self.device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            
            # Train on clean data
            for epoch in range(epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = F.mse_loss(pred, y_batch)
                    loss.backward()
                    optimizer.step()
            
            # Test on various noise levels
            noise_results = {}
            model.eval()
            
            for noise in noise_levels:
                X_test_noisy = X_test + np.random.normal(0, noise, X_test.shape).astype(np.float32)
                X_test_norm = (X_test_noisy - X_mean) / X_std
                
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test_norm).to(self.device)
                    pred = model(X_tensor).cpu().numpy().flatten()
                
                pred_denorm = pred * y_std + y_mean
                mse = mean_squared_error(y_test, pred_denorm)
                noise_results[f'noise_{noise}'] = float(mse)
            
            result = {
                'branches': branch_name,
                'noise_mse': noise_results,
                'degradation': noise_results['noise_0.5'] / (noise_results['noise_0.0'] + 1e-8),
            }
            
            task_results['experiments'].append(result)
            print(f"  {branch_name:25s} Clean: {noise_results['noise_0.0']:.4f}  Noisy(0.5): {noise_results['noise_0.5']:.4f}  Degradation: {result['degradation']:.2f}x")
        
        return task_results


class FewShotExperiment:
    """Test generalization with limited training data."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_name: str,
        branches_list: List,
        sample_sizes: List[int] = [50, 100, 200, 500, 1000],
        hidden_dims: List[int] = [32, 16],
        epochs: int = 200,
    ) -> Dict:
        """Test performance with varying amounts of training data."""
        
        # Fixed test set (last 1000 samples)
        X_test = X[-1000:]
        y_test = y[-1000:]
        X_pool = X[:-1000]
        y_pool = y[:-1000]
        
        task_results = {'task': task_name, 'experiments': []}
        
        for branches in branches_list:
            branch_name = branches if isinstance(branches, str) else '_'.join(branches)
            size_results = {}
            
            for n_samples in sample_sizes:
                set_seed(42)
                
                # Random subset for training
                indices = np.random.choice(len(X_pool), min(n_samples, len(X_pool)), replace=False)
                X_train = X_pool[indices]
                y_train = y_pool[indices]
                
                # Normalize
                X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
                y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
                
                X_train_norm = (X_train - X_mean) / X_std
                X_test_norm = (X_test - X_mean) / X_std
                y_train_norm = (y_train - y_mean) / y_std
                
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train_norm),
                    torch.FloatTensor(y_train_norm).unsqueeze(1)
                )
                train_loader = DataLoader(train_dataset, batch_size=min(32, n_samples), shuffle=True)
                
                model = HybridKAN(
                    input_dim=X.shape[1],
                    hidden_dims=hidden_dims,
                    num_classes=1,
                    activation_functions=branches,
                    use_cnn=False,
                    use_residual=True,
                    dropout_rate=0.1,
                    regression=True,
                )
                model = model.to(self.device)
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                
                for epoch in range(epochs):
                    model.train()
                    for X_batch, y_batch in train_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        optimizer.zero_grad()
                        pred = model(X_batch)
                        loss = F.mse_loss(pred, y_batch)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test_norm).to(self.device)
                    pred = model(X_tensor).cpu().numpy().flatten()
                
                pred_denorm = pred * y_std + y_mean
                mse = mean_squared_error(y_test, pred_denorm)
                size_results[f'n_{n_samples}'] = float(mse)
            
            result = {
                'branches': branch_name,
                'sample_mse': size_results,
            }
            task_results['experiments'].append(result)
            
            print(f"  {branch_name:20s}", end='')
            for n in sample_sizes:
                print(f"  n={n}: {size_results[f'n_{n}']:.4f}", end='')
            print()
        
        return task_results


# =============================================================================
# MAIN RESEARCH PIPELINE
# =============================================================================

def run_full_exploration():
    """Run the complete research exploration."""
    
    print("\n" + "="*80)
    print("HYBRIDKAN RESEARCH EXPLORATION")
    print("Finding where multi-basis activation functions provide value")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Branch configurations to test
    branches_list = [
        ['relu'],                                    # Baseline
        'all',                                       # All branches
        ['fourier', 'relu'],                         # Fourier + ReLU
        ['legendre', 'chebyshev', 'relu'],          # Polynomials + ReLU
        ['gabor', 'fourier'],                        # Oscillatory
        ['legendre', 'chebyshev', 'hermite'],       # Pure polynomials
        ['fourier'],                                 # Pure Fourier
        ['gabor', 'legendre', 'fourier', 'relu'],   # Best mix guess
    ]
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'experiments': {},
    }
    
    # =========================================================================
    # EXPERIMENT 1: SYNTHETIC FUNCTION FITTING
    # =========================================================================
    print("\n" + "#"*80)
    print("# EXPERIMENT 1: SYNTHETIC FUNCTION FITTING")
    print("# Testing on functions that NEED specific bases")
    print("#"*80)
    
    reg_exp = RegressionExperiment(device)
    synthetic_results = []
    
    synthetic_tasks = [
        SyntheticDataset.polynomial_function,
        SyntheticDataset.sinusoidal_function,
        SyntheticDataset.gaussian_mixture,
        SyntheticDataset.mixed_function,
        SyntheticDataset.high_frequency,
        SyntheticDataset.step_function,
    ]
    
    for task_fn in synthetic_tasks:
        X, y, name = task_fn(n_samples=5000)
        
        # Split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"\n--- {name} ---")
        result = reg_exp.run(X_train, y_train, X_test, y_test, name, branches_list, epochs=300)
        synthetic_results.append(result)
    
    all_results['experiments']['synthetic_regression'] = synthetic_results
    
    # =========================================================================
    # EXPERIMENT 2: MULTIVARIATE REGRESSION
    # =========================================================================
    print("\n" + "#"*80)
    print("# EXPERIMENT 2: MULTIVARIATE COMPLEX FUNCTION")
    print("#"*80)
    
    X, y, name = SyntheticDataset.multivariate_complex(n_samples=6000, n_features=5)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\n--- {name} ---")
    multivar_result = reg_exp.run(
        X_train, y_train, X_test, y_test, name, branches_list,
        hidden_dims=[128, 64, 32], epochs=300
    )
    all_results['experiments']['multivariate'] = multivar_result
    
    # =========================================================================
    # EXPERIMENT 3: TABULAR CLASSIFICATION (WHERE MLPs MATTER)
    # =========================================================================
    print("\n" + "#"*80)
    print("# EXPERIMENT 3: TABULAR CLASSIFICATION")
    print("# Real-world datasets where features need complex transformations")
    print("#"*80)
    
    if SKLEARN_AVAILABLE:
        clf_exp = ClassificationExperiment(device)
        tabular_results = []
        
        # Wine dataset
        print("\n--- Wine Classification ---")
        wine = load_wine()
        X, y = wine.data.astype(np.float32), wine.target
        indices = np.random.permutation(len(X))
        split = int(0.8 * len(X))
        X_train, X_test = X[indices[:split]], X[indices[split:]]
        y_train, y_test = y[indices[:split]], y[indices[split:]]
        result = clf_exp.run(X_train, y_train, X_test, y_test, "Wine", branches_list, num_classes=3, epochs=150)
        tabular_results.append(result)
        
        # Iris dataset
        print("\n--- Iris Classification ---")
        iris = load_iris()
        X, y = iris.data.astype(np.float32), iris.target
        indices = np.random.permutation(len(X))
        split = int(0.8 * len(X))
        X_train, X_test = X[indices[:split]], X[indices[split:]]
        y_train, y_test = y[indices[:split]], y[indices[split:]]
        result = clf_exp.run(X_train, y_train, X_test, y_test, "Iris", branches_list, num_classes=3, epochs=150)
        tabular_results.append(result)
        
        # Breast Cancer
        print("\n--- Breast Cancer Classification ---")
        bc = load_breast_cancer()
        X, y = bc.data.astype(np.float32), bc.target
        indices = np.random.permutation(len(X))
        split = int(0.8 * len(X))
        X_train, X_test = X[indices[:split]], X[indices[split:]]
        y_train, y_test = y[indices[:split]], y[indices[split:]]
        result = clf_exp.run(X_train, y_train, X_test, y_test, "Breast Cancer", branches_list, num_classes=2, epochs=150)
        tabular_results.append(result)
        
        all_results['experiments']['tabular_classification'] = tabular_results
    
    # =========================================================================
    # EXPERIMENT 4: ROBUSTNESS TO NOISE
    # =========================================================================
    print("\n" + "#"*80)
    print("# EXPERIMENT 4: ROBUSTNESS TO INPUT NOISE")
    print("# Train clean, test noisy - which branches degrade gracefully?")
    print("#"*80)
    
    robust_exp = RobustnessExperiment(device)
    
    # Test on sinusoidal (where Fourier should shine)
    X, y, name = SyntheticDataset.sinusoidal_function(n_samples=5000)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\n--- Robustness: {name} ---")
    robustness_result = robust_exp.run(X_train, y_train, X_test, y_test, name, branches_list)
    all_results['experiments']['robustness'] = robustness_result
    
    # =========================================================================
    # EXPERIMENT 5: FEW-SHOT LEARNING
    # =========================================================================
    print("\n" + "#"*80)
    print("# EXPERIMENT 5: FEW-SHOT GENERALIZATION")
    print("# Can HybridKAN learn with less data?")
    print("#"*80)
    
    fewshot_exp = FewShotExperiment(device)
    
    # Mixed function - needs multiple bases
    X, y, name = SyntheticDataset.mixed_function(n_samples=6000)
    print(f"\n--- Few-Shot: {name} ---")
    fewshot_result = fewshot_exp.run(X, y, name, branches_list[:5])  # Subset for speed
    all_results['experiments']['fewshot'] = fewshot_result
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_dir = Path('results_research')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'exploration_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # =========================================================================
    # SUMMARY ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("RESEARCH EXPLORATION SUMMARY")
    print("="*80)
    
    print("\n📊 KEY FINDINGS:\n")
    
    # Find where HybridKAN beats ReLU
    wins = []
    losses = []
    
    for exp_name, exp_results in all_results['experiments'].items():
        if isinstance(exp_results, dict):
            exp_results = [exp_results]
        
        for task_result in exp_results:
            if 'experiments' not in task_result:
                continue
            
            task_name = task_result['task']
            relu_score = None
            best_hybrid = None
            best_hybrid_name = None
            
            for exp in task_result['experiments']:
                branches = exp['branches']
                
                # Get score (lower is better for MSE, higher for accuracy)
                if 'best_mse' in exp:
                    score = exp['best_mse']
                    is_lower_better = True
                elif 'best_accuracy' in exp:
                    score = exp['best_accuracy']
                    is_lower_better = False
                else:
                    continue
                
                if branches == 'relu':
                    relu_score = score
                elif branches != 'relu':
                    if best_hybrid is None:
                        best_hybrid = score
                        best_hybrid_name = branches
                    elif is_lower_better and score < best_hybrid:
                        best_hybrid = score
                        best_hybrid_name = branches
                    elif not is_lower_better and score > best_hybrid:
                        best_hybrid = score
                        best_hybrid_name = branches
            
            if relu_score is not None and best_hybrid is not None:
                if is_lower_better:
                    hybrid_wins = best_hybrid < relu_score
                    improvement = (relu_score - best_hybrid) / relu_score * 100
                else:
                    hybrid_wins = best_hybrid > relu_score
                    improvement = best_hybrid - relu_score
                
                if hybrid_wins:
                    wins.append((task_name, best_hybrid_name, improvement))
                else:
                    losses.append((task_name, best_hybrid_name, -improvement))
    
    print("✅ TASKS WHERE HYBRIDKAN BEATS RELU:")
    if wins:
        for task, branches, imp in sorted(wins, key=lambda x: -x[2]):
            print(f"   • {task}: {branches} ({imp:+.2f}% better)")
    else:
        print("   None found in this run")
    
    print("\n❌ TASKS WHERE RELU WINS:")
    if losses:
        for task, branches, imp in sorted(losses, key=lambda x: x[2]):
            print(f"   • {task}: ReLU wins by {-imp:.2f}%")
    else:
        print("   None!")
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_dir / 'exploration_results.json'}")
    print("="*80)
    
    return all_results


if __name__ == '__main__':
    run_full_exploration()
