"""
Run HybridKAN experiments on additional datasets:
- Classification: Wine, Breast Cancer, Iris
- Regression: California Housing, Diabetes

Generates CSV results for each dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json

from hybridkan import HybridKAN, set_seed

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS_CLASSIFICATION = 100
EPOCHS_REGRESSION = 150
BATCH_SIZE = 32

# Activation configurations to test
ACTIVATION_CONFIGS = [
    {'name': 'relu', 'activations': ['relu']},
    {'name': 'all', 'activations': ['gabor', 'legendre', 'chebyshev', 'hermite', 'fourier', 'relu']},
    {'name': 'fourier', 'activations': ['fourier']},
    {'name': 'gabor', 'activations': ['gabor']},
    {'name': 'legendre', 'activations': ['legendre']},
    {'name': 'chebyshev', 'activations': ['chebyshev']},
    {'name': 'fourier_gabor', 'activations': ['fourier', 'gabor']},
    {'name': 'all_except_relu', 'activations': ['gabor', 'legendre', 'chebyshev', 'hermite', 'fourier']},
]


def load_classification_data(name):
    """Load sklearn classification datasets."""
    if name == 'wine':
        data = load_wine()
    elif name == 'iris':
        data = load_iris()
    elif name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    X, y = data.data.astype(np.float32), data.target
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    return {
        'X_train': torch.FloatTensor(X_train),
        'X_test': torch.FloatTensor(X_test),
        'y_train': torch.LongTensor(y_train),
        'y_test': torch.LongTensor(y_test),
        'input_dim': X.shape[1],
        'num_classes': len(np.unique(y)),
        'name': name
    }


def load_regression_data(name):
    """Load sklearn regression datasets."""
    if name == 'california_housing':
        data = fetch_california_housing()
    elif name == 'diabetes':
        data = load_diabetes()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)
    
    # Normalize features
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    # Normalize targets
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    
    return {
        'X_train': torch.FloatTensor(X_train),
        'X_test': torch.FloatTensor(X_test),
        'y_train': torch.FloatTensor(y_train),
        'y_test': torch.FloatTensor(y_test),
        'input_dim': X.shape[1],
        'name': name,
        'scaler_y': scaler_y
    }


def train_classification(data, activations, epochs=100):
    """Train classification model."""
    set_seed(SEED)
    
    model = HybridKAN(
        input_dim=data['input_dim'],
        hidden_dims=[64, 32],
        num_classes=data['num_classes'],
        activation_functions=activations,
        use_cnn=False,
        use_residual=True,
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create data loaders
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    test_dataset = TensorDataset(data['X_test'], data['y_test'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    best_acc = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
    
    return {
        'best_accuracy': best_acc,
        'best_epoch': best_epoch,
        'params': model.count_parameters()['total']
    }


def train_regression(data, activations, epochs=150):
    """Train regression model."""
    set_seed(SEED)
    
    model = HybridKAN(
        input_dim=data['input_dim'],
        hidden_dims=[64, 32],
        num_classes=1,
        activation_functions=activations,
        use_cnn=False,
        use_residual=True,
        regression=True,
        use_batch_norm=False,  # Disable BatchNorm to avoid issues with small batches
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Create data loaders
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    test_dataset = TensorDataset(data['X_test'], data['y_test'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    best_mse = float('inf')
    best_mae = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch).squeeze()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mse = np.mean((all_preds - all_targets) ** 2)
        mae = np.mean(np.abs(all_preds - all_targets))
        
        if mse < best_mse:
            best_mse = mse
            best_mae = mae
            best_epoch = epoch + 1
    
    return {
        'best_mse': best_mse,
        'best_mae': best_mae,
        'best_epoch': best_epoch,
        'params': model.count_parameters()['total']
    }


def run_all_experiments():
    """Run all experiments and save results."""
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    results = {
        'classification': {},
        'regression': {}
    }
    
    # Classification datasets
    classification_datasets = ['wine', 'iris', 'breast_cancer']
    
    for dataset_name in classification_datasets:
        print(f"\n{'='*60}")
        print(f"Classification: {dataset_name.upper()}")
        print("=" * 60)
        
        data = load_classification_data(dataset_name)
        print(f"Samples: {len(data['X_train'])} train, {len(data['X_test'])} test")
        print(f"Features: {data['input_dim']}, Classes: {data['num_classes']}")
        
        results['classification'][dataset_name] = {}
        
        for config in tqdm(ACTIVATION_CONFIGS, desc=f"{dataset_name}"):
            result = train_classification(data, config['activations'], EPOCHS_CLASSIFICATION)
            results['classification'][dataset_name][config['name']] = result
            print(f"  {config['name']}: {result['best_accuracy']:.2f}%")
    
    # Regression datasets
    regression_datasets = ['california_housing', 'diabetes']
    
    for dataset_name in regression_datasets:
        print(f"\n{'='*60}")
        print(f"Regression: {dataset_name.upper()}")
        print("=" * 60)
        
        data = load_regression_data(dataset_name)
        print(f"Samples: {len(data['X_train'])} train, {len(data['X_test'])} test")
        print(f"Features: {data['input_dim']}")
        
        results['regression'][dataset_name] = {}
        
        for config in tqdm(ACTIVATION_CONFIGS, desc=f"{dataset_name}"):
            result = train_regression(data, config['activations'], EPOCHS_REGRESSION)
            results['regression'][dataset_name][config['name']] = result
            print(f"  {config['name']}: MSE={result['best_mse']:.4f}, MAE={result['best_mae']:.4f}")
    
    return results


def save_results(results):
    """Save results to CSV and JSON files."""
    output_dir = 'results_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save classification results
    clf_rows = []
    for dataset, activations in results['classification'].items():
        for act_name, metrics in activations.items():
            clf_rows.append({
                'dataset': dataset,
                'activation': act_name,
                'test_accuracy': metrics['best_accuracy'],
                'best_epoch': metrics['best_epoch'],
                'parameters': metrics['params']
            })
    
    clf_df = pd.DataFrame(clf_rows)
    clf_df.to_csv(f'{output_dir}/additional_classification_results.csv', index=False)
    print(f"\nSaved: {output_dir}/additional_classification_results.csv")
    
    # Save regression results
    reg_rows = []
    for dataset, activations in results['regression'].items():
        for act_name, metrics in activations.items():
            reg_rows.append({
                'dataset': dataset,
                'activation': act_name,
                'test_mse': metrics['best_mse'],
                'test_mae': metrics['best_mae'],
                'best_epoch': metrics['best_epoch'],
                'parameters': metrics['params']
            })
    
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(f'{output_dir}/additional_regression_results.csv', index=False)
    print(f"Saved: {output_dir}/additional_regression_results.csv")
    
    # Save combined JSON
    with open(f'{output_dir}/additional_datasets_full_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir}/additional_datasets_full_results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - BEST ACTIVATION PER DATASET")
    print("=" * 60)
    
    print("\nClassification (Best Accuracy):")
    for dataset in results['classification']:
        best = max(results['classification'][dataset].items(), key=lambda x: x[1]['best_accuracy'])
        relu_acc = results['classification'][dataset]['relu']['best_accuracy']
        print(f"  {dataset}: {best[0]} ({best[1]['best_accuracy']:.2f}%) vs ReLU ({relu_acc:.2f}%)")
    
    print("\nRegression (Lowest MAE):")
    for dataset in results['regression']:
        best = min(results['regression'][dataset].items(), key=lambda x: x[1]['best_mae'])
        relu_mae = results['regression'][dataset]['relu']['best_mae']
        print(f"  {dataset}: {best[0]} (MAE={best[1]['best_mae']:.4f}) vs ReLU (MAE={relu_mae:.4f})")
    
    # Create best activation summary CSV
    summary_rows = []
    
    for dataset, activations in results['classification'].items():
        best = max(activations.items(), key=lambda x: x[1]['best_accuracy'])
        relu_result = activations['relu']
        summary_rows.append({
            'dataset': dataset,
            'task': 'classification',
            'best_activation': best[0],
            'best_value': best[1]['best_accuracy'],
            'relu_value': relu_result['best_accuracy'],
            'hybridkan_wins': best[0] != 'relu',
            'improvement': best[1]['best_accuracy'] - relu_result['best_accuracy']
        })
    
    for dataset, activations in results['regression'].items():
        best = min(activations.items(), key=lambda x: x[1]['best_mae'])
        relu_result = activations['relu']
        summary_rows.append({
            'dataset': dataset,
            'task': 'regression',
            'best_activation': best[0],
            'best_value': best[1]['best_mae'],
            'relu_value': relu_result['best_mae'],
            'hybridkan_wins': best[0] != 'relu',
            'improvement': relu_result['best_mae'] - best[1]['best_mae']  # Positive = HybridKAN better
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{output_dir}/additional_best_per_dataset.csv', index=False)
    print(f"\nSaved: {output_dir}/additional_best_per_dataset.csv")


if __name__ == '__main__':
    print("Running HybridKAN on Additional Datasets")
    print("=" * 60)
    
    results = run_all_experiments()
    save_results(results)
    
    print("\n" + "=" * 60)
    print("DONE! All results saved to results_comparison/")
    print("=" * 60)
