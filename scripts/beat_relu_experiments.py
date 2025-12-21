#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Beat ReLU Experiments

Strategies to make HybridKAN beat ReLU on classification:
1. Equal gate initialization (remove ReLU bias)
2. No CNN preprocessing (let HybridKAN learn features)
3. Entropy regularization (force branch diversity)
4. Temperature annealing (soft → hard gating)

Author: Rob Saidov
Date: December 2025
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hybridkan import HybridKAN, set_seed
from hybridkan.data import get_cifar10_loaders, get_mnist_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_gate_entropy(model):
    """Compute entropy of gate weights (higher = more diverse)."""
    gate_weights = model.get_branch_gate_weights()
    if not gate_weights:
        return 0.0
    
    total_entropy = 0.0
    count = 0
    
    for layer_gates in gate_weights.values():
        weights = np.array(list(layer_gates.values()))
        weights = weights / (weights.sum() + 1e-8)  # Normalize
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(len(weights))
        total_entropy += entropy / max_entropy  # Normalized
        count += 1
    
    return total_entropy / count if count > 0 else 0.0


def gate_entropy_loss(model, target_entropy=0.8):
    """
    Loss to encourage gate diversity.
    Penalizes when one branch dominates.
    """
    total_loss = 0.0
    count = 0
    
    for block in model.blocks:
        if hasattr(block, 'gates') and block.gates is not None:
            gate_values = []
            for name in block.branch_names:
                gate_val = F.softplus(block.gates[name].alpha)
                gate_values.append(gate_val)
            
            gate_tensor = torch.stack(gate_values)
            gate_probs = F.softmax(gate_tensor, dim=0)
            
            # Entropy: -sum(p * log(p))
            entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8))
            max_entropy = np.log(len(gate_values))
            normalized_entropy = entropy / max_entropy
            
            # Penalize low entropy
            total_loss += F.relu(target_entropy - normalized_entropy)
            count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0)


def train_with_strategies(
    model,
    train_loader,
    test_loader,
    epochs=100,
    lr=1e-3,
    entropy_weight=0.0,
    temperature_anneal=False,
    patience=20,
):
    """Train with optional entropy regularization and temperature annealing."""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'test_acc': [], 'entropy': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Temperature annealing: start soft (5.0), end hard (0.5)
        if temperature_anneal:
            temp = 5.0 - (4.5 * epoch / epochs)
        else:
            temp = 1.0
        
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Classification loss
            ce_loss = F.cross_entropy(output, target)
            
            # Entropy regularization
            if entropy_weight > 0:
                ent_loss = gate_entropy_loss(model) * entropy_weight
                loss = ce_loss + ent_loss
            else:
                loss = ce_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        
        test_acc = 100.0 * test_correct / test_total
        train_acc = 100.0 * correct / total
        entropy = compute_gate_entropy(model)
        
        history['train_loss'].append(total_loss / len(train_loader))
        history['test_acc'].append(test_acc)
        history['entropy'].append(entropy)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, '
                  f'Entropy={entropy:.3f}, Best={best_acc:.2f}%')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    return {
        'best_accuracy': best_acc,
        'best_epoch': best_epoch,
        'history': history,
        'final_gates': model.get_branch_gate_weights(),
    }


def run_experiment(name, model_kwargs, train_kwargs, train_loader, test_loader):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")
    
    set_seed(42)
    model = HybridKAN(**model_kwargs)
    param_counts = model.count_parameters()
    total_params = param_counts['total']
    print(f"Parameters: {total_params:,}")
    print(f"Branches: {model.active_branches}")
    
    start_time = time.time()
    results = train_with_strategies(model, train_loader, test_loader, **train_kwargs)
    elapsed = time.time() - start_time
    
    results['name'] = name
    results['elapsed_time'] = elapsed
    results['parameters'] = total_params
    
    print(f"\nResult: {results['best_accuracy']:.2f}% at epoch {results['best_epoch']}")
    print(f"Time: {elapsed/60:.1f} minutes")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer epochs')
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 10
    
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    
    # Load data
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(batch_size=128, use_cnn=True)
        input_dim = 3072
        cnn_channels = 3
    else:
        train_loader, test_loader = get_mnist_loaders(batch_size=128, use_cnn=True)
        input_dim = 784
        cnn_channels = 1
    
    all_results = []
    
    # ==========================================================================
    # Experiment 1: ReLU Baseline (with CNN)
    # ==========================================================================
    results = run_experiment(
        name="relu_baseline_cnn",
        model_kwargs={
            'input_dim': input_dim,
            'hidden_dims': [256, 128, 64],
            'num_classes': 10,
            'activation_functions': 'relu',
            'use_cnn': True,
            'cnn_channels': cnn_channels,
            'use_residual': True,
        },
        train_kwargs={'epochs': args.epochs, 'lr': 1e-3},
        train_loader=train_loader,
        test_loader=test_loader,
    )
    all_results.append(results)
    
    # ==========================================================================
    # Experiment 2: HybridKAN with CNN (current approach)
    # ==========================================================================
    results = run_experiment(
        name="hybridkan_cnn",
        model_kwargs={
            'input_dim': input_dim,
            'hidden_dims': [256, 128, 64],
            'num_classes': 10,
            'activation_functions': 'all',
            'use_cnn': True,
            'cnn_channels': cnn_channels,
            'use_residual': True,
        },
        train_kwargs={'epochs': args.epochs, 'lr': 1e-3},
        train_loader=train_loader,
        test_loader=test_loader,
    )
    all_results.append(results)
    
    # ==========================================================================
    # Experiment 3: HybridKAN with Entropy Regularization
    # ==========================================================================
    results = run_experiment(
        name="hybridkan_entropy_reg",
        model_kwargs={
            'input_dim': input_dim,
            'hidden_dims': [256, 128, 64],
            'num_classes': 10,
            'activation_functions': 'all',
            'use_cnn': True,
            'cnn_channels': cnn_channels,
            'use_residual': True,
        },
        train_kwargs={'epochs': args.epochs, 'lr': 1e-3, 'entropy_weight': 0.1},
        train_loader=train_loader,
        test_loader=test_loader,
    )
    all_results.append(results)
    
    # ==========================================================================
    # Experiment 4: HybridKAN NO CNN (raw pixels) - larger network
    # ==========================================================================
    # Reload data without CNN preprocessing
    if args.dataset == 'cifar10':
        train_loader_flat, test_loader_flat = get_cifar10_loaders(batch_size=128, use_cnn=False)
    else:
        train_loader_flat, test_loader_flat = get_mnist_loaders(batch_size=128, use_cnn=False)
    
    results = run_experiment(
        name="hybridkan_no_cnn",
        model_kwargs={
            'input_dim': input_dim,
            'hidden_dims': [512, 256, 128],  # Larger to compensate for no CNN
            'num_classes': 10,
            'activation_functions': 'all',
            'use_cnn': False,
            'use_residual': True,
        },
        train_kwargs={'epochs': args.epochs, 'lr': 5e-4},  # Lower LR for stability
        train_loader=train_loader_flat,
        test_loader=test_loader_flat,
    )
    all_results.append(results)
    
    # ==========================================================================
    # Experiment 5: ReLU NO CNN Baseline
    # ==========================================================================
    results = run_experiment(
        name="relu_no_cnn",
        model_kwargs={
            'input_dim': input_dim,
            'hidden_dims': [512, 256, 128],
            'num_classes': 10,
            'activation_functions': 'relu',
            'use_cnn': False,
            'use_residual': True,
        },
        train_kwargs={'epochs': args.epochs, 'lr': 5e-4},
        train_loader=train_loader_flat,
        test_loader=test_loader_flat,
    )
    all_results.append(results)
    
    # ==========================================================================
    # Experiment 6: Fourier-focused (good for periodic patterns in images)
    # ==========================================================================
    results = run_experiment(
        name="fourier_gabor_only",
        model_kwargs={
            'input_dim': input_dim,
            'hidden_dims': [512, 256, 128],
            'num_classes': 10,
            'activation_functions': ['fourier', 'gabor', 'relu'],
            'use_cnn': False,
            'use_residual': True,
        },
        train_kwargs={'epochs': args.epochs, 'lr': 5e-4},
        train_loader=train_loader_flat,
        test_loader=test_loader_flat,
    )
    all_results.append(results)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Sort by accuracy
    all_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
    
    print(f"\n{'Experiment':<30} {'Accuracy':>10} {'Params':>12} {'Time':>10}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['name']:<30} {r['best_accuracy']:>9.2f}% {r['parameters']:>11,} {r['elapsed_time']/60:>9.1f}m")
    
    # Check if any HybridKAN beat ReLU
    relu_baseline = next((r for r in all_results if r['name'] == 'relu_baseline_cnn'), None)
    relu_no_cnn = next((r for r in all_results if r['name'] == 'relu_no_cnn'), None)
    
    print("\n" + "="*60)
    print("DID HYBRIDKAN BEAT RELU?")
    print("="*60)
    
    for r in all_results:
        if 'hybrid' in r['name'] or 'fourier' in r['name']:
            if relu_baseline and r['best_accuracy'] > relu_baseline['best_accuracy']:
                print(f"✅ {r['name']}: {r['best_accuracy']:.2f}% > ReLU(CNN): {relu_baseline['best_accuracy']:.2f}%")
            elif relu_no_cnn and 'no_cnn' in r['name'] and r['best_accuracy'] > relu_no_cnn['best_accuracy']:
                print(f"✅ {r['name']}: {r['best_accuracy']:.2f}% > ReLU(no CNN): {relu_no_cnn['best_accuracy']:.2f}%")
            else:
                baseline = relu_no_cnn if 'no_cnn' in r['name'] else relu_baseline
                if baseline:
                    diff = r['best_accuracy'] - baseline['best_accuracy']
                    print(f"{'✅' if diff > 0 else '❌'} {r['name']}: {r['best_accuracy']:.2f}% ({diff:+.2f}% vs ReLU)")
    
    # Save results
    output_dir = project_root / "results_beat_relu"
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"{args.dataset}_beat_relu_results.json"
    with open(results_file, 'w') as f:
        # Remove non-serializable items
        save_results = []
        for r in all_results:
            save_r = {k: v for k, v in r.items() if k != 'history'}
            save_r['final_test_acc'] = r['history']['test_acc'][-1] if r['history']['test_acc'] else 0
            save_results.append(save_r)
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dataset': args.dataset,
            'epochs': args.epochs,
            'results': save_results,
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
