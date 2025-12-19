#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HybridKAN Experiment Runner V2 - Enhanced with Better Metrics & Novel Training Strategies

New Features:
1. Comprehensive Metrics:
   - Per-class precision, recall, F1
   - Macro/Micro/Weighted averages
   - Confusion matrix
   - Gate entropy (branch diversity)
   - Learning rate tracking
   - Training/validation curves
   - Inference time
   - Parameter efficiency

2. Novel Training Strategies to Beat ReLU:
   - Gate entropy regularization (encourages using multiple branches)
   - Temperature annealing (soft → hard gating)
   - Warmup with equal gates (not ReLU-biased)
   - Separate learning rates for gates
   - Branch dropout (force diversity)

Usage:
    python scripts/run_experiments_v2.py --dataset cifar10 --experiment main
    python scripts/run_experiments_v2.py --dataset cifar10 --strategy entropy_reg
    python scripts/run_experiments_v2.py --dataset cifar10 --strategy all_strategies
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hybridkan import HybridKAN, set_seed
from hybridkan.data import (
    get_mnist_loaders,
    get_cifar10_loaders,
    DATASET_INFO,
)

try:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
        accuracy_score,
        top_k_accuracy_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# COMPREHENSIVE METRICS
# =============================================================================

class ComprehensiveMetrics:
    """Collects and computes comprehensive evaluation metrics."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, probs: Optional[torch.Tensor] = None):
        """Update with batch predictions."""
        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())
    
    def compute(self) -> Dict:
        """Compute all metrics."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(targets, preds) * 100
        
        if SKLEARN_AVAILABLE:
            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, preds, average=None, zero_division=0
            )
            
            # Store per-class
            metrics['per_class'] = {}
            for i, name in enumerate(self.class_names):
                metrics['per_class'][name] = {
                    'precision': precision[i] * 100,
                    'recall': recall[i] * 100,
                    'f1': f1[i] * 100,
                    'support': int(support[i]),
                }
            
            # Macro averages (equal weight to each class)
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                targets, preds, average='macro', zero_division=0
            )
            metrics['macro_precision'] = p_macro * 100
            metrics['macro_recall'] = r_macro * 100
            metrics['macro_f1'] = f1_macro * 100
            
            # Micro averages (global counts)
            p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
                targets, preds, average='micro', zero_division=0
            )
            metrics['micro_precision'] = p_micro * 100
            metrics['micro_recall'] = r_micro * 100
            metrics['micro_f1'] = f1_micro * 100
            
            # Weighted averages (weighted by support)
            p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
                targets, preds, average='weighted', zero_division=0
            )
            metrics['weighted_precision'] = p_weighted * 100
            metrics['weighted_recall'] = r_weighted * 100
            metrics['weighted_f1'] = f1_weighted * 100
            
            # Confusion matrix
            cm = confusion_matrix(targets, preds)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Top-k accuracy if we have probabilities
            if len(self.all_probs) > 0:
                probs = np.array(self.all_probs)
                if self.num_classes >= 5:
                    metrics['top5_accuracy'] = top_k_accuracy_score(targets, probs, k=5) * 100
                if self.num_classes >= 3:
                    metrics['top3_accuracy'] = top_k_accuracy_score(targets, probs, k=3) * 100
        
        return metrics


def compute_gate_entropy(model: nn.Module) -> Dict:
    """
    Compute gate entropy to measure branch diversity.
    
    High entropy = using multiple branches equally (good!)
    Low entropy = dominated by one branch (bad, not using full capacity)
    """
    gate_weights = model.get_branch_gate_weights()
    
    if not gate_weights:
        return {}
    
    entropies = {}
    for layer_name, branches in gate_weights.items():
        weights = np.array(list(branches.values()))
        
        # Normalize to probability distribution
        weights_normalized = weights / (weights.sum() + 1e-8)
        
        # Compute entropy
        entropy = -np.sum(weights_normalized * np.log(weights_normalized + 1e-8))
        max_entropy = np.log(len(weights))  # Maximum entropy (uniform)
        
        entropies[layer_name] = {
            'entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'normalized_entropy': float(entropy / max_entropy) if max_entropy > 0 else 0,
            'branch_weights': {k: float(v) for k, v in branches.items()},
        }
    
    # Average across layers
    avg_entropy = np.mean([e['normalized_entropy'] for e in entropies.values()])
    entropies['average_normalized_entropy'] = float(avg_entropy)
    
    return entropies


def measure_inference_time(model: nn.Module, input_shape: Tuple, device: torch.device, num_runs: int = 100) -> Dict:
    """Measure inference time statistics."""
    model.eval()
    
    # Warmup
    dummy_input = torch.randn(1, *input_shape).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
    }


# =============================================================================
# NOVEL TRAINING STRATEGIES
# =============================================================================

class GateEntropyRegularizer:
    """
    Regularizer that encourages high gate entropy (using multiple branches).
    
    This penalizes the model for relying too heavily on a single branch,
    forcing it to explore and utilize the different basis functions.
    """
    
    def __init__(self, weight: float = 0.01, target_entropy: float = 0.8):
        self.weight = weight
        self.target_entropy = target_entropy  # Target normalized entropy
    
    def __call__(self, model: nn.Module) -> torch.Tensor:
        """Compute entropy regularization loss."""
        total_reg = torch.tensor(0.0, device=next(model.parameters()).device)
        
        # Get gate parameters
        for name, module in model.named_modules():
            if hasattr(module, 'gates') and module.gates is not None:
                gate_values = []
                for gate in module.gates.values():
                    gate_values.append(F.softplus(gate.alpha))
                
                if gate_values:
                    weights = torch.stack(gate_values)
                    weights_normalized = weights / (weights.sum() + 1e-8)
                    
                    # Compute negative entropy (we want to maximize entropy, so minimize negative)
                    entropy = -torch.sum(weights_normalized * torch.log(weights_normalized + 1e-8))
                    max_entropy = torch.log(torch.tensor(float(len(weights))))
                    normalized_entropy = entropy / max_entropy
                    
                    # Penalize if entropy is below target
                    reg = F.relu(self.target_entropy - normalized_entropy)
                    total_reg = total_reg + reg
        
        return self.weight * total_reg


class TemperatureAnnealer:
    """
    Anneals softmax temperature for gating from high (soft) to low (hard).
    
    High temperature → soft decisions → explore different branches
    Low temperature → hard decisions → commit to best branches
    """
    
    def __init__(
        self,
        initial_temp: float = 5.0,
        final_temp: float = 1.0,
        warmup_epochs: int = 20,
        anneal_epochs: int = 50,
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs
    
    def get_temperature(self, epoch: int) -> float:
        """Get temperature for current epoch."""
        if epoch < self.warmup_epochs:
            return self.initial_temp
        
        progress = min(1.0, (epoch - self.warmup_epochs) / self.anneal_epochs)
        return self.initial_temp - progress * (self.initial_temp - self.final_temp)


class BranchDropout(nn.Module):
    """
    Randomly drops branches during training to force diversity.
    
    Similar to dropout but at the branch level - forces the model
    to not rely on any single branch.
    """
    
    def __init__(self, drop_prob: float = 0.1, min_branches: int = 2):
        super().__init__()
        self.drop_prob = drop_prob
        self.min_branches = min_branches
    
    def forward(self, branch_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if not self.training or len(branch_outputs) <= self.min_branches:
            return branch_outputs
        
        # Decide which branches to keep
        num_branches = len(branch_outputs)
        keep_mask = torch.rand(num_branches) > self.drop_prob
        
        # Ensure minimum branches
        if keep_mask.sum() < self.min_branches:
            indices = torch.randperm(num_branches)[:self.min_branches]
            keep_mask[indices] = True
        
        # Apply mask and rescale
        scale = num_branches / keep_mask.sum().item()
        return [out * scale if keep else torch.zeros_like(out) 
                for out, keep in zip(branch_outputs, keep_mask)]


# =============================================================================
# ENHANCED TRAINER
# =============================================================================

class EnhancedTrainer:
    """
    Enhanced trainer with novel strategies and comprehensive metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        # Novel strategies
        use_entropy_reg: bool = False,
        entropy_weight: float = 0.01,
        use_temp_annealing: bool = False,
        use_separate_gate_lr: bool = False,
        gate_lr_scale: float = 10.0,
        use_warmup_equal_gates: bool = False,
        # Output
        output_dir: str = "results",
        experiment_name: str = "experiment",
        num_classes: int = 10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup strategies
        self.entropy_reg = GateEntropyRegularizer(weight=entropy_weight) if use_entropy_reg else None
        self.temp_annealer = TemperatureAnnealer() if use_temp_annealing else None
        
        # Setup optimizer with optional separate LR for gates
        if use_separate_gate_lr:
            gate_params = []
            other_params = []
            for name, param in model.named_parameters():
                if 'gate' in name.lower() or 'alpha' in name.lower():
                    gate_params.append(param)
                else:
                    other_params.append(param)
            
            self.optimizer = torch.optim.AdamW([
                {'params': other_params, 'lr': learning_rate},
                {'params': gate_params, 'lr': learning_rate * gate_lr_scale},
            ], weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        
        # Scheduler
        steps_per_epoch = len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
        
        # Initialize equal gates if requested
        if use_warmup_equal_gates:
            self._initialize_equal_gates()
        
        # Metrics tracking
        self.metrics_history = []
        self.best_accuracy = 0.0
        self.best_epoch = 0
        
        # AMP
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
    
    def _initialize_equal_gates(self):
        """Initialize all gates to equal values (not ReLU-biased)."""
        for module in self.model.modules():
            if hasattr(module, 'gates') and module.gates is not None:
                for gate in module.gates.values():
                    # Set all gates to same initial value
                    nn.init.constant_(gate.alpha, 0.5)
    
    def train(self, verbose: bool = True) -> Dict:
        """Run training with all enhancements."""
        from tqdm import tqdm
        
        start_time = time.time()
        
        progress_bar = tqdm(range(1, self.epochs + 1), desc="Training", disable=not verbose)
        
        for epoch in progress_bar:
            # Train epoch
            train_loss, train_acc = self._train_epoch(epoch)
            
            # Validate
            val_metrics = self._validate()
            
            # Track LR
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Gate entropy
            gate_entropy = compute_gate_entropy(self.model)
            avg_entropy = gate_entropy.get('average_normalized_entropy', 0)
            
            # Record metrics
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_metrics.get('loss', 0),
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics.get('macro_precision', 0),
                'val_recall': val_metrics.get('macro_recall', 0),
                'val_f1': val_metrics.get('macro_f1', 0),
                'learning_rate': current_lr,
                'gate_entropy': avg_entropy,
                'elapsed_time': time.time() - start_time,
            }
            self.metrics_history.append(epoch_metrics)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{train_loss:.4f}',
                'Train': f'{train_acc:.2f}%',
                'Val': f'{val_metrics["accuracy"]:.2f}%',
                'Entropy': f'{avg_entropy:.3f}',
            })
            
            # Best model
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self.best_epoch = epoch
                self._save_checkpoint('best')
        
        # Final comprehensive evaluation
        final_metrics = self._comprehensive_evaluate()
        
        # Inference time
        input_shape = next(iter(self.val_loader))[0][0].shape
        inference_time = measure_inference_time(self.model, input_shape, self.device)
        
        # Summary
        summary = {
            'experiment_name': self.experiment_name,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'total_epochs': self.epochs,
            'total_time': time.time() - start_time,
            'final_metrics': final_metrics,
            'inference_time': inference_time,
            'gate_entropy': compute_gate_entropy(self.model),
            'metrics_history': self.metrics_history,
            'parameter_count': sum(p.numel() for p in self.model.parameters()),
        }
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = F.nll_loss(output, target)
                    
                    # Add entropy regularization
                    if self.entropy_reg is not None:
                        loss = loss + self.entropy_reg(self.model)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = F.nll_loss(output, target)
                
                if self.entropy_reg is not None:
                    loss = loss + self.entropy_reg(self.model)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        return total_loss / len(self.train_loader), 100 * correct / total
    
    @torch.no_grad()
    def _validate(self) -> Dict:
        """Quick validation."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = F.nll_loss(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100 * correct / total,
        }
    
    @torch.no_grad()
    def _comprehensive_evaluate(self) -> Dict:
        """Comprehensive evaluation with all metrics."""
        self.model.eval()
        
        class_names = [f'class_{i}' for i in range(self.num_classes)]
        metrics_computer = ComprehensiveMetrics(self.num_classes, class_names)
        
        total_loss = 0.0
        
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = F.nll_loss(output, target)
            
            total_loss += loss.item()
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            metrics_computer.update(preds, target, probs)
        
        metrics = metrics_computer.compute()
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def _save_checkpoint(self, tag: str):
        """Save checkpoint."""
        path = self.output_dir / f'{self.experiment_name}_{tag}.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'epoch': self.best_epoch,
        }, path)
    
    def _save_results(self, summary: Dict):
        """Save results to JSON."""
        path = self.output_dir / f'{self.experiment_name}_results.json'
        
        # Convert numpy arrays to lists
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(convert(summary), f, indent=2)


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

DATASET_CONFIGS = {
    'mnist': {
        'loader_fn': get_mnist_loaders,
        'input_dim': 784,
        'num_classes': 10,
        'hidden_dims': [256, 128, 64],
        'use_cnn': True,
        'cnn_channels': 1,
        'cnn_output_dim': 256,
        'batch_size': 128,
        'epochs': 50,
        'quick_epochs': 10,
    },
    'cifar10': {
        'loader_fn': get_cifar10_loaders,
        'input_dim': 3072,
        'num_classes': 10,
        'hidden_dims': [256, 128, 64],
        'use_cnn': True,
        'cnn_channels': 3,
        'cnn_output_dim': 256,
        'batch_size': 64,
        'epochs': 100,
        'quick_epochs': 15,
    },
}

STRATEGY_CONFIGS = {
    'baseline': {
        'name': 'Baseline (no enhancements)',
        'use_entropy_reg': False,
        'use_temp_annealing': False,
        'use_separate_gate_lr': False,
        'use_warmup_equal_gates': False,
    },
    'entropy_reg': {
        'name': 'Gate Entropy Regularization',
        'use_entropy_reg': True,
        'entropy_weight': 0.01,
        'use_temp_annealing': False,
        'use_separate_gate_lr': False,
        'use_warmup_equal_gates': False,
    },
    'separate_lr': {
        'name': 'Separate Gate Learning Rate (10x)',
        'use_entropy_reg': False,
        'use_temp_annealing': False,
        'use_separate_gate_lr': True,
        'gate_lr_scale': 10.0,
        'use_warmup_equal_gates': False,
    },
    'equal_init': {
        'name': 'Equal Gate Initialization',
        'use_entropy_reg': False,
        'use_temp_annealing': False,
        'use_separate_gate_lr': False,
        'use_warmup_equal_gates': True,
    },
    'entropy_equal': {
        'name': 'Entropy Reg + Equal Init',
        'use_entropy_reg': True,
        'entropy_weight': 0.01,
        'use_temp_annealing': False,
        'use_separate_gate_lr': False,
        'use_warmup_equal_gates': True,
    },
    'all_strategies': {
        'name': 'All Strategies Combined',
        'use_entropy_reg': True,
        'entropy_weight': 0.005,
        'use_temp_annealing': False,
        'use_separate_gate_lr': True,
        'gate_lr_scale': 5.0,
        'use_warmup_equal_gates': True,
    },
}


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_experiment(
    dataset: str,
    branches: str | List[str],
    strategy: str = 'baseline',
    quick: bool = False,
    seed: int = 42,
    output_dir: str = 'results_v2',
) -> Dict:
    """Run a single experiment with specified configuration."""
    set_seed(seed)
    
    cfg = DATASET_CONFIGS[dataset]
    strategy_cfg = STRATEGY_CONFIGS[strategy]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = cfg['quick_epochs'] if quick else cfg['epochs']
    
    # Create model
    model = HybridKAN(
        input_dim=cfg['input_dim'],
        hidden_dims=cfg['hidden_dims'],
        num_classes=cfg['num_classes'],
        activation_functions=branches,
        use_cnn=cfg['use_cnn'],
        cnn_channels=cfg['cnn_channels'],
        cnn_output_dim=cfg['cnn_output_dim'],
        use_residual=True,
        dropout_rate=0.3,
    )
    
    # Get data
    loader_fn = cfg['loader_fn']
    kwargs = {'batch_size': cfg['batch_size'], 'use_cnn': cfg['use_cnn']}
    train_loader, test_loader = loader_fn(**kwargs)
    
    # Experiment name
    branch_str = 'all' if branches == 'all' else '_'.join(branches)
    exp_name = f"{dataset}_{branch_str}_{strategy}"
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Strategy: {strategy_cfg['name']}")
    print(f"Branches: {branches}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}\n")
    
    # Create trainer
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        epochs=epochs,
        output_dir=output_dir,
        experiment_name=exp_name,
        num_classes=cfg['num_classes'],
        **{k: v for k, v in strategy_cfg.items() if k != 'name'},
    )
    
    # Train
    results = trainer.train()
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {exp_name}")
    print(f"{'='*60}")
    print(f"Best Accuracy: {results['best_accuracy']:.2f}%")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Parameters: {results['parameter_count']:,}")
    print(f"Gate Entropy: {results['gate_entropy'].get('average_normalized_entropy', 0):.3f}")
    print(f"Inference Time: {results['inference_time']['mean_ms']:.2f}ms")
    
    # Print comprehensive metrics
    fm = results['final_metrics']
    print(f"\nComprehensive Metrics:")
    print(f"  Macro Precision: {fm.get('macro_precision', 0):.2f}%")
    print(f"  Macro Recall: {fm.get('macro_recall', 0):.2f}%")
    print(f"  Macro F1: {fm.get('macro_f1', 0):.2f}%")
    if 'top3_accuracy' in fm:
        print(f"  Top-3 Accuracy: {fm['top3_accuracy']:.2f}%")
    if 'top5_accuracy' in fm:
        print(f"  Top-5 Accuracy: {fm['top5_accuracy']:.2f}%")
    
    return results


def run_comparison(dataset: str, quick: bool = False):
    """Run full comparison: ReLU vs HybridKAN with different strategies."""
    results = []
    
    # 1. ReLU baseline
    print("\n" + "#"*60)
    print("# RUNNING RELU BASELINE")
    print("#"*60)
    relu_result = run_experiment(dataset, ['relu'], 'baseline', quick)
    relu_result['experiment_type'] = 'relu_baseline'
    results.append(relu_result)
    
    # 2. HybridKAN baseline
    print("\n" + "#"*60)
    print("# RUNNING HYBRIDKAN BASELINE")
    print("#"*60)
    hybrid_baseline = run_experiment(dataset, 'all', 'baseline', quick)
    hybrid_baseline['experiment_type'] = 'hybrid_baseline'
    results.append(hybrid_baseline)
    
    # 3. HybridKAN with entropy regularization
    print("\n" + "#"*60)
    print("# RUNNING HYBRIDKAN + ENTROPY REG")
    print("#"*60)
    hybrid_entropy = run_experiment(dataset, 'all', 'entropy_reg', quick)
    hybrid_entropy['experiment_type'] = 'hybrid_entropy'
    results.append(hybrid_entropy)
    
    # 4. HybridKAN with equal init
    print("\n" + "#"*60)
    print("# RUNNING HYBRIDKAN + EQUAL INIT")
    print("#"*60)
    hybrid_equal = run_experiment(dataset, 'all', 'equal_init', quick)
    hybrid_equal['experiment_type'] = 'hybrid_equal_init'
    results.append(hybrid_equal)
    
    # 5. HybridKAN with all strategies
    print("\n" + "#"*60)
    print("# RUNNING HYBRIDKAN + ALL STRATEGIES")
    print("#"*60)
    hybrid_all = run_experiment(dataset, 'all', 'all_strategies', quick)
    hybrid_all['experiment_type'] = 'hybrid_all_strategies'
    results.append(hybrid_all)
    
    # Summary
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Experiment':<30} {'Accuracy':>10} {'Gate Entropy':>15} {'Params':>12}")
    print("-"*80)
    
    for r in results:
        ent = r['gate_entropy'].get('average_normalized_entropy', 0)
        print(f"{r['experiment_type']:<30} {r['best_accuracy']:>9.2f}% {ent:>14.3f} {r['parameter_count']:>12,}")
    
    # Did we beat ReLU?
    relu_acc = relu_result['best_accuracy']
    best_hybrid = max(results[1:], key=lambda x: x['best_accuracy'])
    
    print("\n" + "="*80)
    if best_hybrid['best_accuracy'] > relu_acc:
        diff = best_hybrid['best_accuracy'] - relu_acc
        print(f"SUCCESS! Best HybridKAN ({best_hybrid['experiment_type']}) beats ReLU by {diff:.2f}%")
    else:
        diff = relu_acc - best_hybrid['best_accuracy']
        print(f"ReLU still wins by {diff:.2f}%")
        print("Suggestions:")
        print("  1. Try longer training (more epochs)")
        print("  2. Increase entropy regularization weight")
        print("  3. Try different branch combinations")
        print("  4. Adjust learning rate for gates")
    print("="*80)
    
    # Save comparison
    output_dir = Path('results_v2') / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'comparison_summary.json', 'w', encoding='utf-8') as f:
        summary = {
            'dataset': dataset,
            'relu_accuracy': relu_acc,
            'best_hybrid_accuracy': best_hybrid['best_accuracy'],
            'best_hybrid_strategy': best_hybrid['experiment_type'],
            'beat_relu': best_hybrid['best_accuracy'] > relu_acc,
            'difference': best_hybrid['best_accuracy'] - relu_acc,
        }
        json.dump(summary, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='HybridKAN Enhanced Experiments V2')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'])
    parser.add_argument('--branches', type=str, default='all', help='all or comma-separated list')
    parser.add_argument('--strategy', type=str, default='baseline', choices=list(STRATEGY_CONFIGS.keys()))
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer epochs')
    parser.add_argument('--comparison', action='store_true', help='Run full comparison')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.comparison:
        run_comparison(args.dataset, args.quick)
    else:
        branches = args.branches if args.branches == 'all' else args.branches.split(',')
        run_experiment(args.dataset, branches, args.strategy, args.quick, args.seed)


if __name__ == '__main__':
    main()
