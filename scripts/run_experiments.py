#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HybridKAN Experiment Runner

Comprehensive experiment suite for evaluating HybridKAN architecture:
- Main comparison: All branches vs ReLU-only vs No residual
- Ablation study: Leave-one-out for each branch
- Branch combinations: Various interesting combinations

Usage:
    python scripts/run_experiments.py --dataset cifar10 --experiment full
    python scripts/run_experiments.py --dataset mnist --quick
    python scripts/run_experiments.py --dataset all --experiment full
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
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hybridkan import HybridKAN, Trainer, set_seed
from hybridkan.data import (
    get_mnist_loaders,
    get_cifar10_loaders,
    get_wine_loaders,
    get_california_housing_loaders,
    DATASET_INFO,
)
from hybridkan.trainer import TrainingConfig


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

# All available branches
ALL_BRANCHES = ['gabor', 'legendre', 'chebyshev', 'hermite', 'fourier', 'relu']

# Dataset-specific configurations
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
        'train_size': 60000,
    },
    'cifar10': {
        'loader_fn': get_cifar10_loaders,
        'input_dim': 3072,
        'num_classes': 10,
        'hidden_dims': [256, 128, 64],  # Reduced for 6GB VRAM
        'use_cnn': True,
        'cnn_channels': 3,
        'cnn_output_dim': 256,  # Reduced
        'batch_size': 64,  # Reduced from 128 for memory
        'epochs': 100,
        'quick_epochs': 15,
        'train_size': 50000,
    },
    'wine': {
        'loader_fn': get_wine_loaders,
        'input_dim': 13,
        'num_classes': 3,
        'hidden_dims': [64, 32],
        'use_cnn': False,
        'cnn_channels': None,
        'cnn_output_dim': None,
        'batch_size': 32,
        'epochs': 100,
        'quick_epochs': 20,
        'train_size': None,
    },
    'california_housing': {
        'loader_fn': get_california_housing_loaders,
        'input_dim': 8,
        'num_classes': 1,  # Regression
        'hidden_dims': [128, 64, 32],
        'use_cnn': False,
        'cnn_channels': None,
        'cnn_output_dim': None,
        'batch_size': 64,
        'epochs': 100,
        'quick_epochs': 20,
        'train_size': None,
    },
}

# Experiment configurations
EXPERIMENT_CONFIGS = {
    'main': {
        'name': 'Main Comparison',
        'variants': [
            {'name': 'all_branches', 'branches': 'all', 'use_residual': True},
            {'name': 'relu_only', 'branches': ['relu'], 'use_residual': True},
            {'name': 'all_no_residual', 'branches': 'all', 'use_residual': False},
        ],
    },
    'ablation': {
        'name': 'Ablation Study (Leave-One-Out)',
        'variants': [
            {'name': 'all_branches', 'branches': 'all', 'use_residual': True},
            {'name': 'all_except_gabor', 'branches': ['legendre', 'chebyshev', 'hermite', 'fourier', 'relu'], 'use_residual': True},
            {'name': 'all_except_legendre', 'branches': ['gabor', 'chebyshev', 'hermite', 'fourier', 'relu'], 'use_residual': True},
            {'name': 'all_except_chebyshev', 'branches': ['gabor', 'legendre', 'hermite', 'fourier', 'relu'], 'use_residual': True},
            {'name': 'all_except_hermite', 'branches': ['gabor', 'legendre', 'chebyshev', 'fourier', 'relu'], 'use_residual': True},
            {'name': 'all_except_fourier', 'branches': ['gabor', 'legendre', 'chebyshev', 'hermite', 'relu'], 'use_residual': True},
            {'name': 'all_except_relu', 'branches': ['gabor', 'legendre', 'chebyshev', 'hermite', 'fourier'], 'use_residual': True},
        ],
    },
    'combinations': {
        'name': 'Branch Combinations',
        'variants': [
            {'name': 'fourier_relu', 'branches': ['fourier', 'relu'], 'use_residual': True},
            {'name': 'legendre_relu', 'branches': ['legendre', 'relu'], 'use_residual': True},
            {'name': 'polynomials_only', 'branches': ['legendre', 'chebyshev', 'hermite'], 'use_residual': True},
            {'name': 'periodic_localized', 'branches': ['fourier', 'gabor'], 'use_residual': True},
            {'name': 'best_three', 'branches': ['fourier', 'legendre', 'relu'], 'use_residual': True},
        ],
    },
    'full': {
        'name': 'Full Experiment Suite',
        'variants': None,  # Will combine main + ablation
    },
}


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """Runs and manages HybridKAN experiments."""
    
    def __init__(
        self,
        dataset: str,
        output_dir: str = 'results',
        seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        self.dataset = dataset
        self.dataset_config = DATASET_CONFIGS[dataset]
        self.output_dir = Path(output_dir) / dataset
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.results: List[Dict] = []
        
    def _create_model(
        self,
        branches: str | List[str],
        use_residual: bool,
    ) -> HybridKAN:
        """Create a HybridKAN model with specified configuration."""
        cfg = self.dataset_config
        
        model = HybridKAN(
            input_dim=cfg['input_dim'],
            hidden_dims=cfg['hidden_dims'],
            num_classes=cfg['num_classes'],
            activation_functions=branches,
            use_cnn=cfg['use_cnn'],
            cnn_channels=cfg['cnn_channels'],
            cnn_output_dim=cfg['cnn_output_dim'],
            use_residual=use_residual,
            dropout_rate=0.3,
        )
        
        return model
    
    def _get_data_loaders(self) -> Tuple:
        """Get train and test data loaders for the dataset."""
        cfg = self.dataset_config
        loader_fn = cfg['loader_fn']
        
        kwargs = {'batch_size': cfg['batch_size']}
        if cfg['train_size']:
            kwargs['train_size'] = cfg['train_size']
        if cfg['use_cnn']:
            kwargs['use_cnn'] = True
        
        return loader_fn(**kwargs)
    
    def run_single_experiment(
        self,
        variant_name: str,
        branches: str | List[str],
        use_residual: bool,
        epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """Run a single experiment variant."""
        set_seed(self.seed)
        
        cfg = self.dataset_config
        epochs = epochs or cfg['epochs']
        
        print(f"\n{'='*60}")
        print(f"Running: {variant_name}")
        print(f"Branches: {branches}")
        print(f"Residual: {use_residual}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        # Create model
        model = self._create_model(branches, use_residual)
        
        # Get data loaders
        train_loader, test_loader = self._get_data_loaders()
        
        # Create output directory for this variant
        variant_dir = self.output_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        
        # Training config
        training_config = TrainingConfig(
            epochs=epochs,
            learning_rate=1e-3,
            weight_decay=1e-4,
            patience=15,
            use_amp=torch.cuda.is_available(),
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            config=training_config,
            output_dir=str(variant_dir),
            experiment_name=variant_name,
            device=self.device,
        )
        
        # Train
        start_time = time.time()
        results = trainer.train(verbose=verbose)
        elapsed_time = time.time() - start_time
        
        # Compile results
        experiment_result = {
            'dataset': self.dataset,
            'variant_name': variant_name,
            'branches': branches if isinstance(branches, list) else ALL_BRANCHES,
            'use_residual': use_residual,
            'best_accuracy': results['best_accuracy'],
            'best_epoch': results['best_epoch'],
            'total_epochs': results['total_epochs'],
            'elapsed_time': elapsed_time,
            'parameters': results.get('parameter_counts', {}).get('total', 0),
            'gate_summary': results.get('gate_summary', {}),
        }
        
        # Save individual result
        with open(variant_dir / 'summary.json', 'w') as f:
            json.dump(experiment_result, f, indent=2, default=str)
        
        self.results.append(experiment_result)
        
        print(f"\n✓ Completed {variant_name}: {results['best_accuracy']:.2f}% accuracy")
        
        return experiment_result
    
    def run_experiment_suite(
        self,
        experiment_type: str = 'main',
        quick: bool = False,
    ) -> List[Dict]:
        """Run a full experiment suite."""
        epochs = self.dataset_config['quick_epochs'] if quick else None
        
        if experiment_type == 'full':
            # Combine main and ablation experiments
            variants = (
                EXPERIMENT_CONFIGS['main']['variants'] +
                EXPERIMENT_CONFIGS['ablation']['variants'][1:]  # Skip duplicate all_branches
            )
        else:
            variants = EXPERIMENT_CONFIGS[experiment_type]['variants']
        
        print(f"\n{'#'*60}")
        print(f"# Running {experiment_type.upper()} experiments on {self.dataset.upper()}")
        print(f"# Device: {self.device}")
        print(f"# Quick mode: {quick}")
        print(f"# Variants: {len(variants)}")
        print(f"{'#'*60}")
        
        for variant in variants:
            self.run_single_experiment(
                variant_name=variant['name'],
                branches=variant['branches'],
                use_residual=variant['use_residual'],
                epochs=epochs,
            )
        
        # Generate summary report
        self._generate_report()
        
        return self.results
    
    def _generate_report(self):
        """Generate markdown report of all results."""
        if not self.results:
            return
        
        # Sort results by accuracy
        sorted_results = sorted(self.results, key=lambda x: x['best_accuracy'], reverse=True)
        
        # Find baseline (all_branches) for computing deltas
        baseline_acc = None
        for r in sorted_results:
            if r['variant_name'] == 'all_branches':
                baseline_acc = r['best_accuracy']
                break
        
        # Generate markdown
        report_lines = [
            f"# HybridKAN Experiment Results - {self.dataset.upper()}",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nDevice: {self.device}",
            f"\n## Summary",
            "",
        ]
        
        # Main results table
        report_lines.extend([
            "### Main Comparison",
            "",
            "| Model | Accuracy | Parameters | Time (s) |",
            "|-------|----------|------------|----------|",
        ])
        
        for r in sorted_results:
            if r['variant_name'] in ['all_branches', 'relu_only', 'all_no_residual']:
                params = f"{r['parameters']:,}" if r['parameters'] else "N/A"
                report_lines.append(
                    f"| {r['variant_name']} | {r['best_accuracy']:.2f}% | {params} | {r['elapsed_time']:.1f} |"
                )
        
        # Ablation results table
        ablation_results = [r for r in sorted_results if r['variant_name'].startswith('all_except')]
        if ablation_results:
            report_lines.extend([
                "",
                "### Ablation Study (Leave-One-Out)",
                "",
                "| Excluded Branch | Accuracy | Δ from All |",
                "|-----------------|----------|------------|",
            ])
            
            # Add baseline
            if baseline_acc:
                report_lines.append(f"| None (All) | {baseline_acc:.2f}% | — |")
            
            for r in ablation_results:
                excluded = r['variant_name'].replace('all_except_', '- ').title()
                delta = r['best_accuracy'] - baseline_acc if baseline_acc else 0
                delta_str = f"{delta:+.2f}%" if baseline_acc else "N/A"
                report_lines.append(f"| {excluded} | {r['best_accuracy']:.2f}% | {delta_str} |")
        
        # Branch combinations
        combo_results = [r for r in sorted_results 
                        if r['variant_name'] not in ['all_branches', 'relu_only', 'all_no_residual']
                        and not r['variant_name'].startswith('all_except')]
        if combo_results:
            report_lines.extend([
                "",
                "### Branch Combinations",
                "",
                "| Combination | Accuracy | Branches |",
                "|-------------|----------|----------|",
            ])
            for r in combo_results:
                branches_str = ', '.join(r['branches']) if isinstance(r['branches'], list) else r['branches']
                report_lines.append(f"| {r['variant_name']} | {r['best_accuracy']:.2f}% | {branches_str} |")
        
        # Best result summary
        best = sorted_results[0]
        report_lines.extend([
            "",
            "## Best Configuration",
            f"- **Model**: {best['variant_name']}",
            f"- **Accuracy**: {best['best_accuracy']:.2f}%",
            f"- **Best Epoch**: {best['best_epoch']}",
            f"- **Training Time**: {best['elapsed_time']:.1f}s",
        ])
        
        # Write report
        report_path = self.output_dir / 'RESULTS.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n✓ Report saved to {report_path}")
        
        # Also save CSV summary
        csv_path = self.output_dir / 'results_summary.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'dataset', 'variant_name', 'best_accuracy', 'parameters',
                'elapsed_time', 'best_epoch', 'total_epochs', 'use_residual'
            ])
            writer.writeheader()
            for r in sorted_results:
                writer.writerow({
                    'dataset': r['dataset'],
                    'variant_name': r['variant_name'],
                    'best_accuracy': r['best_accuracy'],
                    'parameters': r['parameters'],
                    'elapsed_time': r['elapsed_time'],
                    'best_epoch': r['best_epoch'],
                    'total_epochs': r['total_epochs'],
                    'use_residual': r['use_residual'],
                })
        
        print(f"✓ CSV saved to {csv_path}")


def generate_global_report(results_dir: Path):
    """Generate a combined report across all datasets."""
    all_results = []
    
    for dataset_dir in results_dir.iterdir():
        if dataset_dir.is_dir():
            csv_path = dataset_dir / 'results_summary.csv'
            if csv_path.exists():
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_results.append(row)
    
    if not all_results:
        return
    
    # Generate global RESULTS.md
    report_lines = [
        "# HybridKAN Comprehensive Experiment Results",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    # Group by dataset
    datasets = sorted(set(r['dataset'] for r in all_results))
    
    for dataset in datasets:
        dataset_results = [r for r in all_results if r['dataset'] == dataset]
        dataset_results.sort(key=lambda x: float(x['best_accuracy']), reverse=True)
        
        report_lines.extend([
            f"\n## {dataset.upper()}",
            "",
            "| Model | Accuracy | Parameters |",
            "|-------|----------|------------|",
        ])
        
        for r in dataset_results[:5]:  # Top 5
            acc = float(r['best_accuracy'])
            params = r['parameters'] if r['parameters'] else "N/A"
            report_lines.append(f"| {r['variant_name']} | {acc:.2f}% | {params} |")
    
    global_report_path = results_dir / 'RESULTS.md'
    with open(global_report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Combined CSV
    combined_csv_path = results_dir / 'results_summary.csv'
    with open(combined_csv_path, 'w', newline='') as f:
        fieldnames = list(all_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n✓ Global report saved to {global_report_path}")
    print(f"✓ Combined CSV saved to {combined_csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run HybridKAN experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_experiments.py --dataset mnist --quick
  python scripts/run_experiments.py --dataset cifar10 --experiment full
  python scripts/run_experiments.py --dataset all --experiment main
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mnist', 'cifar10', 'wine', 'california_housing', 'all'],
        default='cifar10',
        help='Dataset to run experiments on (default: cifar10)'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['main', 'ablation', 'combinations', 'full'],
        default='main',
        help='Type of experiment to run (default: main)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick experiments with fewer epochs'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Print system info
    print("\n" + "="*60)
    print("HybridKAN Experiment Runner")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Seed: {args.seed}")
    print("="*60 + "\n")
    
    # Determine datasets to run
    if args.dataset == 'all':
        datasets = ['mnist', 'cifar10', 'wine']
    else:
        datasets = [args.dataset]
    
    # Run experiments
    for dataset in datasets:
        runner = ExperimentRunner(
            dataset=dataset,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        runner.run_experiment_suite(
            experiment_type=args.experiment,
            quick=args.quick,
        )
    
    # Generate global report if multiple datasets
    if len(datasets) > 1:
        generate_global_report(Path(args.output_dir))
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print(f"Results saved to: {args.output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()
