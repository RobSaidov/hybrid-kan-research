#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate publication-quality figures for HybridKAN research paper.

Creates:
1. Ablation study bar chart
2. Training curves comparison
3. Gate evolution heatmap
4. Architecture diagram
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pandas as pd

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color scheme
COLORS = {
    'all_branches': '#3B82F6',      # Blue
    'relu_only': '#F59E0B',         # Amber
    'all_no_residual': '#6B7280',   # Gray
    'gabor': '#3B82F6',
    'legendre': '#10B981',
    'chebyshev': '#059669',
    'hermite': '#6EE7B7',
    'fourier': '#8B5CF6',
    'relu': '#F59E0B',
}


def load_results(results_dir: Path) -> dict:
    """Load all experiment results from directory."""
    results = {}
    for variant_dir in results_dir.iterdir():
        if variant_dir.is_dir():
            summary_file = variant_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    results[variant_dir.name] = json.load(f)
    return results


def plot_ablation_study(results: dict, output_path: Path):
    """Create ablation study bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract ablation results
    baseline = results.get('all_branches', {}).get('best_accuracy', 0)
    
    ablation_data = []
    for name, data in results.items():
        if name.startswith('all_except_'):
            branch = name.replace('all_except_', '').title()
            acc = data.get('best_accuracy', 0)
            delta = acc - baseline
            ablation_data.append((branch, acc, delta))
    
    # Sort by accuracy
    ablation_data.sort(key=lambda x: x[1], reverse=True)
    
    # Add baseline
    branches = ['All Branches'] + [d[0] for d in ablation_data]
    accuracies = [baseline] + [d[1] for d in ablation_data]
    deltas = [0] + [d[2] for d in ablation_data]
    
    # Create bars with colors based on delta
    colors = ['#3B82F6']  # Blue for baseline
    for d in deltas[1:]:
        if d > 0:
            colors.append('#10B981')  # Green for improvement
        else:
            colors.append('#EF4444')  # Red for degradation
    
    bars = ax.bar(branches, accuracies, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, acc, delta in zip(bars, accuracies, deltas):
        height = bar.get_height()
        label = f'{acc:.2f}%'
        if delta != 0:
            sign = '+' if delta > 0 else ''
            label += f'\n({sign}{delta:.2f}%)'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Ablation Study: Leave-One-Out Branch Removal (CIFAR-10)')
    ax.set_ylim(84, 87)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#3B82F6', label='Baseline (All Branches)'),
        mpatches.Patch(color='#10B981', label='Improvement when removed'),
        mpatches.Patch(color='#EF4444', label='Degradation when removed'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved ablation study plot: {output_path}")


def plot_main_comparison(results: dict, output_path: Path):
    """Create main model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ['all_branches', 'relu_only', 'all_no_residual']
    labels = ['HybridKAN\n(All Branches)', 'ReLU Only\n(Baseline)', 'All Branches\n(No Residual)']
    colors = ['#3B82F6', '#F59E0B', '#6B7280']
    
    accuracies = [results.get(m, {}).get('best_accuracy', 0) for m in models]
    params = [results.get(m, {}).get('parameters', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.6
    
    bars = ax.bar(x, accuracies, width, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, acc, param in zip(bars, accuracies, params):
        height = bar.get_height()
        ax.annotate(f'{acc:.2f}%\n({param/1e6:.2f}M params)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Model Comparison on CIFAR-10')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(84, 87.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved main comparison plot: {output_path}")


def plot_training_curves(results_dir: Path, output_path: Path):
    """Plot training curves for main models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = ['all_branches', 'relu_only', 'all_no_residual']
    labels = ['All Branches', 'ReLU Only', 'No Residual']
    colors = ['#3B82F6', '#F59E0B', '#6B7280']
    
    for model, label, color in zip(models, labels, colors):
        metrics_file = results_dir / model / 'logs' / f'{model}_metrics.csv'
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            axes[0].plot(df['epoch'], df['train_accuracy'], label=label, color=color, alpha=0.8)
            axes[1].plot(df['epoch'], df['test_accuracy'], label=label, color=color, alpha=0.8)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Accuracy (%)')
    axes[0].set_title('Training Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved training curves: {output_path}")


def plot_gate_evolution(results_dir: Path, output_path: Path):
    """Plot gate weight evolution during training."""
    gates_file = results_dir / 'all_branches' / 'logs' / 'all_branches_gates.json'
    
    if not gates_file.exists():
        print(f"⚠ Gates file not found: {gates_file}")
        return
    
    with open(gates_file, 'r') as f:
        gates_data = json.load(f)
    
    branch_gates = gates_data.get('branch_gates', [])
    if not branch_gates:
        return
    
    # Extract data for layer 0
    epochs = []
    gate_values = {branch: [] for branch in ['gabor', 'legendre', 'chebyshev', 'hermite', 'fourier', 'relu']}
    
    for record in branch_gates:
        epochs.append(record['epoch'])
        layer_0_gates = record['gates'].get('0', {})
        for branch in gate_values.keys():
            gate_values[branch].append(layer_0_gates.get(branch, 0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for branch, values in gate_values.items():
        ax.plot(epochs, values, label=branch.title(), color=COLORS.get(branch, '#333'), linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gate Weight (after softplus)')
    ax.set_title('Branch Gate Evolution During Training (Layer 0)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved gate evolution plot: {output_path}")


def plot_final_gate_heatmap(results: dict, output_path: Path):
    """Create heatmap of final gate weights across layers."""
    all_branches = results.get('all_branches', {})
    gate_summary = all_branches.get('gate_summary', {})
    final_gates = gate_summary.get('final_branch_gates', {}).get('gates', {})
    
    if not final_gates:
        print("⚠ No gate data found")
        return
    
    branches = ['gabor', 'legendre', 'chebyshev', 'hermite', 'fourier', 'relu']
    layers = sorted(final_gates.keys(), key=int)
    
    # Build matrix
    matrix = np.zeros((len(branches), len(layers)))
    for j, layer in enumerate(layers):
        for i, branch in enumerate(branches):
            matrix[i, j] = final_gates[layer].get(branch, 0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    
    # Labels
    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(branches)))
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.set_yticklabels([b.title() for b in branches])
    
    # Add values
    for i in range(len(branches)):
        for j in range(len(layers)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="white" if matrix[i, j] > 0.5 else "black")
    
    ax.set_title('Learned Branch Gate Weights by Layer')
    fig.colorbar(im, ax=ax, label='Gate Weight')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved gate heatmap: {output_path}")


def plot_branch_importance(results: dict, output_path: Path):
    """Create branch importance chart based on ablation study."""
    baseline = results.get('all_branches', {}).get('best_accuracy', 0)
    
    importance = {}
    for name, data in results.items():
        if name.startswith('all_except_'):
            branch = name.replace('all_except_', '')
            acc = data.get('best_accuracy', 0)
            # Negative delta = branch is important (removing hurts)
            importance[branch] = baseline - acc
    
    # Sort by importance
    sorted_branches = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    branches = [b[0].title() for b in sorted_branches]
    values = [b[1] for b in sorted_branches]
    colors = ['#10B981' if v > 0 else '#EF4444' for v in values]
    
    bars = ax.barh(branches, values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Accuracy Drop When Removed (%)')
    ax.set_title('Branch Importance (CIFAR-10)')
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.annotate(f'{val:+.2f}%',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(3 if val >= 0 else -3, 0),
                    textcoords="offset points",
                    ha='left' if val >= 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved branch importance plot: {output_path}")


def main():
    """Generate all figures."""
    results_dir = Path('results/cifar10')
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*50)
    print("Generating Publication Figures")
    print("="*50 + "\n")
    
    # Load results
    results = load_results(results_dir)
    print(f"Loaded {len(results)} experiment results\n")
    
    # Generate figures
    plot_main_comparison(results, figures_dir / 'main_comparison.png')
    plot_ablation_study(results, figures_dir / 'ablation_study.png')
    plot_training_curves(results_dir, figures_dir / 'training_curves.png')
    plot_gate_evolution(results_dir, figures_dir / 'gate_evolution.png')
    plot_final_gate_heatmap(results, figures_dir / 'gate_heatmap.png')
    plot_branch_importance(results, figures_dir / 'branch_importance.png')
    
    print("\n" + "="*50)
    print(f"All figures saved to: {figures_dir.absolute()}")
    print("="*50)


if __name__ == '__main__':
    main()
