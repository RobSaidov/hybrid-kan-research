# -*- coding: utf-8 -*-
"""
Visualization Module for HybridKAN

Creates publication-quality figures for:
- Architecture diagrams
- Training curves
- Gate evolution plots
- Confusion matrices
- Basis function visualizations

All figures are generated as vector graphics (PDF/SVG) for scalability.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


# Publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
})


# Color scheme for branches
BRANCH_COLORS = {
    'gabor': '#3B82F6',      # Blue
    'legendre': '#10B981',   # Emerald
    'chebyshev': '#059669',  # Green
    'hermite': '#6EE7B7',    # Light green
    'fourier': '#8B5CF6',    # Purple
    'relu': '#F59E0B',       # Amber
}

RESIDUAL_COLOR = '#EF4444'   # Red for skip connections
GATE_COLOR = '#FBBF24'       # Yellow for gates


def create_architecture_diagram(
    hidden_dims: List[int] = [256, 128, 64],
    branches: List[str] = ['gabor', 'legendre', 'chebyshev', 'hermite', 'fourier', 'relu'],
    use_residual: bool = True,
    use_cnn: bool = True,
    show_gates: bool = True,
    gate_weights: Optional[Dict] = None,
    residual_weights: Optional[Dict] = None,
    output_path: str = "architecture.pdf",
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 1200,
) -> plt.Figure:
    """
    Create publication-quality architecture diagram.
    
    Args:
        hidden_dims: Layer widths
        branches: Active branches
        use_residual: Show residual connections
        use_cnn: Show CNN preprocessor
        show_gates: Show gate values
        gate_weights: Optional dict of gate weights to display
        residual_weights: Optional dict of residual gate weights
        output_path: Save path (supports .pdf, .svg, .png)
        figsize: Figure size in inches
        dpi: Resolution for raster outputs
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Layout parameters
    start_x = 5
    block_width = 12
    block_spacing = 3
    branch_height = 4
    branch_spacing = 0.8
    
    def draw_rounded_box(x, y, width, height, color='#F8FAFC', 
                         edge_color='#334155', linewidth=1.5, label=None,
                         fontsize=9, alpha=1.0):
        """Draw a rounded rectangle with optional label."""
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            facecolor=color, edgecolor=edge_color,
            linewidth=linewidth, alpha=alpha, zorder=2
        )
        ax.add_patch(box)
        if label:
            ax.text(x + width/2, y + height/2, label,
                   ha='center', va='center', fontsize=fontsize,
                   fontweight='medium', zorder=3)
        return box
    
    def draw_arrow(start, end, color='#475569', style='->', linewidth=1.2,
                  connectionstyle='arc3,rad=0'):
        """Draw an arrow between two points."""
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle=style, color=color,
                                  lw=linewidth, connectionstyle=connectionstyle),
                   zorder=1)
    
    def draw_gate_value(x, y, value, branch_name):
        """Draw gate value indicator."""
        color = BRANCH_COLORS.get(branch_name, '#666666')
        circle = plt.Circle((x, y), 0.8, color=GATE_COLOR, 
                           ec=color, linewidth=1.5, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, f'γ', ha='center', va='center', fontsize=7,
               fontweight='bold', zorder=5)
        if value is not None:
            ax.text(x, y - 1.2, f'{value:.2f}', ha='center', va='top',
                   fontsize=6, color='#666666', zorder=5)
    
    current_x = start_x
    
    # ==================== INPUT ====================
    input_box = draw_rounded_box(current_x, 25, 8, 6, '#E0F2FE', label='Input\nx ∈ ℝᵈ')
    current_x += 10
    
    # ==================== CNN PREPROCESSOR ====================
    if use_cnn:
        # CNN container
        cnn_box = draw_rounded_box(current_x, 20, 15, 16, '#F0FDF4', 
                                   edge_color='#86EFAC', linewidth=2)
        ax.text(current_x + 7.5, 37, 'CNN Preprocessor (optional)',
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               color='#166534')
        
        # Conv blocks
        conv_y = 32
        for i, (filters, label) in enumerate([
            (32, 'Conv 32\n3×3, Pool'),
            (64, 'Conv 64\n3×3, Pool'),
            (128, 'Conv 128\n3×3, AvgPool')
        ]):
            draw_rounded_box(current_x + 1 + i*4.5, conv_y - 8, 4, 6, 
                           '#DCFCE7', label=label, fontsize=7)
        
        # Flatten
        draw_rounded_box(current_x + 4, 22, 7, 4, '#BBF7D0', label='Flatten')
        
        draw_arrow((start_x + 8, 28), (current_x, 28))
        current_x += 17
    
    # ==================== LAYER NORM ====================
    ln_box = draw_rounded_box(current_x, 25, 6, 6, '#EEF2FF', label='Layer\nNorm')
    draw_arrow((current_x - 2, 28), (current_x, 28))
    current_x += 8
    
    # Store block positions for residual connections
    block_positions = []
    block_output_x = []
    
    # ==================== HYBRID BLOCKS ====================
    for block_idx, hidden_dim in enumerate(hidden_dims):
        block_start_x = current_x
        
        # Block container
        block_height = len(branches) * (branch_height + branch_spacing) + 4
        block_y_start = 28 - block_height/2
        
        # Block background
        draw_rounded_box(current_x - 1, block_y_start - 1, block_width + 2, 
                        block_height + 2, '#FFFBEB', edge_color='#FCD34D',
                        linewidth=2, alpha=0.5)
        
        ax.text(current_x + block_width/2, block_y_start + block_height + 1.5,
               f'Block {block_idx + 1}\n(dim={hidden_dim})',
               ha='center', va='bottom', fontsize=9, fontweight='bold',
               color='#92400E')
        
        # Draw branches
        branch_y = block_y_start + block_height - 3
        branch_outputs = []
        
        for branch_name in branches:
            color = BRANCH_COLORS.get(branch_name, '#999999')
            
            # Branch box
            draw_rounded_box(current_x, branch_y, 6, branch_height - 0.5,
                           color + '33', edge_color=color, linewidth=1.2,
                           label=branch_name.capitalize(), fontsize=8)
            
            # Gate indicator
            if show_gates:
                gate_val = None
                if gate_weights and block_idx in gate_weights:
                    gate_val = gate_weights[block_idx].get(branch_name)
                draw_gate_value(current_x + 7.5, branch_y + branch_height/2 - 0.25,
                              gate_val, branch_name)
            
            branch_outputs.append(branch_y + branch_height/2 - 0.25)
            branch_y -= (branch_height + branch_spacing)
        
        # Concatenation symbol
        concat_x = current_x + 9.5
        concat_y = block_y_start + block_height/2
        circle = plt.Circle((concat_x, concat_y), 1.2, color='white',
                           ec='#334155', linewidth=1.5, zorder=4)
        ax.add_patch(circle)
        ax.text(concat_x, concat_y, '⊕', ha='center', va='center',
               fontsize=14, fontweight='bold', zorder=5)
        
        # Connect branches to concat
        for by in branch_outputs:
            draw_arrow((current_x + 8.3, by), (concat_x - 1.2, concat_y),
                      connectionstyle='arc3,rad=0.1')
        
        # Post-concat processing
        post_x = concat_x + 2
        draw_rounded_box(post_x, concat_y - 1.5, 4, 3, '#F3F4F6',
                        label='BN\nGELU\nDrop', fontsize=6)
        draw_arrow((concat_x + 1.2, concat_y), (post_x, concat_y))
        
        # Projection
        proj_x = post_x + 4.5
        draw_rounded_box(proj_x, concat_y - 1, 3.5, 2, '#DBEAFE',
                        label='Proj', fontsize=8)
        draw_arrow((post_x + 4, concat_y), (proj_x, concat_y))
        
        block_positions.append((block_start_x, concat_y))
        block_output_x.append(proj_x + 3.5)
        
        # Draw input arrow to block
        if block_idx == 0:
            draw_arrow((current_x - 2, 28), (current_x, 28))
        else:
            draw_arrow((block_output_x[block_idx - 1], 28), (current_x, 28))
        
        current_x += block_width + block_spacing + 8
    
    # ==================== RESIDUAL CONNECTIONS ====================
    if use_residual and len(block_positions) > 1:
        # Input to output skip
        skip_y_top = 45
        ax.annotate('', xy=(current_x - 5, skip_y_top - 2),
                   xytext=(start_x + 4, skip_y_top - 2),
                   arrowprops=dict(arrowstyle='->', color=RESIDUAL_COLOR,
                                  lw=2, linestyle='--',
                                  connectionstyle='arc3,rad=0.15'),
                   zorder=1)
        
        # Add residual gate indicator
        res_gate_x = (start_x + 4 + current_x - 5) / 2
        circle = plt.Circle((res_gate_x, skip_y_top - 2), 1.0, 
                           color='white', ec=RESIDUAL_COLOR, 
                           linewidth=1.5, zorder=4)
        ax.add_patch(circle)
        ax.text(res_gate_x, skip_y_top - 2, 'α', ha='center', va='center',
               fontsize=9, fontweight='bold', color=RESIDUAL_COLOR, zorder=5)
        
        if residual_weights:
            ax.text(res_gate_x, skip_y_top - 4.2, 
                   f'{list(residual_weights.values())[0]:.2f}',
                   ha='center', va='top', fontsize=7, color=RESIDUAL_COLOR)
        
        ax.text(res_gate_x, skip_y_top + 0.8, 'Residual Skip',
               ha='center', va='bottom', fontsize=8, 
               color=RESIDUAL_COLOR, fontweight='medium')
    
    # ==================== OUTPUT HEAD ====================
    output_x = current_x
    draw_rounded_box(output_x, 25, 8, 6, '#FEE2E2', 
                    edge_color='#F87171', label='Output\nHead')
    draw_arrow((block_output_x[-1], 28), (output_x, 28))
    
    # Final output
    ax.text(output_x + 11, 28, 'ŷ', ha='left', va='center',
           fontsize=14, fontweight='bold')
    draw_arrow((output_x + 8, 28), (output_x + 10, 28))
    
    # ==================== LEGEND ====================
    legend_elements = []
    for name, color in BRANCH_COLORS.items():
        if name in branches:
            legend_elements.append(
                mpatches.Patch(facecolor=color + '33', edgecolor=color,
                             linewidth=1.5, label=name.capitalize())
            )
    
    legend_elements.append(
        Line2D([0], [0], color=RESIDUAL_COLOR, linewidth=2, 
               linestyle='--', label='Residual')
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GATE_COLOR,
               markeredgecolor='#666', markersize=10, label='Gate (γ/α)')
    )
    
    ax.legend(handles=legend_elements, loc='lower right',
             frameon=True, fancybox=True, framealpha=0.95,
             ncol=4, fontsize=8)
    
    # ==================== TITLE ====================
    ax.text(50, 58, 'HybridKAN Architecture',
           ha='center', va='bottom', fontsize=16, fontweight='bold')
    ax.text(50, 55.5, 'Hybrid Kolmogorov-Arnold Network with Multi-Basis Activation Functions',
           ha='center', va='bottom', fontsize=10, fontstyle='italic', color='#666666')
    
    plt.tight_layout()
    
    # Save with high DPI for publication
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   pad_inches=0.1, format=output_path.split('.')[-1])
        print(f"Saved architecture diagram to: {output_path}")
    
    return fig


def plot_training_curves(
    metrics_path: str,
    output_path: str = "training_curves.pdf",
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    Plot training curves from metrics CSV.
    
    Args:
        metrics_path: Path to metrics CSV file
        output_path: Output path for figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    import pandas as pd
    
    df = pd.read_csv(metrics_path)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss curves
    ax = axes[0]
    ax.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(df['epoch'], df['test_loss'], 'r--', label='Test Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[1]
    ax.plot(df['epoch'], df['train_accuracy'], 'b-', label='Train', linewidth=1.5)
    ax.plot(df['epoch'], df['test_accuracy'], 'r--', label='Test', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Accuracy')
    ax.legend(frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[2]
    ax.plot(df['epoch'], df['learning_rate'], 'g-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('OneCycleLR Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {output_path}")
    
    return fig


def plot_gate_evolution(
    gates_path: str,
    output_path: str = "gate_evolution.pdf",
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Plot gate value evolution throughout training.
    
    Args:
        gates_path: Path to gates JSON file
        output_path: Output path for figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    with open(gates_path, 'r') as f:
        gates_data = json.load(f)
    
    branch_gates = gates_data.get('branch_gates', [])
    
    if not branch_gates:
        print("No gate data found")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Extract gate trajectories per block
    epochs = [entry['epoch'] for entry in branch_gates]
    
    # Plot for first block
    ax = axes[0]
    for branch_name, color in BRANCH_COLORS.items():
        values = []
        for entry in branch_gates:
            block_gates = entry['gates'].get(0, {})
            values.append(block_gates.get(branch_name, np.nan))
        
        if not all(np.isnan(values)):
            ax.plot(epochs, values, color=color, label=branch_name.capitalize(),
                   linewidth=1.5, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gate Value (γ)')
    ax.set_title('Block 1 Gate Evolution')
    ax.legend(frameon=True, fancybox=True, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Residual gates
    residual_gates = gates_data.get('residual_gates', [])
    if residual_gates:
        ax = axes[1]
        res_epochs = [entry['epoch'] for entry in residual_gates]
        
        for key in residual_gates[0]['gates'].keys():
            values = [entry['gates'].get(key, np.nan) for entry in residual_gates]
            ax.plot(res_epochs, values, linewidth=1.5, marker='s', markersize=3,
                   label=key.replace('_', ' ').title())
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Residual Gate (α)')
        ax.set_title('Residual Connection Strength')
        ax.legend(frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No residual gates', ha='center', va='center',
                    transform=axes[1].transAxes)
        axes[1].set_title('Residual Gates')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved gate evolution to: {output_path}")
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "confusion_matrix.pdf",
    figsize: Tuple[float, float] = (8, 7),
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot publication-quality confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: Class labels
        output_path: Output path
        figsize: Figure size
        normalize: Normalize by row (true class)
        
    Returns:
        matplotlib Figure
    """
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm_display, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Proportion' if normalize else 'Count', rotation=-90, va='bottom')
    
    # Set ticks
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_display[i, j]
            color = 'white' if val > thresh else 'black'
            text = f'{val:{fmt}}'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to: {output_path}")
    
    return fig


if __name__ == "__main__":
    # Generate example architecture diagram
    fig = create_architecture_diagram(
        hidden_dims=[256, 128, 64],
        branches=['gabor', 'legendre', 'chebyshev', 'hermite', 'fourier', 'relu'],
        use_residual=True,
        use_cnn=True,
        show_gates=True,
        gate_weights={
            0: {'gabor': 0.18, 'legendre': 0.42, 'chebyshev': 0.38, 
                'hermite': 0.35, 'fourier': 0.45, 'relu': 0.52},
        },
        residual_weights={'residual_gate_0': 0.23},
        output_path="hybridkan_architecture.pdf",
    )
    plt.show()
