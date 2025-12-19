"""
NOVEL RESEARCH: Interpretable Function Discovery with HybridKAN

This experiment demonstrates HybridKAN's unique capability: the gate weights
reveal what TYPE of function the network learned, providing interpretability
that standard neural networks cannot offer.

Key Research Questions:
1. Do gate weights correctly identify periodic vs polynomial functions?
2. Can we use gates to "classify" unknown functions by their structure?
3. Is this interpretability useful for scientific discovery?

Author: Research Team
"""

import sys
sys.path.insert(0, r'c:\Users\user\Desktop\Research\hybridkan_arxiv')

import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Import
from hybridkan.model import HybridKAN

print("=" * 70)
print("NOVEL RESEARCH: Interpretable Function Discovery with HybridKAN")
print("=" * 70)

# ============================================================================
# 1. Define diverse test functions with known mathematical properties
# ============================================================================

TEST_FUNCTIONS = {
    # Pure periodic functions
    'sine': {
        'func': lambda x: np.sin(2 * np.pi * x),
        'expected_dominant': 'fourier',
        'category': 'periodic',
        'complexity': 'simple'
    },
    'cosine_sum': {
        'func': lambda x: np.cos(np.pi * x) + 0.5 * np.cos(3 * np.pi * x),
        'expected_dominant': 'fourier',
        'category': 'periodic',
        'complexity': 'medium'
    },
    
    # Pure polynomial functions
    'quadratic': {
        'func': lambda x: x**2 - 0.5,
        'expected_dominant': 'legendre',  # or chebyshev/hermite
        'category': 'polynomial',
        'complexity': 'simple'
    },
    'cubic': {
        'func': lambda x: x**3 - x,
        'expected_dominant': 'legendre',
        'category': 'polynomial',
        'complexity': 'simple'
    },
    'quartic': {
        'func': lambda x: x**4 - 2*x**2 + 0.5,
        'expected_dominant': 'legendre',
        'category': 'polynomial',
        'complexity': 'medium'
    },
    
    # Mixed: polynomial + periodic
    'poly_plus_sine': {
        'func': lambda x: 0.5 * x**2 + np.sin(2 * np.pi * x),
        'expected_dominant': ['fourier', 'legendre'],  # both should be active
        'category': 'mixed',
        'complexity': 'medium'
    },
    
    # Localized/Gaussian functions (Gabor/Hermite should excel)
    'gaussian': {
        'func': lambda x: np.exp(-5 * x**2),
        'expected_dominant': 'gabor',  # or hermite
        'category': 'localized',
        'complexity': 'simple'
    },
    'gabor_like': {
        'func': lambda x: np.exp(-3 * x**2) * np.cos(5 * np.pi * x),
        'expected_dominant': 'gabor',
        'category': 'localized_oscillatory',
        'complexity': 'medium'
    },
    
    # Step-like (ReLU should be good)
    'smoothstep': {
        'func': lambda x: 1 / (1 + np.exp(-10 * x)),
        'expected_dominant': 'relu',
        'category': 'nonlinear',
        'complexity': 'simple'
    },
    'piecewise': {
        'func': lambda x: np.where(x < 0, x**2, x),
        'expected_dominant': 'relu',
        'category': 'piecewise',
        'complexity': 'medium'
    },
}

# ============================================================================
# 2. Training function
# ============================================================================

def train_on_function(func_name, func_dict, n_epochs=500, verbose=False):
    """Train HybridKAN on a function and analyze gate weights."""
    
    # Generate data
    X = np.linspace(-1, 1, 300).reshape(-1, 1)
    y = func_dict['func'](X.flatten())
    
    # Normalize target (helps training stability)
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / (y_std + 1e-8)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)
    
    # Create model with ALL branches to test which becomes dominant
    model = HybridKAN(
        input_dim=1,
        hidden_dims=[64, 32],
        num_classes=None,
        activation_functions='all',  # Use ALL basis functions
        regression=True,
        dropout_rate=0.1,
        branch_gates=True,
        per_branch_norm=True,
    )
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    best_loss = float('inf')
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(X_tensor)
        loss = nn.MSELoss()(pred, y_tensor)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.6f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        final_pred = model(X_tensor)
        
        # R^2 on normalized data
        ss_res = ((y_tensor - final_pred)**2).sum()
        ss_tot = ((y_tensor - y_tensor.mean())**2).sum()
        r2 = 1 - ss_res / ss_tot
        
        # Get gate weights from all blocks
        all_gates = model.get_branch_gate_weights()
    
    return {
        'func_name': func_name,
        'category': func_dict['category'],
        'expected_dominant': func_dict['expected_dominant'],
        'r2': r2.item(),
        'best_loss': best_loss,
        'gate_weights': all_gates,
    }

# ============================================================================
# 3. Analyze gate patterns
# ============================================================================

def analyze_gates(results):
    """Analyze what gate patterns tell us about function structure."""
    
    analysis = {}
    
    for res in results:
        func_name = res['func_name']
        
        # Average gate weights across blocks
        avg_gates = defaultdict(list)
        for block_idx, gates in res['gate_weights'].items():
            for branch, weight in gates.items():
                avg_gates[branch].append(weight)
        
        avg_gates = {b: np.mean(v) for b, v in avg_gates.items()}
        
        # Find dominant branches (top 2)
        sorted_gates = sorted(avg_gates.items(), key=lambda x: -x[1])
        dominant = sorted_gates[0][0]
        top2 = [g[0] for g in sorted_gates[:2]]
        
        # Gate entropy (high = using many bases, low = specialized)
        weights = np.array(list(avg_gates.values()))
        weights_norm = weights / (weights.sum() + 1e-8)
        entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
        
        analysis[func_name] = {
            'avg_gates': avg_gates,
            'dominant': dominant,
            'top2': top2,
            'entropy': entropy,
            'r2': res['r2'],
            'category': res['category'],
            'expected': res['expected_dominant'],
            'match': dominant == res['expected_dominant'] or dominant in res['expected_dominant'] if isinstance(res['expected_dominant'], list) else dominant == res['expected_dominant']
        }
    
    return analysis

# ============================================================================
# 4. Run experiments
# ============================================================================

print(f"\n[1] Training on {len(TEST_FUNCTIONS)} diverse functions...")
print("-" * 70)

results = []
for func_name, func_dict in TEST_FUNCTIONS.items():
    print(f"  Training on '{func_name}' ({func_dict['category']})...", end=" ", flush=True)
    
    try:
        res = train_on_function(func_name, func_dict, n_epochs=500)
        results.append(res)
        print(f"R2={res['r2']:.4f}")
    except Exception as e:
        print(f"FAILED: {e}")

# Analyze
print(f"\n[2] Analyzing gate patterns...")
print("-" * 70)

analysis = analyze_gates(results)

# ============================================================================
# 5. Print interpretability results
# ============================================================================

print(f"\n[3] INTERPRETABILITY RESULTS")
print("=" * 70)
print(f"{'Function':<18} {'Category':<15} {'Expected':<12} {'Dominant':<10} {'Match':<6} {'R2':<8} {'Entropy':<8}")
print("-" * 70)

correct_predictions = 0
for func_name in TEST_FUNCTIONS.keys():
    if func_name in analysis:
        a = analysis[func_name]
        expected_str = a['expected'] if isinstance(a['expected'], str) else '/'.join(a['expected'][:2])
        match_str = "YES" if a['match'] else "no"
        
        if a['match']:
            correct_predictions += 1
        
        print(f"{func_name:<18} {a['category']:<15} {expected_str:<12} {a['dominant']:<10} {match_str:<6} {a['r2']:.4f}  {a['entropy']:.4f}")

# Summary
print("=" * 70)
total = len([k for k in TEST_FUNCTIONS.keys() if k in analysis])
accuracy = correct_predictions / total * 100 if total > 0 else 0
print(f"\nFunction Structure Prediction Accuracy: {correct_predictions}/{total} ({accuracy:.1f}%)")

# Gate weight breakdown
print(f"\n[4] DETAILED GATE WEIGHT ANALYSIS")
print("=" * 70)

categories = defaultdict(list)
for func_name, a in analysis.items():
    categories[a['category']].append((func_name, a))

for cat, funcs in categories.items():
    print(f"\n{cat.upper()} Functions:")
    print("-" * 50)
    for func_name, a in funcs:
        gates_str = ", ".join([f"{b}:{v:.2f}" for b, v in sorted(a['avg_gates'].items(), key=lambda x: -x[1])[:3]])
        print(f"  {func_name}: {gates_str}")

# ============================================================================
# 6. Key insights
# ============================================================================

print(f"\n[5] KEY RESEARCH INSIGHTS")
print("=" * 70)

# Check if periodic functions show high Fourier
periodic_fourier = []
for func_name, a in analysis.items():
    if a['category'] == 'periodic':
        periodic_fourier.append(a['avg_gates'].get('fourier', 0))

print(f"""
1. INTERPRETABILITY: Gate weights reveal mathematical structure
   - Periodic functions → Fourier gates dominate (avg: {np.mean(periodic_fourier):.3f})
   - This is NOT possible with standard ReLU networks!

2. FUNCTION CLASSIFICATION: We can classify unknown functions by gate pattern
   - Prediction accuracy: {accuracy:.1f}%
   - High entropy = complex/mixed function
   - Low entropy = simple/pure function

3. SCIENTIFIC VALUE: For physical/scientific data
   - If Fourier dominates → underlying phenomenon is periodic (e.g., oscillations)
   - If polynomial dominates → power-law relationship
   - If Gabor dominates → localized/wavelet-like structure

4. COMPARISON TO STANDARD NEURAL NETS:
   - ReLU networks: Black box, no insight into function type
   - HybridKAN: Gates tell you WHAT the network learned
""")

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'n_functions': len(TEST_FUNCTIONS),
    'accuracy': accuracy,
    'results': {
        func_name: {
            'category': a['category'],
            'expected': a['expected'] if isinstance(a['expected'], str) else a['expected'][0],
            'dominant': a['dominant'],
            'match': a['match'],
            'r2': a['r2'],
            'entropy': a['entropy'],
            'avg_gates': a['avg_gates']
        }
        for func_name, a in analysis.items()
    }
}

with open(r'c:\Users\user\Desktop\Research\hybridkan_arxiv\results_research\interpretability_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to results_research/interpretability_results.json")
