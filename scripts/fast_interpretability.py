"""
FAST EXPERIMENT: Feature Importance & Interpretability

A quick demonstration of HybridKAN's unique interpretability:
Gate weights reveal which mathematical basis the network relies on.

This is a 5-minute experiment with clear, publishable results.

Author: Research Team
"""

import sys
sys.path.insert(0, r'c:\Users\user\Desktop\Research\hybridkan_arxiv')

import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

from hybridkan.model import HybridKAN

print("=" * 70)
print("FAST EXPERIMENT: Interpretability via Gate Weights")
print("=" * 70)

# Define simple test functions where we KNOW what basis should dominate
TESTS = {
    'pure_cosine': {
        'func': lambda x: np.cos(4 * np.pi * x),
        'expected': 'fourier',
        'reason': 'Pure periodic → Fourier'
    },
    'polynomial_cubic': {
        'func': lambda x: x**3 - 0.5*x,
        'expected': 'legendre',
        'reason': 'Polynomial → Legendre/Chebyshev'
    },
    'gaussian_bell': {
        'func': lambda x: np.exp(-8 * x**2),
        'expected': 'gabor',
        'reason': 'Gaussian → Gabor/Hermite'
    },
    'relu_like': {
        'func': lambda x: np.maximum(x, 0) + 0.1*x,
        'expected': 'relu',
        'reason': 'Piecewise linear → ReLU'
    },
}

def train_and_get_gates(func, branches=['relu', 'fourier', 'legendre', 'gabor'], 
                        n_epochs=300, lr=0.01):
    """Train HybridKAN and return final gate weights."""
    
    # Generate data
    X = np.linspace(-1, 1, 200).reshape(-1, 1)
    y = func(X.flatten())
    y = (y - y.mean()) / (y.std() + 1e-8)  # Normalize
    
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    
    model = HybridKAN(
        input_dim=1,
        hidden_dims=[32, 16],
        num_classes=None,
        activation_functions=branches,
        regression=True,
        dropout_rate=0.0,
        branch_gates=True,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(X_t)
        loss = nn.MSELoss()(pred, y_t)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Get final gate weights
    model.eval()
    all_gates = model.get_branch_gate_weights()
    
    # Average across blocks
    avg_gates = {}
    for block_idx, gates in all_gates.items():
        for branch, weight in gates.items():
            if branch not in avg_gates:
                avg_gates[branch] = []
            avg_gates[branch].append(weight)
    
    avg_gates = {b: np.mean(v) for b, v in avg_gates.items()}
    
    # Get R² 
    with torch.no_grad():
        pred = model(X_t)
        ss_res = ((y_t - pred)**2).sum()
        ss_tot = ((y_t - y_t.mean())**2).sum()
        r2 = (1 - ss_res / ss_tot).item()
    
    return avg_gates, r2

# Run tests
print("\n" + "-" * 70)
print(f"{'Function':<20} {'Expected':<10} {'Actual Top':<10} {'Match':<6} {'R2':<8} {'Gate Weights'}")
print("-" * 70)

results = {}
correct = 0

for name, info in TESTS.items():
    gates, r2 = train_and_get_gates(info['func'])
    
    # Find dominant gate
    dominant = max(gates.keys(), key=lambda k: gates[k])
    
    # Check if match (flexible - e.g., legendre or chebyshev for polynomial)
    expected = info['expected']
    match = dominant == expected
    if expected == 'legendre' and dominant in ['legendre', 'chebyshev']:
        match = True
    if expected == 'gabor' and dominant in ['gabor', 'hermite']:
        match = True
    
    if match:
        correct += 1
    
    match_str = "YES" if match else "no"
    
    # Format gate weights
    gates_str = ", ".join([f"{k[:3]}:{v:.2f}" for k, v in sorted(gates.items(), key=lambda x: -x[1])])
    
    print(f"{name:<20} {expected:<10} {dominant:<10} {match_str:<6} {r2:.4f}   {gates_str}")
    
    results[name] = {
        'expected': expected,
        'actual_dominant': dominant,
        'match': match,
        'r2': r2,
        'gates': gates,
        'reason': info['reason']
    }

accuracy = 100 * correct / len(TESTS)
print("-" * 70)
print(f"Prediction Accuracy: {correct}/{len(TESTS)} ({accuracy:.0f}%)")

# Analysis
print(f"""
======================================================================
INTERPRETATION
======================================================================

The gate weights reveal what mathematical structure the network learned:

1. PURE COSINE: Fourier basis should dominate (periodic signal)
2. POLYNOMIAL: Legendre/Chebyshev should dominate (polynomial structure)  
3. GAUSSIAN: Gabor/Hermite should dominate (localized bell curve)
4. RELU-LIKE: ReLU basis should dominate (piecewise linear)

This is INTERPRETABILITY that standard neural networks cannot provide!

APPLICATIONS:
- Scientific discovery: Identify unknown function structure
- Model debugging: Verify network learned expected patterns
- Feature engineering: Use dominant basis as prior knowledge

======================================================================
""")

# Save
with open(r'c:\Users\user\Desktop\Research\hybridkan_arxiv\results_research\interpretability_fast.json', 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'accuracy': accuracy,
        'results': results
    }, f, indent=2)

print("Results saved to results_research/interpretability_fast.json")
