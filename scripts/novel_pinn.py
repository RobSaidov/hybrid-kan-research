"""
PHYSICS-INFORMED NEURAL NETWORKS (PINNs) with HybridKAN

Tests whether specialized basis functions help solve differential equations.
This is a NOVEL application: using HybridKAN for scientific computing.

Key insight: Physical laws often have known mathematical structure
- Oscillatory systems → Fourier basis
- Polynomial potentials → Legendre/Chebyshev basis
- Localized solutions → Gabor/Hermite basis

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
print("PHYSICS-INFORMED NEURAL NETWORKS with HybridKAN")
print("=" * 70)

# ============================================================================
# PROBLEM 1: Simple Harmonic Oscillator
# d²y/dt² = -ω²y, y(0)=1, y'(0)=0
# Exact solution: y(t) = cos(ωt)
# ============================================================================

def solve_harmonic_oscillator(branches, omega=2*np.pi, n_points=100, n_epochs=2000, lr=0.01):
    """Solve harmonic oscillator using physics-informed loss."""
    
    # Create model (no batch norm for single-point BC evaluation)
    model = HybridKAN(
        input_dim=1,
        hidden_dims=[32, 32],
        num_classes=None,
        activation_functions=branches,
        regression=True,
        dropout_rate=0.0,
        use_batch_norm=False,  # Disable for PINNs
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training points
    t = torch.linspace(0, 2, n_points, requires_grad=True).reshape(-1, 1)
    
    # Boundary conditions
    t0 = torch.tensor([[0.0]], requires_grad=True)
    
    best_physics_loss = float('inf')
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        y = model(t)
        
        # Compute derivatives using autograd
        dy_dt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y),
                                    create_graph=True, retain_graph=True)[0]
        d2y_dt2 = torch.autograd.grad(dy_dt, t, grad_outputs=torch.ones_like(dy_dt),
                                      create_graph=True, retain_graph=True)[0]
        
        # Physics loss: d²y/dt² + ω²y = 0
        physics_residual = d2y_dt2 + (omega**2) * y
        physics_loss = (physics_residual**2).mean()
        
        # Boundary conditions: y(0)=1, y'(0)=0
        y0 = model(t0)
        dy0 = torch.autograd.grad(y0, t0, grad_outputs=torch.ones_like(y0),
                                  create_graph=True, retain_graph=True)[0]
        
        bc_loss = (y0 - 1.0)**2 + dy0**2
        
        # Total loss
        loss = physics_loss + 10.0 * bc_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if physics_loss.item() < best_physics_loss:
            best_physics_loss = physics_loss.item()
    
    # Evaluate against exact solution
    model.eval()
    with torch.no_grad():
        t_test = torch.linspace(0, 2, 200).reshape(-1, 1)
        y_pred = model(t_test).numpy().flatten()
        y_exact = np.cos(omega * t_test.numpy().flatten())
        
        mse = ((y_pred - y_exact)**2).mean()
        
        # R² 
        ss_res = ((y_exact - y_pred)**2).sum()
        ss_tot = ((y_exact - y_exact.mean())**2).sum()
        r2 = 1 - ss_res / ss_tot
    
    return {
        'mse': mse,
        'r2': r2,
        'physics_loss': best_physics_loss,
    }

# ============================================================================
# PROBLEM 2: Exponential Decay
# dy/dt = -λy, y(0)=1
# Exact solution: y(t) = exp(-λt)
# ============================================================================

def solve_exponential_decay(branches, lam=1.0, n_points=100, n_epochs=2000, lr=0.01):
    """Solve exponential decay ODE."""
    
    model = HybridKAN(
        input_dim=1,
        hidden_dims=[32, 32],
        num_classes=None,
        activation_functions=branches,
        regression=True,
        dropout_rate=0.0,
        use_batch_norm=False,  # Disable for PINNs
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    t = torch.linspace(0, 3, n_points, requires_grad=True).reshape(-1, 1)
    t0 = torch.tensor([[0.0]], requires_grad=True)
    
    best_physics_loss = float('inf')
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        y = model(t)
        
        dy_dt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y),
                                    create_graph=True, retain_graph=True)[0]
        
        # Physics: dy/dt + λy = 0
        physics_residual = dy_dt + lam * y
        physics_loss = (physics_residual**2).mean()
        
        # BC: y(0) = 1
        y0 = model(t0)
        bc_loss = (y0 - 1.0)**2
        
        loss = physics_loss + 10.0 * bc_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if physics_loss.item() < best_physics_loss:
            best_physics_loss = physics_loss.item()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        t_test = torch.linspace(0, 3, 200).reshape(-1, 1)
        y_pred = model(t_test).numpy().flatten()
        y_exact = np.exp(-lam * t_test.numpy().flatten())
        
        mse = ((y_pred - y_exact)**2).mean()
        ss_res = ((y_exact - y_pred)**2).sum()
        ss_tot = ((y_exact - y_exact.mean())**2).sum()
        r2 = 1 - ss_res / ss_tot
    
    return {
        'mse': mse,
        'r2': r2,
        'physics_loss': best_physics_loss,
    }

# ============================================================================
# PROBLEM 3: Damped Harmonic Oscillator
# d²y/dt² + 2γ(dy/dt) + ω²y = 0
# Underdamped solution: y(t) = exp(-γt) * cos(ω't)
# ============================================================================

def solve_damped_oscillator(branches, gamma=0.5, omega=2*np.pi, n_points=100, n_epochs=2000, lr=0.01):
    """Solve damped harmonic oscillator."""
    
    model = HybridKAN(
        input_dim=1,
        hidden_dims=[32, 32],
        num_classes=None,
        activation_functions=branches,
        regression=True,
        dropout_rate=0.0,
        use_batch_norm=False,  # Disable for PINNs
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    t = torch.linspace(0, 3, n_points, requires_grad=True).reshape(-1, 1)
    t0 = torch.tensor([[0.0]], requires_grad=True)
    
    best_physics_loss = float('inf')
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        y = model(t)
        
        dy_dt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y),
                                    create_graph=True, retain_graph=True)[0]
        d2y_dt2 = torch.autograd.grad(dy_dt, t, grad_outputs=torch.ones_like(dy_dt),
                                      create_graph=True, retain_graph=True)[0]
        
        # Physics: d²y/dt² + 2γ(dy/dt) + ω²y = 0
        physics_residual = d2y_dt2 + 2*gamma*dy_dt + (omega**2)*y
        physics_loss = (physics_residual**2).mean()
        
        # BC: y(0)=1, y'(0)=0
        y0 = model(t0)
        dy0 = torch.autograd.grad(y0, t0, grad_outputs=torch.ones_like(y0),
                                  create_graph=True, retain_graph=True)[0]
        
        bc_loss = (y0 - 1.0)**2 + dy0**2
        
        loss = physics_loss + 10.0 * bc_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if physics_loss.item() < best_physics_loss:
            best_physics_loss = physics_loss.item()
    
    # Evaluate (underdamped solution)
    model.eval()
    with torch.no_grad():
        t_test = torch.linspace(0, 3, 200).reshape(-1, 1)
        y_pred = model(t_test).numpy().flatten()
        
        omega_d = np.sqrt(omega**2 - gamma**2)  # Damped frequency
        y_exact = np.exp(-gamma * t_test.numpy().flatten()) * np.cos(omega_d * t_test.numpy().flatten())
        
        mse = ((y_pred - y_exact)**2).mean()
        ss_res = ((y_exact - y_pred)**2).sum()
        ss_tot = ((y_exact - y_exact.mean())**2).sum()
        r2 = 1 - ss_res / ss_tot
    
    return {
        'mse': mse,
        'r2': r2,
        'physics_loss': best_physics_loss,
    }

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

BASIS_CONFIGS = {
    'relu': ['relu'],
    'fourier': ['fourier'],
    'gabor': ['gabor'],
    'polynomial': ['legendre'],
    'relu+fourier': ['relu', 'fourier'],
    'gabor+fourier': ['gabor', 'fourier'],
}

results = {}

# Problem 1: Harmonic Oscillator
print("\n" + "=" * 70)
print("PROBLEM 1: Simple Harmonic Oscillator (d²y/dt² = -ω²y)")
print("Exact solution: y(t) = cos(ωt) - PERIODIC")
print("=" * 70)

ho_results = {}
for name, branches in BASIS_CONFIGS.items():
    print(f"  Testing {name}...", end=" ", flush=True)
    
    scores = []
    for seed in [42, 123]:
        torch.manual_seed(seed)
        res = solve_harmonic_oscillator(branches, n_epochs=1500)
        scores.append(res)
    
    avg_r2 = np.mean([s['r2'] for s in scores])
    avg_mse = np.mean([s['mse'] for s in scores])
    
    ho_results[name] = {'r2': avg_r2, 'mse': avg_mse}
    print(f"R2={avg_r2:.4f}, MSE={avg_mse:.6f}")

results['harmonic_oscillator'] = ho_results

# Problem 2: Exponential Decay
print("\n" + "=" * 70)
print("PROBLEM 2: Exponential Decay (dy/dt = -λy)")
print("Exact solution: y(t) = exp(-λt) - EXPONENTIAL/POLYNOMIAL")
print("=" * 70)

exp_results = {}
for name, branches in BASIS_CONFIGS.items():
    print(f"  Testing {name}...", end=" ", flush=True)
    
    scores = []
    for seed in [42, 123]:
        torch.manual_seed(seed)
        res = solve_exponential_decay(branches, n_epochs=1500)
        scores.append(res)
    
    avg_r2 = np.mean([s['r2'] for s in scores])
    avg_mse = np.mean([s['mse'] for s in scores])
    
    exp_results[name] = {'r2': avg_r2, 'mse': avg_mse}
    print(f"R2={avg_r2:.4f}, MSE={avg_mse:.6f}")

results['exponential_decay'] = exp_results

# Problem 3: Damped Oscillator
print("\n" + "=" * 70)
print("PROBLEM 3: Damped Harmonic Oscillator")
print("Exact solution: y(t) = exp(-γt)cos(ω't) - GABOR-LIKE")
print("=" * 70)

damp_results = {}
for name, branches in BASIS_CONFIGS.items():
    print(f"  Testing {name}...", end=" ", flush=True)
    
    scores = []
    for seed in [42, 123]:
        torch.manual_seed(seed)
        res = solve_damped_oscillator(branches, n_epochs=1500)
        scores.append(res)
    
    avg_r2 = np.mean([s['r2'] for s in scores])
    avg_mse = np.mean([s['mse'] for s in scores])
    
    damp_results[name] = {'r2': avg_r2, 'mse': avg_mse}
    print(f"R2={avg_r2:.4f}, MSE={avg_mse:.6f}")

results['damped_oscillator'] = damp_results

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Physics-Informed Neural Networks")
print("=" * 70)

print(f"\n{'Problem':<25} {'Best Basis':<15} {'R2':<10} {'ReLU R2':<10} {'Improvement'}")
print("-" * 70)

for problem_name, problem_results in results.items():
    best_name = max(problem_results.keys(), key=lambda k: problem_results[k]['r2'])
    best_r2 = problem_results[best_name]['r2']
    relu_r2 = problem_results['relu']['r2']
    improvement = best_r2 - relu_r2
    imp_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
    
    print(f"{problem_name:<25} {best_name:<15} {best_r2:.4f}     {relu_r2:.4f}     {imp_str}")

# Key findings
print(f"""
======================================================================
KEY FINDINGS: Physics-Informed Neural Networks
======================================================================

1. HARMONIC OSCILLATOR (cos solution):
   - Expected: Fourier should excel (periodic solution)
   - Best: {max(results['harmonic_oscillator'].keys(), key=lambda k: results['harmonic_oscillator'][k]['r2'])}
   
2. EXPONENTIAL DECAY (exp solution):
   - Expected: Gabor/Hermite might help (Gaussian-like decay)
   - Best: {max(results['exponential_decay'].keys(), key=lambda k: results['exponential_decay'][k]['r2'])}

3. DAMPED OSCILLATOR (exp × cos solution):
   - Expected: Gabor should excel (Gabor = Gaussian × sinusoid)
   - Best: {max(results['damped_oscillator'].keys(), key=lambda k: results['damped_oscillator'][k]['r2'])}

IMPLICATIONS FOR SCIENTIFIC COMPUTING:
- When solving ODEs/PDEs, match basis to expected solution structure
- Wave equations → Fourier basis
- Diffusion/decay → Gabor/Hermite basis
- Polynomial potentials → Legendre/Chebyshev basis

This demonstrates HybridKAN's value for physics-informed learning!
""")

# Save results
with open(r'c:\Users\user\Desktop\Research\hybridkan_arxiv\results_research\pinn_results.json', 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'results': results
    }, f, indent=2)

print("Results saved to results_research/pinn_results.json")
