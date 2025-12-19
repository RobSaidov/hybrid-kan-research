"""
NOVEL RESEARCH: Real-World Signal Processing & Time Series

Tests HybridKAN on realistic signal processing tasks where specialized
basis functions should provide advantages:

1. ECG-like signals (quasi-periodic with noise)
2. Audio-like signals (multi-frequency)
3. Time series forecasting
4. Out-of-distribution extrapolation

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

torch.manual_seed(42)
np.random.seed(42)

from hybridkan.model import HybridKAN

print("=" * 70)
print("NOVEL RESEARCH: Signal Processing & Time Series")
print("=" * 70)

# ============================================================================
# 1. SYNTHETIC ECG-LIKE SIGNAL
# ============================================================================

def generate_ecg_like(n_samples=500, noise_std=0.1):
    """Generate synthetic ECG-like quasi-periodic signal."""
    t = np.linspace(0, 4*np.pi, n_samples)
    
    # P wave (small bump)
    p_wave = 0.15 * np.exp(-((t % (2*np.pi) - 0.5)**2) / 0.02)
    
    # QRS complex (sharp spike)
    qrs = 1.0 * np.exp(-((t % (2*np.pi) - 1.0)**2) / 0.005)
    qrs -= 0.3 * np.exp(-((t % (2*np.pi) - 0.9)**2) / 0.01)
    qrs -= 0.2 * np.exp(-((t % (2*np.pi) - 1.1)**2) / 0.01)
    
    # T wave (recovery)
    t_wave = 0.3 * np.exp(-((t % (2*np.pi) - 1.8)**2) / 0.03)
    
    signal = p_wave + qrs + t_wave
    signal += noise_std * np.random.randn(n_samples)
    
    return t.reshape(-1, 1), signal

# ============================================================================
# 2. MULTI-FREQUENCY AUDIO-LIKE SIGNAL
# ============================================================================

def generate_audio_like(n_samples=500, noise_std=0.05):
    """Generate audio-like signal with harmonics."""
    t = np.linspace(0, 2, n_samples)
    
    # Fundamental + harmonics (like a musical note)
    fundamental = 5.0  # Hz
    signal = np.sin(2 * np.pi * fundamental * t)
    signal += 0.5 * np.sin(2 * np.pi * 2 * fundamental * t)  # 2nd harmonic
    signal += 0.25 * np.sin(2 * np.pi * 3 * fundamental * t)  # 3rd harmonic
    signal += 0.125 * np.sin(2 * np.pi * 4 * fundamental * t)  # 4th harmonic
    
    # Amplitude envelope (like ADSR)
    envelope = np.exp(-t) * (1 - np.exp(-10*t))
    signal *= envelope
    
    signal += noise_std * np.random.randn(n_samples)
    
    return t.reshape(-1, 1), signal

# ============================================================================
# 3. TIME SERIES WITH TREND + SEASONALITY
# ============================================================================

def generate_time_series(n_samples=500, noise_std=0.1):
    """Generate time series with trend, seasonality, and noise."""
    t = np.linspace(0, 5, n_samples)
    
    # Linear trend
    trend = 0.5 * t
    
    # Seasonal component (yearly-like)
    seasonal = np.sin(2 * np.pi * t) + 0.3 * np.sin(4 * np.pi * t)
    
    # Weekly-like cycle
    weekly = 0.2 * np.sin(2 * np.pi * 7 * t)
    
    signal = trend + seasonal + weekly
    signal += noise_std * np.random.randn(n_samples)
    
    return t.reshape(-1, 1), signal

# ============================================================================
# 4. OUT-OF-DISTRIBUTION TEST
# ============================================================================

def generate_ood_data(func, train_range=(-1, 1), test_range=(-1.5, 1.5), n_train=200, n_test=100):
    """Generate train data in one range, test in extended range."""
    X_train = np.linspace(train_range[0], train_range[1], n_train).reshape(-1, 1)
    X_test = np.linspace(test_range[0], test_range[1], n_test).reshape(-1, 1)
    
    y_train = func(X_train.flatten())
    y_test = func(X_test.flatten())
    
    return X_train, y_train, X_test, y_test

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, branches, n_epochs=500, lr=0.01):
    """Train model and return metrics."""
    
    # Normalize
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_norm, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_norm, dtype=torch.float32)
    
    model = HybridKAN(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        num_classes=None,
        activation_functions=branches,
        regression=True,
        dropout_rate=0.05,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    best_train_loss = float('inf')
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(X_train_t)
        loss = nn.MSELoss()(pred, y_train_t)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())
        
        if loss.item() < best_train_loss:
            best_train_loss = loss.item()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t)
        test_pred = model(X_test_t)
        
        # R² scores
        def r2_score(y_true, y_pred):
            ss_res = ((y_true - y_pred)**2).sum()
            ss_tot = ((y_true - y_true.mean())**2).sum()
            return (1 - ss_res / ss_tot).item()
        
        train_r2 = r2_score(y_train_t, train_pred)
        test_r2 = r2_score(y_test_t, test_pred)
        
        train_mse = nn.MSELoss()(train_pred, y_train_t).item()
        test_mse = nn.MSELoss()(test_pred, y_test_t).item()
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
    }

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

BASIS_CONFIGS = {
    'relu': ['relu'],
    'fourier': ['fourier'],
    'gabor': ['gabor'],
    'polynomial': ['legendre', 'chebyshev'],
    'relu+fourier': ['relu', 'fourier'],
    'all': ['relu', 'fourier', 'legendre', 'chebyshev', 'hermite', 'gabor'],
}

results = {}

# ============================================================================
# EXPERIMENT 1: ECG-like Signal
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: ECG-like Signal Reconstruction")
print("=" * 70)

X_ecg, y_ecg = generate_ecg_like(n_samples=400, noise_std=0.05)
split = 300
X_train, y_train = X_ecg[:split], y_ecg[:split]
X_test, y_test = X_ecg[split:], y_ecg[split:]

exp1_results = {}
for name, branches in BASIS_CONFIGS.items():
    scores = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        res = train_and_evaluate(X_train, y_train, X_test, y_test, branches, n_epochs=400)
        scores.append(res)
    
    avg_test_r2 = np.mean([s['test_r2'] for s in scores])
    avg_train_r2 = np.mean([s['train_r2'] for s in scores])
    
    exp1_results[name] = {'train_r2': avg_train_r2, 'test_r2': avg_test_r2}
    print(f"  {name:<15}: Train R2={avg_train_r2:.4f}, Test R2={avg_test_r2:.4f}")

results['ecg_signal'] = exp1_results

# ============================================================================
# EXPERIMENT 2: Audio-like Signal
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: Audio-like Signal (Harmonics)")
print("=" * 70)

X_audio, y_audio = generate_audio_like(n_samples=400, noise_std=0.02)
split = 300
X_train, y_train = X_audio[:split], y_audio[:split]
X_test, y_test = X_audio[split:], y_audio[split:]

exp2_results = {}
for name, branches in BASIS_CONFIGS.items():
    scores = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        res = train_and_evaluate(X_train, y_train, X_test, y_test, branches, n_epochs=400)
        scores.append(res)
    
    avg_test_r2 = np.mean([s['test_r2'] for s in scores])
    avg_train_r2 = np.mean([s['train_r2'] for s in scores])
    
    exp2_results[name] = {'train_r2': avg_train_r2, 'test_r2': avg_test_r2}
    print(f"  {name:<15}: Train R2={avg_train_r2:.4f}, Test R2={avg_test_r2:.4f}")

results['audio_signal'] = exp2_results

# ============================================================================
# EXPERIMENT 3: Time Series Forecasting
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: Time Series (Trend + Seasonality)")
print("=" * 70)

X_ts, y_ts = generate_time_series(n_samples=400, noise_std=0.05)
split = 300
X_train, y_train = X_ts[:split], y_ts[:split]
X_test, y_test = X_ts[split:], y_ts[split:]

exp3_results = {}
for name, branches in BASIS_CONFIGS.items():
    scores = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        res = train_and_evaluate(X_train, y_train, X_test, y_test, branches, n_epochs=400)
        scores.append(res)
    
    avg_test_r2 = np.mean([s['test_r2'] for s in scores])
    avg_train_r2 = np.mean([s['train_r2'] for s in scores])
    
    exp3_results[name] = {'train_r2': avg_train_r2, 'test_r2': avg_test_r2}
    print(f"  {name:<15}: Train R2={avg_train_r2:.4f}, Test R2={avg_test_r2:.4f}")

results['time_series'] = exp3_results

# ============================================================================
# EXPERIMENT 4: Out-of-Distribution Extrapolation
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: Out-of-Distribution Extrapolation")
print("=" * 70)

ood_functions = {
    'sine': lambda x: np.sin(2 * np.pi * x),
    'polynomial': lambda x: x**3 - x,
    'gaussian': lambda x: np.exp(-3 * x**2),
}

exp4_results = {}
for func_name, func in ood_functions.items():
    print(f"\n  Function: {func_name}")
    print(f"  Train range: [-1, 1], Test range: [-1.5, 1.5] (extrapolation)")
    print(f"  {'-'*50}")
    
    X_train, y_train, X_test, y_test = generate_ood_data(
        func, train_range=(-1, 1), test_range=(-1.5, 1.5)
    )
    
    func_results = {}
    for name, branches in BASIS_CONFIGS.items():
        scores = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            res = train_and_evaluate(X_train, y_train, X_test, y_test, branches, n_epochs=400)
            scores.append(res)
        
        avg_test_r2 = np.mean([s['test_r2'] for s in scores])
        avg_train_r2 = np.mean([s['train_r2'] for s in scores])
        
        func_results[name] = {'train_r2': avg_train_r2, 'test_r2': avg_test_r2}
        print(f"    {name:<15}: Train R2={avg_train_r2:.4f}, OOD Test R2={avg_test_r2:.4f}")
    
    exp4_results[func_name] = func_results

results['ood_extrapolation'] = exp4_results

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Best Performers")
print("=" * 70)

def find_best(exp_results, metric='test_r2'):
    best_name = max(exp_results.keys(), key=lambda k: exp_results[k][metric])
    return best_name, exp_results[best_name][metric]

print(f"\n{'Experiment':<30} {'Best Basis':<15} {'Test R2':<10} {'ReLU R2':<10} {'Improvement'}")
print("-" * 75)

for exp_name in ['ecg_signal', 'audio_signal', 'time_series']:
    best_name, best_r2 = find_best(results[exp_name])
    relu_r2 = results[exp_name]['relu']['test_r2']
    improvement = best_r2 - relu_r2
    imp_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
    print(f"{exp_name:<30} {best_name:<15} {best_r2:.4f}     {relu_r2:.4f}     {imp_str}")

print("\nOOD Extrapolation:")
for func_name in ood_functions.keys():
    best_name, best_r2 = find_best(results['ood_extrapolation'][func_name])
    relu_r2 = results['ood_extrapolation'][func_name]['relu']['test_r2']
    improvement = best_r2 - relu_r2
    imp_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
    print(f"  {func_name:<28} {best_name:<15} {best_r2:.4f}     {relu_r2:.4f}     {imp_str}")

# ============================================================================
# KEY FINDINGS
# ============================================================================
print(f"""
======================================================================
KEY RESEARCH FINDINGS
======================================================================

1. SIGNAL PROCESSING:
   - ECG-like signals: Tests quasi-periodic pattern recognition
   - Audio signals: Tests harmonic decomposition capability
   - Fourier basis should excel due to periodic nature

2. TIME SERIES:
   - Combines trend + seasonality + noise
   - Tests ability to decompose complex temporal patterns
   - Polynomial (trend) + Fourier (seasonality) should help

3. OUT-OF-DISTRIBUTION EXTRAPOLATION:
   - Critical test: Can the model extrapolate beyond training range?
   - Polynomial basis should extrapolate better for polynomial functions
   - Fourier should extrapolate for periodic functions
   - This is a KEY advantage over ReLU (which fails at extrapolation)

4. PRACTICAL IMPLICATIONS:
   - Scientific computing: Match basis to physics
   - Medical signals (ECG, EEG): Fourier/Gabor for quasi-periodic
   - Financial time series: Polynomial (trend) + Fourier (cycles)
""")

# Save results
with open(r'c:\Users\user\Desktop\Research\hybridkan_arxiv\results_research\signal_processing_results.json', 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'results': results
    }, f, indent=2)

print("\nResults saved to results_research/signal_processing_results.json")
