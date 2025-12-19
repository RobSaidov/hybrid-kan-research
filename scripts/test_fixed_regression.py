"""
Quick fix test: Skip LayerNorm for 1D input
"""

import sys
sys.path.insert(0, r'c:\Users\user\Desktop\Research\hybridkan_arxiv')

import torch
import torch.nn as nn
import numpy as np

# Manually build a working HybridKAN for regression
from hybridkan.model import HybridKANBlock

torch.manual_seed(42)
np.random.seed(42)

# Generate data
X = np.linspace(-1, 1, 200).reshape(-1, 1)
y = np.sin(2 * np.pi * X.flatten()) + 0.5 * X.flatten()**2

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

class FixedHybridKAN(nn.Module):
    """Fixed version that doesn't kill 1D inputs"""
    
    def __init__(self, input_dim, hidden_dims, branches=['relu', 'fourier']):
        super().__init__()
        
        # For 1D inputs, use a simple linear projection instead of LayerNorm
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Build blocks (starting from hidden_dims[0])
        self.blocks = nn.ModuleList()
        
        # Default configs
        branch_configs = {
            'gabor': {'amp_init': 0.1, 'sigma_init': 1.0, 'freq_init': 1.0, 'gate_init': 0.4},
            'legendre': {'degree': 8, 'gate_init': 0.4},
            'chebyshev': {'degree': 8, 'gate_init': 0.4},
            'hermite': {'degree': 6, 'gate_init': 0.4},
            'fourier': {'n_frequencies': 8, 'gate_init': 0.4},
            'relu': {'gate_init': 0.4}
        }
        start_degrees = {'legendre': 0, 'chebyshev': 2, 'hermite': 2}
        
        prev_dim = hidden_dims[0]
        for i, hidden_dim in enumerate(hidden_dims):
            block = HybridKANBlock(
                in_features=prev_dim,
                out_features=hidden_dim,
                branches=branches,
                branch_configs=branch_configs,
                start_degrees=start_degrees,
                dropout_rate=0.1,
            )
            self.blocks.append(block)
            prev_dim = hidden_dim
        
        # Output head
        self.output_head = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        # Project input to hidden dimension
        x = torch.relu(self.input_proj(x))
        
        # Through blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        return self.output_head(x).squeeze(-1)

# Test
print("=" * 60)
print("Testing Fixed HybridKAN")
print("=" * 60)

model = FixedHybridKAN(1, [64, 32], branches=['relu', 'fourier'])
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# Initial check
model.eval()
with torch.no_grad():
    out = model(X_tensor)
    print(f"\nInitial output std: {out.std().item():.4f}")

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\nTraining...")
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    
    pred = model(X_tensor)
    loss = nn.MSELoss()(pred, y_tensor)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 100 == 0:
        with torch.no_grad():
            ss_res = ((y_tensor - pred)**2).sum()
            ss_tot = ((y_tensor - y_tensor.mean())**2).sum()
            r2 = 1 - ss_res / ss_tot
        print(f"  Epoch {epoch}: loss={loss.item():.4f}, R2={r2.item():.4f}")

# Final
model.eval()
with torch.no_grad():
    final_pred = model(X_tensor)
    ss_res = ((y_tensor - final_pred)**2).sum()
    ss_tot = ((y_tensor - y_tensor.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    
    print(f"\n[Final Results]")
    print(f"  R2: {r2.item():.4f}")
    print(f"  Pred range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
    print(f"  Target range: [{y_tensor.min():.4f}, {y_tensor.max():.4f}]")
    
    # Get gate weights
    print(f"\n[Gate Weights]")
    for i, block in enumerate(model.blocks):
        gates = block.get_gate_weights()
        print(f"  Block {i}: {gates}")
