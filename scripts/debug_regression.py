"""
Debug script to find why HybridKAN regression doesn't learn.
"""

import sys
sys.path.insert(0, r'c:\Users\user\Desktop\Research\hybridkan_arxiv')

import torch
import torch.nn as nn
import numpy as np
from hybridkan.model import HybridKAN

# Simple test function
torch.manual_seed(42)
np.random.seed(42)

# Generate data: y = sin(2*pi*x) + 0.5*x^2
X = np.linspace(-1, 1, 200).reshape(-1, 1)
y = np.sin(2 * np.pi * X.flatten()) + 0.5 * X.flatten()**2

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

print("=" * 60)
print("DEBUG: HybridKAN Regression")
print("=" * 60)

# Create model
model = HybridKAN(
    input_dim=1,
    hidden_dims=[32, 32],
    num_classes=None,
    activation_functions=['relu'],  # Start simple
    regression=True,
)

print(f"\n[Model Info]")
print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"  Regression: {model.regression}")
print(f"  Output head: {model.output_head}")

# Check initial output
model.eval()
with torch.no_grad():
    out = model(X_tensor)
    print(f"\n[Initial Output]")
    print(f"  Shape: {out.shape}")
    print(f"  Mean: {out.mean().item():.4f}")
    print(f"  Std: {out.std().item():.4f}")
    print(f"  Min/Max: {out.min().item():.4f} / {out.max().item():.4f}")
    print(f"  Sample outputs: {out[:5].numpy()}")

# Check forward pass step by step
print("\n[Forward Pass Debug]")
model.train()
x = X_tensor[:5]

# Step 1: Input norm
x1 = model.input_norm(x)
print(f"  After input_norm: mean={x1.mean():.4f}, std={x1.std():.4f}")

# Step 2: Through blocks
x_block = x1
for i, block in enumerate(model.blocks):
    x_before = x_block.clone()
    x_block = block(x_block)
    print(f"  Block {i}: input mean={x_before.mean():.4f}, output mean={x_block.mean():.4f}, output std={x_block.std():.4f}")
    
    # Check gate weights
    gate_weights = block.get_gate_weights()
    print(f"           gates: {gate_weights}")

# Step 3: Output head
out = model.output_head(x_block)
print(f"  Output head: {out.squeeze()[:5]}")

# Check gradients
print("\n[Gradient Check]")
model.zero_grad()
out_full = model(X_tensor)
loss = nn.MSELoss()(out_full, y_tensor)
loss.backward()

# Check which parameters have gradients
grad_info = []
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_norm = p.grad.norm().item()
        if grad_norm > 1e-6:
            grad_info.append((name, grad_norm))

print(f"  Loss: {loss.item():.6f}")
print(f"  Parameters with significant gradients ({len(grad_info)}):")
for name, norm in sorted(grad_info, key=lambda x: -x[1])[:10]:
    print(f"    {name}: {norm:.6f}")

# Try training a few steps
print("\n[Training Test]")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(100):
    model.train()
    optimizer.zero_grad()
    
    pred = model(X_tensor)
    loss = nn.MSELoss()(pred, y_tensor)
    loss.backward()
    
    # Check gradient norms
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    if step % 20 == 0:
        print(f"  Step {step}: loss={loss.item():.6f}, grad_norm={total_grad:.4f}, pred_std={pred.std().item():.4f}")

# Final output
model.eval()
with torch.no_grad():
    final_pred = model(X_tensor)
    final_loss = nn.MSELoss()(final_pred, y_tensor).item()
    
    # R^2
    ss_res = ((y_tensor - final_pred)**2).sum()
    ss_tot = ((y_tensor - y_tensor.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    
    print(f"\n[Final Results]")
    print(f"  Loss: {final_loss:.6f}")
    print(f"  R2: {r2.item():.6f}")
    print(f"  Pred range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
    print(f"  Target range: [{y_tensor.min():.4f}, {y_tensor.max():.4f}]")
