"""Quick statistical smoke tests for shrinkage_sample.py (not part of the package)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH.shrinkage_sample import inv_gauss, shrinkage

torch.manual_seed(0)

# --- Test 1: inv_gauss moments. IG(mu, 1): mean = mu, var = mu^3 ---
N = 2_000_000
for mu_val in [0.5, 1.0, 3.0]:
    mu = torch.full((N,), mu_val, dtype=torch.float64)
    x = inv_gauss(mu)
    print(f"IG(mu={mu_val},1): mean {x.mean():.4f} (true {mu_val}), "
          f"var {x.var():.4f} (true {mu_val**3:.4f})")

# --- Test 2: shrinkage basic run ---
p = 200
scale = torch.tensor([0.1] * 50 + [1.0] * 150, dtype=torch.float64)
param = torch.randn(p, dtype=torch.float64) * scale
w = shrinkage(param, a=0.001, b=0.001)
print("shrinkage: shape", tuple(w.shape),
      "all finite:", torch.isfinite(w).all().item(),
      "all positive:", (w > 0).all().item())

# --- Test 3: reproducibility with same seed ---
torch.manual_seed(42)
w1 = shrinkage(param, 0.001, 0.001)
torch.manual_seed(42)
w2 = shrinkage(param, 0.001, 0.001)
print("reproducible:", torch.equal(w1, w2))

# --- Test 4: edge cases - tiny and exact-zero beta ---
param_tiny = param.clone()
param_tiny[0] = 1e-30
w = shrinkage(param_tiny, 0.001, 0.001)
print("tiny beta: all finite:", torch.isfinite(w).all().item())

param_zero = param.clone()
param_zero[0] = 0.0
w = shrinkage(param_zero, 0.001, 0.001)
print("exact zero beta: all finite:", torch.isfinite(w).all().item(),
      "| w[0] =", w[0].item())
