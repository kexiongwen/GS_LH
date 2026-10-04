"""Diagnose the sigma^2 collapse observed in Test 1 (collapsed S1 update)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH.beta_sample import beta_sample
from GS_LH.shrinkage_sample import shrinkage
from GS_LH.sigma2_sample import sigma2_sample

dtype = torch.float64
torch.manual_seed(1)
n, p = 100, 500
X = torch.randn(n, p, dtype=dtype)
X = X * (n ** 0.5 / X.norm(dim=0))
bt = torch.zeros(p, dtype=dtype)
bt[:5] = torch.tensor([2.0, -1.5, 1.2, 0.9, -0.7], dtype=dtype)
Y = X @ bt + 1.0 * torch.randn(n, dtype=dtype)

torch.manual_seed(42)
w = torch.ones(p, dtype=dtype)
print("iter |   sigma2 | med(w) |   max(w) | sum sqrt|beta~|")
for it in range(3001):
    sigma2, L = sigma2_sample(X, Y, w, 1e-3, 1e-3, return_L=True)
    sigma = sigma2.sqrt()
    beta = beta_sample(X, Y, w, sigma, L=L)
    bt_std = beta / sigma
    w = shrinkage(bt_std, 1e-3, 1e-3)
    if it % 300 == 0 or it == 2999:
        print(f"{it:5d} | {sigma2.item():8.4f} | {w.median():6.2f} | "
              f"{w.max():9.2f} | {bt_std.abs().sqrt().sum():8.2f}")

# reference: residual-based check of where sigma2 'should' be
resid = Y - X @ bt
print(f"\ntrue-model residual variance: {resid.var():.3f}")
