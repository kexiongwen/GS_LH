"""Hyperprior matrix experiment for the sigma^2 collapse (Test 1 setting)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH import GS_LH
from _test_GS_LH import simulate

n, p = 100, 500
signals = {0: 2.0, 1: -1.5, 2: 1.2, 3: 0.9, 4: -0.7}
X, Y, bt = simulate(n, p, signals, sigma=1.0, seed=1)

print(f"{'a1,b1 (sigma2)':>18s} | {'a,b (lambda)':>14s} | sigma2 mean | "
      f"max sig err | noise max")
for a1, b1 in [(1e-3, 1e-3), (1.0, 1.0), (1.0, 0.5)]:
    for a, b in [(1e-3, 1e-3), (1.0, 1.0), (5.0, 1.0)]:
        out = GS_LH(X, Y, n_iter=4000, burnin=1000, method="fast",
                    standardize=False,
                    a1=a1, b1=b1, a=a, b=b, seed=42)
        s2 = out["sigma2"].mean().item()
        bm = out["beta"].mean(0)
        sig_err = max(abs(bm[j].item() - v) for j, v in signals.items())
        print(f"{str((a1, b1)):>18s} | {str((a, b)):>14s} | {s2:10.3f} | "
              f"{sig_err:11.3f} | {bm[5:].abs().max():.3f}")
