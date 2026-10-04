"""GPU benchmarks for beta_sample (RTX 3060 Ti)."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH.beta_sample import beta_sample
from GS_LH.shrinkage_sample import shrinkage

dev = torch.device("cuda")


def bench(fn, args, reps):
    fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def make_dense(n, p, dtype, s=20, seed=3):
    torch.manual_seed(seed)
    X = torch.randn(n, p, dtype=dtype, device=dev)
    Y = torch.randn(n, dtype=dtype, device=dev)
    w = torch.full((p,), 0.02, dtype=dtype, device=dev)
    w[:s] = 3.0
    sigma = torch.tensor(1.0, dtype=dtype, device=dev)
    return X, Y, w, sigma


# --- 0) smoke: shrinkage module on cuda ---
param = torch.randn(1000, device=dev)
w0 = shrinkage(param, 1e-3, 1e-3)
print("shrinkage on cuda, all finite:", torch.isfinite(w0).all().item())

# --- 1) quick correctness sanity on GPU (float64) ---
torch.manual_seed(0)
n, p, N = 30, 200, 20000
X, Y, w, sigma = make_dense(n, p, torch.float64)
A = (X.T @ X + torch.diag(w.pow(-2))).cpu()
mu = torch.linalg.solve(A, (X.T @ Y).cpu())
sd = torch.linalg.inv(A).diagonal().sqrt()
s1 = torch.zeros(p, dtype=torch.float64, device=dev)
for _ in range(N):
    s1 += beta_sample(X, Y, w, sigma)
z = ((s1 / N).cpu() - mu).abs().max().item()
print(f"GPU float64 beta_sample: max |mean - mu| = {z:.2e} (MC scale ~{3e-3:.0e})")

# --- 2) dense benchmarks, float64 vs float32 ---
print("\nDense timing per draw (sparse w, s=20):")
for dtype in [torch.float64, torch.float32]:
    for n, p, reps in [(100, 4000, 20), (2000, 2000, 5), (4000, 4000, 3)]:
        X, Y, w, sigma = make_dense(n, p, dtype)
        t1 = bench(beta_sample, (X, Y, w, sigma), reps)
        print(f"  {str(dtype)[6:]:8s} n={n:5d}, p={p:5d}: "
              f"Alg1 {t1 * 1000:8.2f} ms")
