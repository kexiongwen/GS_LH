"""GPU driver benchmark, single precision: GS_LH (PyTorch) on CUDA.

Driver-level end-to-end throughput (Gibbs iterations kept per second),
float32, mirroring the methodology of test_JAX/_bench_cpu_vs_torch.py:
a 30-iteration warm-up run precedes (and is excluded from) the timed
run. Timing-only data (plain Gaussian X/Y, columns scaled to
||X_j||^2 = n, standardize=False) -- values do not affect per-iteration
cost. Data is generated with torch only (no numpy import, hence no
OpenMP-runtime clash with MKL).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH import GS_LH as GS_LH_torch

assert torch.cuda.is_available(), "CUDA device required"
print(f"device: {torch.cuda.get_device_name(0)}, "
      f"torch {torch.__version__}, float32")


def make(n, p, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, p, generator=g)          # float32
    X = X * (n ** 0.5 / X.norm(dim=0))          # ||X_j||^2 = n
    return X.cuda(), torch.randn(n, generator=g).cuda()


REGIMES = [
    (100, 500, "fast", 4000),
    (100, 4000, "fast", 2000),
    (800, 800, "fast", 400),
    (2000, 500, "direct", 800),
    (8000, 500, "direct", 300),
]

print(f"{'regime':22s} | {'method':6s} | {'iters':>5s} | {'torch it/s':>10s}")
for n, p, method, total in REGIMES:
    X, Y = make(n, p, seed=0)
    # warm-up (context/allocator/JIT-cuDNN paths), excluded from timing
    GS_LH_torch(X, Y, n_iter=30, burnin=0, method=method,
                standardize=False, seed=1)
    out = GS_LH_torch(X, Y, n_iter=total, burnin=0, method=method,
                      standardize=False, seed=2)
    print(f"n={n:5d}, p={p:5d}     | {method:6s} | {total:5d} | "
          f"{total / out['runtime_sec']:10.0f}")
