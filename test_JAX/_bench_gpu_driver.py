"""GPU driver benchmark, single precision: GS_LH_JAX on the GPU backend.

Driver-level end-to-end throughput (Gibbs iterations kept per second),
float32, mirroring the methodology of _bench_cpu_vs_torch.py (the JAX
driver's internal AOT compilation is excluded from runtime_sec).  Each
regime is timed twice: the plain chunked-scan runners, and graph=True
(XLA command buffers: each chunk's scan While loop recorded as one CUDA
graph).  Run where jax[cuda12] is installed, e.g. under WSL2/Linux; on
a partly occupied GPU export XLA_PYTHON_CLIENT_PREALLOCATE=false first.
Timing-only data (plain Gaussian X/Y, columns scaled to ||X_j||^2 = n,
standardize=False).
"""
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH_JAX import GS_LH as GS_LH_jax

print(f"devices: {jax.devices()}, jax {jax.__version__}, float32")


def make(n, p, seed=0):
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, p)).astype(np.float32)
    X *= n ** 0.5 / np.linalg.norm(X, axis=0)   # ||X_j||^2 = n
    return X, g.standard_normal(n).astype(np.float32)


REGIMES = [
    (100, 500, "fast", 4000),
    (100, 4000, "fast", 2000),
    (800, 800, "fast", 400),
    (2000, 500, "direct", 800),
    (8000, 500, "direct", 300),
]

print(f"{'regime':22s} | {'method':6s} | {'iters':>5s} | "
      f"{'plain it/s':>10s} | {'graph it/s':>10s} | {'speedup':>7s}")
for n, p, method, total in REGIMES:
    X, Y = make(n, p, seed=0)
    rates = {}
    for mode, gkw in [("plain", {}), ("graph", {"graph": True})]:
        out = GS_LH_jax(X, Y, n_iter=total, burnin=0, method=method,
                        standardize=False, seed=2, **gkw)
        rates[mode] = total / out["runtime_sec"]
    print(f"n={n:5d}, p={p:5d}     | {method:6s} | {total:5d} | "
          f"{rates['plain']:10.0f} | {rates['graph']:10.0f} | "
          f"{rates['graph'] / rates['plain'] - 1:6.0%}")
