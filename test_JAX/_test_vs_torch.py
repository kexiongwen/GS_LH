"""Cross-check GS_LH_JAX (float32) against the reference PyTorch
implementation GS_LH (float64): same data, same settings; the posterior
means of the two independent chains must agree within Monte Carlo error.
"""
import os

# Windows: conda's numpy (MKL) and PyTorch each ship a copy of the Intel
# OpenMP runtime (libiomp5md.dll); loading both in one process aborts with
# OMP Error #15. Allow the duplicate for this cross-check process only
# (must be set before numpy/torch are imported).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH import GS_LH as GS_LH_torch
from GS_LH_JAX import GS_LH as GS_LH_jax
from _test_GS_LH import simulate


def cross_check(method):
    n, p = 100, 500
    signals = {0: 2.0, 1: -1.5, 2: 1.2, 3: 0.9, 4: -0.7}
    X, Y, bt = simulate(n, p, signals, sigma=1.0, seed=1)

    out_j = GS_LH_jax(X, Y, n_iter=4000, burnin=1000, method=method,
                      standardize=False, seed=42)
    out_t = GS_LH_torch(torch.tensor(X, dtype=torch.float64),
                        torch.tensor(Y, dtype=torch.float64),
                        n_iter=4000, burnin=1000, method=method,
                        standardize=False, seed=42)

    bm_j = out_j["beta"].mean(0).astype(np.float64)
    bm_t = out_t["beta"].mean(0).numpy()
    s2_j, s2_t = float(out_j["sigma2"].mean()), float(out_t["sigma2"].mean())

    d_all = np.abs(bm_j - bm_t).max()
    d_sig = max(abs(bm_j[j] - bm_t[j]) for j in signals)
    s2_rel = s2_j / s2_t - 1
    print(f"  method={method!r}: max |mean diff| = {d_all:.4f} "
          f"(signals {d_sig:.4f}), sigma2 JAX {s2_j:.3f} vs torch {s2_t:.3f} "
          f"({s2_rel:+.1%})")
    ok = d_all < 0.10 and abs(s2_rel) < 0.25
    print("  PASS" if ok else "  FAIL")


if __name__ == "__main__":
    print("=== JAX (float32) vs PyTorch (float64) cross-check ===")
    cross_check("fast")
    cross_check("direct")
