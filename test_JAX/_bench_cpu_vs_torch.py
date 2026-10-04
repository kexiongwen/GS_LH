"""CPU benchmark, single precision: GS_LH_JAX vs GS_LH (PyTorch).

Same simulated data (numpy float32), same method and iteration counts for
both frameworks. PyTorch runs on CPU float32 (a short warm-up run precedes
the timed run, mirroring the JAX driver's internal JIT warm-up, which is
excluded from its timing). Per-step micro-benchmarks compare the
standalone samplers (JAX side: jitted single calls, one sync per call --
the same execution semantics as PyTorch eager).
"""
import os

# Windows: conda's numpy (MKL) and PyTorch each ship a copy of the Intel
# OpenMP runtime; allow the duplicate for this benchmark process only.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH import GS_LH as GS_LH_torch
from GS_LH.beta_sample import beta_sample as t_beta
from GS_LH.shrinkage_sample import shrinkage as t_shr
from GS_LH.sigma2_sample import sigma2_sample as t_s1
from GS_LH_JAX import GS_LH as GS_LH_jax
from GS_LH_JAX.beta_sample import beta_sample as j_beta
from GS_LH_JAX.shrinkage_sample import shrinkage as j_shr
from GS_LH_JAX.sigma2_sample import sigma2_sample as j_s1
from _test_GS_LH import simulate


def bench_driver(n, p, method, total, seed=0):
    """End-to-end driver throughput (it/s) for both frameworks."""
    X, Y, _ = simulate(n, p, {0: 1.5, 1: -1.0}, sigma=1.0, seed=seed)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)   # float32 CPU

    # PyTorch: warm-up (MKL thread pool, allocator), then timed run
    GS_LH_torch(Xt, Yt, n_iter=30, burnin=0, method=method,
                standardize=False, seed=1)
    out_t = GS_LH_torch(Xt, Yt, n_iter=total, burnin=0, method=method,
                        standardize=False, seed=2)
    rate_t = total / out_t["runtime_sec"]

    # JAX: JIT warm-up is internal to GS_LH and excluded from runtime_sec
    out_j = GS_LH_jax(X, Y, n_iter=total, burnin=0, method=method,
                      standardize=False, seed=2)
    rate_j = total / out_j["runtime_sec"]
    return rate_t, rate_j


def bench_steps(n=100, p=4000, reps=200, seed=3):
    """Standalone per-step timing at one (n, p); JAX = jit + one sync/call."""
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, p)).astype(np.float32)
    Y = g.standard_normal(n).astype(np.float32)
    w = np.full(p, 0.02, np.float32)
    w[:20] = 3.0
    a1 = b1 = 1.0
    Xt, Yt = (torch.from_numpy(a) for a in (X, Y))
    wt = torch.from_numpy(w)
    Xj, Yj, wj = (jnp.asarray(a) for a in (X, Y, w))
    sig_t, sig_j = torch.tensor(1.0), jnp.float32(1.0)
    keys = list(jax.random.split(jax.random.PRNGKey(0), reps))

    def t_time(fn):
        for _ in range(10):
            fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    def j_time(fn):
        fn(keys[0]).block_until_ready()
        t0 = time.perf_counter()
        for k in keys:
            fn(k).block_until_ready()
        return (time.perf_counter() - t0) / reps

    rows = []
    rows.append(("S1 sigma2",
                 t_time(lambda: t_s1(Xt, Yt, wt, a1, b1)),
                 j_time(jax.jit(lambda k: j_s1(k, Xj, Yj, wj, a1, b1)))))
    rows.append(("S2 beta",
                 t_time(lambda: t_beta(Xt, Yt, wt, sig_t)),
                 j_time(jax.jit(lambda k: j_beta(k, Xj, Yj, wj, sig_j)))))
    rows.append(("S3-S5 shrinkage",
                 t_time(lambda: t_shr(Yt, 1e-3, 1e-3)),
                 j_time(jax.jit(lambda k: j_shr(k, Yj, 1e-3, 1e-3)))))
    return rows


if __name__ == "__main__":
    print(f"devices: JAX {jax.devices()}, torch CPU "
          f"({torch.get_num_threads()} threads), cpu_count={os.cpu_count()}")

    print("\n=== End-to-end GS_LH driver (CPU, float32) ===")
    print(f"{'regime':22s} | {'method':6s} | {'iters':>5s} | "
          f"{'torch it/s':>10s} | {'JAX it/s':>9s} | {'JAX/torch':>9s}")
    for n, p, method, total in [
            (100, 500, "fast", 4000),
            (100, 4000, "fast", 2000),
            (800, 800, "fast", 400),
            (2000, 500, "direct", 800)]:
        rate_t, rate_j = bench_driver(n, p, method, total)
        print(f"n={n:5d}, p={p:5d}     | {method:6s} | {total:5d} | "
              f"{rate_t:10.0f} | {rate_j:9.0f} | {rate_j / rate_t:8.2f}x")

    print("\n=== Standalone single steps (CPU, float32, n=100, p=4000) ===")
    print(f"{'step':16s} | {'torch ms':>9s} | {'JAX ms':>8s} | {'JAX/torch':>9s}")
    for name, tt, tj in bench_steps():
        print(f"{name:16s} | {tt * 1e3:9.3f} | {tj * 1e3:8.3f} | {tt / tj:8.2f}x")
