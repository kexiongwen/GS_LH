"""Validation of graph=True mode (XLA command buffers) for GS_LH_JAX.

graph=True compiles the chunk runners with XLA command buffers: each
chunk's whole scan While loop is recorded as ONE CUDA graph on first
execution and replayed afterwards.  The same compiled kernels are
replayed, so the chain is BITWISE-IDENTICAL to the plain runners for a
fixed seed -- graph mode changes only how the chunk is launched, not the
math.  Unlike the PyTorch version, graph mode is not gated by the
chi-square decision (exact Gamma is capturable in JAX).

Sections needing a GPU backend print a skip note without one (e.g.
native Windows); run under WSL2 with jax[cuda12] to exercise them.
"""
import importlib
import os
import sys
import time
import warnings

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH_JAX import GS_LH

# the package __init__ re-exports the GS_LH FUNCTION, which shadows the
# submodule attribute: monkeypatching needs the real module (see README).
drv = importlib.import_module("GS_LH_JAX.GS_LH")

on_gpu = jax.default_backend() == "gpu"
results = []


def check(tag, ok):
    results.append((tag, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}: {tag}")


def simulate(n, p, signals, sigma=1.0, seed=0):
    """X columns standardized to ||X_j||^2 = n; sparse true beta."""
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, p)).astype(np.float32)
    X *= (n ** 0.5 / np.linalg.norm(X, axis=0))
    beta_true = np.zeros(p, dtype=np.float32)
    for j, v in signals.items():
        beta_true[j] = v
    Y = (X @ beta_true + sigma * g.standard_normal(n)).astype(np.float32)
    return X, Y, beta_true


def same_run(o1, o2):
    return (np.array_equal(o1["beta"], o2["beta"])
            and np.array_equal(o1["sigma2"], o2["sigma2"])
            and np.array_equal(o1["w_final"], o2["w_final"]))


print("=== 1. non-GPU: graph=True -> warn + plain runners (bitwise) ===")
if on_gpu:
    print("  skipped (GPU backend; see sections 2-5)")
else:
    Xc, Yc, _ = simulate(60, 80, {0: 1.5}, sigma=1.0, seed=3)
    kw = dict(n_iter=60, burnin=10, method="fast", standardize=False,
              seed=5)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        out_g = GS_LH(Xc, Yc, graph=True, **kw)
    hit = any("GPU backend" in str(w.message) for w in ws)
    out_e = GS_LH(Xc, Yc, **kw)
    check("RuntimeWarning mentions the GPU backend", hit)
    check("fallback chain bitwise identical to plain", same_run(out_g, out_e))

print("=== 2. GPU: graph == plain BITWISE (both Gamma paths) ===")
if not on_gpu:
    print(f"  skipped (backend {jax.default_backend()}); run under WSL2 "
          "with jax[cuda12] to exercise them")
else:
    X, Y, _ = simulate(100, 500, {0: 2.0, 1: -1.5, 2: 1.2}, sigma=1.0,
                       seed=4)
    kw = dict(n_iter=200, burnin=50, method="fast", standardize=False,
              seed=42)
    # (a) chi-square engaged (accepted by _accept_chi2 on GPU)
    plain_c = GS_LH(X, Y, **kw)
    graph_c = GS_LH(X, Y, graph=True, **kw)
    check("graph == plain bitwise (chi-square engaged)",
          same_run(plain_c, graph_c))
    # (b) exact Gamma (forced refusal): graph must NOT be gated by chi2
    orig = drv._accept_chi2
    drv._accept_chi2 = lambda *a, **k: False
    try:
        plain_e = GS_LH(X, Y, **kw)
        graph_e = GS_LH(X, Y, graph=True, **kw)
    finally:
        drv._accept_chi2 = orig
    check("graph == plain bitwise (exact Gamma; graph not gated by chi2)",
          same_run(plain_e, graph_e))

    print("=== 3. GPU: reproducibility + chunk invariance under graph ===")
    g1 = GS_LH(X, Y, graph=True, **kw)
    g2 = GS_LH(X, Y, graph=True, **kw)
    check("two graph runs, same seed: bitwise identical", same_run(g1, g2))
    drv._CHUNK = 13
    try:
        g3 = GS_LH(X, Y, graph=True, **kw)
    finally:
        drv._CHUNK = 500
    check("chunk-invariant under graph (chunk=13 == chunk=500 bitwise)",
          same_run(g1, g3))
    g4 = GS_LH(X, Y, **{**kw, "seed": 43})
    check("different seed differs",
          not np.array_equal(g1["beta"], g4["beta"]))

    print("=== 4. GPU: quality under graph (both pairings) ===")
    signals = {0: 2.0, 1: -1.5, 2: 1.2, 3: 0.9, 4: -0.7}
    Xq, Yq, _ = simulate(100, 500, signals, sigma=1.0, seed=4)
    out = GS_LH(Xq, Yq, n_iter=2000, burnin=1000, method="fast",
                standardize=False, seed=42, graph=True)
    bm = out["beta"].mean(0)
    s2m = float(out["sigma2"].mean())
    recovered = all(abs(float(bm[j]) - v) < 0.35 for j, v in signals.items())
    noise_max = float(np.abs(bm[5:]).max())
    print(f"  fast graph: sigma2 {s2m:.3f}, signals "
          f"{[round(float(bm[j]), 2) for j in signals]}, noise {noise_max:.3f}")
    check("fast pairing: recovery under graph",
          recovered and noise_max < 0.2 and s2m > 0.2)
    Xd, Yd, _ = simulate(500, 100, {0: 1.5, 1: -1.0, 2: 0.8}, sigma=0.8,
                         seed=5)
    out_d = GS_LH(Xd, Yd, n_iter=1500, burnin=500, method="direct",
                  standardize=False, seed=7, graph=True)
    bmd = out_d["beta"].mean(0)
    rec_d = all(abs(float(bmd[j]) - v) < 0.35
                for j, v in {0: 1.5, 1: -1.0, 2: 0.8}.items())
    nm_d = float(np.abs(bmd[3:]).max())
    print(f"  direct graph: sigma2 {float(out_d['sigma2'].mean()):.3f}, "
          f"signals {[round(float(bmd[j]), 2) for j in [0, 1, 2]]}, "
          f"noise {nm_d:.3f}")
    check("direct pairing: recovery under graph", rec_d and nm_d < 0.2)

    print("=== 5. GPU: speed plain vs graph (informational) ===")
    kws = dict(n_iter=4000, burnin=1000, method="fast", standardize=False,
               seed=1)
    out_p = GS_LH(Xq, Yq, **kws)
    out_g = GS_LH(Xq, Yq, graph=True, **kws)
    r_p = kws["n_iter"] / out_p["runtime_sec"]
    r_g = kws["n_iter"] / out_g["runtime_sec"]
    print(f"  plain scan: {r_p:.0f} it/s   graph: {r_g:.0f} it/s   "
          f"({r_g / r_p - 1:+.0%})")

print()
failed = [t for t, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed"
      + (f"; FAILED: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
