"""Verification for chi2.py and the driver's _accept_chi2 auto-decision
(JAX port of test/_test_chi2.py).

Rule: on the GPU backend, both Gamma draws of an iteration (S1 sigma^2
shape a1 + n/2, S3 lambda shape 2p + a) use the chi-square path iff
max(rel_lam, rel_s2) <= 1e-2, where rel = |round(2*alpha)/2 - alpha| /
alpha bounds the relative bias of the draw's mean and variance.  Unlike
the PyTorch version the approximation is never MANDATORY in JAX
(jax.random.gamma is a pure XLA primitive): it is an accuracy-for-speed
tradeoff on the GPU backend only.  Refusal is silent (exact Gamma
everywhere); non-GPU backends always use exact Gamma.

The GPU sections use a monkeypatched forced-refusal to prove, bitwise,
(a) that acceptance really engages the approximation (chain differs from
the exact-Gamma chain) and (b) that refusal really means exact Gamma
(chain identical to the forced-exact chain).  Sections needing a GPU
backend print a skip note without one (e.g. native Windows); run under
WSL2 with jax[cuda12] to exercise them.
"""
import importlib
import os
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH_JAX import GS_LH, gamma_sample, make_chi2_mask

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


print("=== 1. _accept_chi2 unit checks (any backend) ===")
check("default hyperparameters accepted (p=500, n=100)",
      drv._accept_chi2(500, 1e-3, 1.0, 100) is True)
check("tiny fractional lambda shape refused (p=5, a=0.3)",
      drv._accept_chi2(5, 0.3, 1.0, 100) is False)
check("tiny fractional sigma2 shape refused (n=10, a1=0.3)",
      drv._accept_chi2(500, 1e-3, 0.3, 10) is False)
check("half-integer shapes exact at any size (p=2, a=0.5)",
      drv._accept_chi2(2, 0.5, 1.5, 3) is True)

print("=== 2. chi-square identity: moment checks (any backend) ===")
N = 200_000
keys = jax.random.split(jax.random.PRNGKey(0), N)
for alpha, rate in [(51.0, 2.3), (124.5, 1.0)]:
    mask = make_chi2_mask([alpha])
    draws = jax.jit(jax.vmap(
        lambda k: gamma_sample(k, alpha, jnp.array(rate, jnp.float32),
                               mask)))(keys)
    m_t, v_t = alpha / rate, alpha / rate ** 2
    m, v = float(draws.mean()), float(draws.var())
    print(f"  Ga({alpha}, {rate}) via chi2: mean {m:.4f} vs {m_t:.4f} "
          f"({m / m_t - 1:+.2%}), var {v:.4f} vs {v_t:.4f} ({v / v_t - 1:+.2%})")
    check(f"moments within 1% of theory (alpha={alpha})",
          abs(m / m_t - 1) < 0.01 and abs(v / v_t - 1) < 0.01)
# non-half-integer shape: moments land on the ROUNDED shape's theory
alpha, rate = 51.2, 2.3
alpha_r = round(2 * alpha) / 2
rel = abs(alpha_r - alpha) / alpha
mask = make_chi2_mask([alpha])
draws = jax.jit(jax.vmap(
    lambda k: gamma_sample(k, alpha, jnp.array(rate, jnp.float32),
                           mask)))(keys)
m = float(draws.mean())
print(f"  Ga({alpha}, {rate}): rounded shape {alpha_r}, rel = {rel:.2e}; "
      f"mean {m:.4f} vs rounded theory {alpha_r / rate:.4f} "
      f"({m / (alpha_r / rate) - 1:+.2%})")
check(f"rounded-shape theory matched within 1% (rel = {rel:.1e} <= tol)",
      abs(m / (alpha_r / rate) - 1) < 0.01)

print("=== 3. gamma_sample dispatch (any backend) ===")
key = jax.random.PRNGKey(7)
rate_t = jnp.array(2.0, jnp.float32)
exact = gamma_sample(key, 51.0, rate_t)
direct = jax.random.gamma(key, 51.0, dtype=jnp.float32) / rate_t
check("mask=None == plain jax.random.gamma (bitwise)",
      bool(jnp.array_equal(exact, direct)))
mask = make_chi2_mask([51.0])
chi2 = gamma_sample(key, 51.0, rate_t, mask)
check("mask engages the chi-square algorithm (draw differs)",
      not bool(jnp.array_equal(exact, chi2)))
check("same key + mask reproducible",
      bool(jnp.array_equal(chi2, gamma_sample(key, 51.0, rate_t, mask))))
check("0-dim rate + (1, nu) mask returns a 0-dim draw",
      chi2.shape == ())

print("=== 4. non-GPU: the decision never applies (forced-accept) ===")
if on_gpu:
    print("  skipped (GPU backend; see section 5)")
else:
    Xc, Yc, _ = simulate(60, 80, {0: 1.5}, sigma=1.0, seed=3)
    kw = dict(n_iter=60, burnin=10, method="fast", standardize=False,
              seed=5)
    base = GS_LH(Xc, Yc, **kw)
    orig = drv._accept_chi2
    drv._accept_chi2 = lambda *a, **k: True   # forced acceptance
    try:
        forced = GS_LH(Xc, Yc, **kw)
    finally:
        drv._accept_chi2 = orig
    same = (np.array_equal(base["beta"], forced["beta"])
            and np.array_equal(base["sigma2"], forced["sigma2"]))
    check("non-GPU chain bitwise identical even with _accept_chi2 "
          "forced True", same)

print("=== 5. GPU sections ===")
if not on_gpu:
    print(f"  skipped (backend {jax.default_backend()}); run under WSL2 "
          "with jax[cuda12] to exercise them")
else:
    signals = {0: 2.0, 1: -1.5, 2: 1.2, 3: 0.9, 4: -0.7}
    X, Y, _ = simulate(100, 500, signals, sigma=1.0, seed=4)

    print("  -- 5a. acceptance really engages the approximation --")
    kw = dict(n_iter=200, burnin=50, method="fast", standardize=False,
              seed=42)
    orig = drv._accept_chi2
    drv._accept_chi2 = lambda *a, **k: False   # forced refusal -> exact
    try:
        ref_exact = GS_LH(X, Y, **kw)
    finally:
        drv._accept_chi2 = orig
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        chi2_a = GS_LH(X, Y, **kw)
        chi2_b = GS_LH(X, Y, **kw)
    engaged = not np.array_equal(chi2_a["beta"], ref_exact["beta"])
    repro = (np.array_equal(chi2_a["beta"], chi2_b["beta"])
             and np.array_equal(chi2_a["sigma2"], chi2_b["sigma2"]))
    check("accepted chain != exact-Gamma chain (approximation engaged)",
          engaged)
    check("accepted chain reproducible under same seed", repro)
    check("acceptance silent (no warnings)", len(ws) == 0)
    # chunk-boundary invariance with chi2 engaged (masks fixed per run)
    drv._CHUNK = 13
    try:
        chi2_c = GS_LH(X, Y, **kw)
    finally:
        drv._CHUNK = 500
    check("chunk-invariant with chi2 (chunk=13 == chunk=500 bitwise)",
          np.array_equal(chi2_a["beta"], chi2_c["beta"]))

    print("  -- 5b. refusal really means exact Gamma (bitwise) --")
    # n=12, p=8, a=0.3, a1=0.3: rel_lam ~ 1.2e-2, rel_s2 ~ 3.2e-2 -> refused
    Xt, Yt, _ = simulate(12, 8, {0: 1.0}, sigma=1.0, seed=6)
    kwt = dict(n_iter=40, burnin=10, method="fast", standardize=False,
               a=0.3, a1=0.3, seed=7)
    assert drv._accept_chi2(8, 0.3, 0.3, 12) is False
    refused = GS_LH(Xt, Yt, **kwt)
    drv._accept_chi2 = lambda *a, **k: False
    try:
        refused_forced = GS_LH(Xt, Yt, **kwt)
    finally:
        drv._accept_chi2 = orig
    same = (np.array_equal(refused["beta"], refused_forced["beta"])
            and np.array_equal(refused["sigma2"], refused_forced["sigma2"]))
    check("refused chain identical to forced-exact chain", same)

    print("  -- 5c. end-to-end quality with the approximation engaged --")
    out = GS_LH(X, Y, n_iter=2000, burnin=1000, method="fast",
                standardize=False, seed=42)
    bm = out["beta"].mean(0)
    s2m = float(out["sigma2"].mean())
    recovered = all(abs(float(bm[j]) - v) < 0.35 for j, v in signals.items())
    noise_max = float(np.abs(bm[5:]).max())
    print(f"  sigma2 mean {s2m:.3f} (true 1.0; ~0.33 expected in p>>n), "
          f"signals {[round(float(bm[j]), 2) for j in signals]}, "
          f"noise max {noise_max:.3f}")
    check("recovery with chi-square engaged",
          recovered and noise_max < 0.2 and s2m > 0.2)

    print("  -- 5d. speed: exact Gamma vs chi-square (informational) --")
    kws = dict(n_iter=4000, burnin=1000, method="fast", standardize=False,
               seed=1)
    drv._accept_chi2 = lambda *a, **k: False
    try:
        out_e = GS_LH(X, Y, **kws)
    finally:
        drv._accept_chi2 = orig
    out_c = GS_LH(X, Y, **kws)
    r_e = kws["n_iter"] / out_e["runtime_sec"]
    r_c = kws["n_iter"] / out_c["runtime_sec"]
    print(f"  exact Gamma: {r_e:.0f} it/s   chi-square: {r_c:.0f} it/s   "
          f"({r_c / r_e - 1:+.0%})")

print("=== 6. non-GPU end-to-end regression ===")
if on_gpu:
    print("  skipped (GPU backend; quality covered by 5c)")
else:
    signals = {0: 2.0, 1: -1.5, 2: 1.2, 3: 0.9, 4: -0.7}
    X, Y, _ = simulate(100, 500, signals, sigma=1.0, seed=4)
    out = GS_LH(X, Y, n_iter=2000, burnin=1000, method="fast",
                standardize=False, seed=42)
    bm = out["beta"].mean(0)
    s2m = float(out["sigma2"].mean())
    recovered = all(abs(float(bm[j]) - v) < 0.35 for j, v in signals.items())
    noise_max = float(np.abs(bm[5:]).max())
    print(f"  sigma2 mean {s2m:.3f}, noise max {noise_max:.3f}")
    check("CPU recovery regression (exact Gamma)",
          recovered and noise_max < 0.2 and s2m > 0.2)

print()
failed = [t for t, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed"
      + (f"; FAILED: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
