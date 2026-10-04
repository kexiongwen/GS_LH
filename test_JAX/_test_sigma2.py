"""Verification for GS_LH_JAX/sigma2_sample.py (float32):
distribution moments, L-sharing, benchmarks.

Reference quantities are computed in float64 numpy (test-side only; the
package itself is strictly single precision).
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH_JAX.beta_sample import beta_sample, beta_sample_direct
from GS_LH_JAX.sigma2_sample import sigma2_sample, sigma2_sample_direct


def exact_q(X, Y, w):
    """q = Y' Sigma^{-1} Y by brute force (float64 numpy)."""
    X64, Y64, w64 = (np.asarray(a, np.float64) for a in (X, Y, w))
    n = X64.shape[0]
    M = (X64 * w64) @ (X64 * w64).T + np.eye(n)
    return float(Y64 @ np.linalg.solve(M, Y64))


def distribution_check(fn, n, p, a1=3.0, b1=2.0, N=200_000, seed=0, **kw):
    """Sample sigma^2 with fixed (X, Y, w); compare to InvGamma theory."""
    g = np.random.default_rng(seed)
    X = jnp.asarray(g.standard_normal((n, p)), jnp.float32)
    Y = jnp.asarray(g.standard_normal(n), jnp.float32)
    w = jnp.asarray(g.uniform(0.1, 2.1, p), jnp.float32)

    q = exact_q(X, Y, w)
    alpha, gamma = a1 + n / 2, b1 + q / 2
    mean_true = gamma / (alpha - 1)
    var_true = gamma ** 2 / ((alpha - 1) ** 2 * (alpha - 2))

    @jax.jit
    def accumulate(key):
        def f(carry, k):
            s1, s2 = carry
            s = fn(k, X, Y, w, a1, b1, **kw)
            return (s1 + s, s2 + s * s), None

        keys = jax.random.split(key, N)
        return jax.lax.scan(f, (jnp.float32(0.0), jnp.float32(0.0)), keys)[0]

    s1, s2 = accumulate(jax.random.PRNGKey(seed))
    m, v = float(s1) / N, float(s2) / N - (float(s1) / N) ** 2
    print(f"{fn.__name__} (n={n}, p={p}): mean {m:.5f} vs {mean_true:.5f} "
          f"({(m / mean_true - 1):+.2%}), var {v:.5f} vs {var_true:.5f} "
          f"({(v / var_true - 1):+.2%})")
    ok = abs(m / mean_true - 1) < 0.01 and abs(v / var_true - 1) < 0.03
    print("  PASS" if ok else "  FAIL")


def sharing_check():
    n, p = 30, 200
    g = np.random.default_rng(1)
    X = jnp.asarray(g.standard_normal((n, p)), jnp.float32)
    Y = jnp.asarray(g.standard_normal(n), jnp.float32)
    w = jnp.asarray(g.uniform(0.1, 1.1, p), jnp.float32)
    a1, b1 = 3.0, 2.0

    # n x n factor shared S1 -> S2 (Algorithm 1)
    k1, k2 = jax.random.split(jax.random.PRNGKey(11))
    s2, L = sigma2_sample(k1, X, Y, w, a1, b1, return_L=True)
    b = beta_sample(k2, X, Y, w, jnp.sqrt(s2), L=L)
    print("n x n L shared with beta_sample: beta finite:",
          bool(jnp.isfinite(b).all()))

    # p x p factor shared S1 -> S2 (direct)
    s2, Lp = sigma2_sample_direct(k1, X, Y, w, a1, b1, return_L=True)
    b = beta_sample_direct(k2, X, Y, w, jnp.sqrt(s2), L=Lp)
    print("p x p L shared with beta_sample_direct: beta finite:",
          bool(jnp.isfinite(b).all()))

    # precomputed XtX/XtY/YtY path (as driven by GS_LH): identical draws
    XtX, XtY, YtY = X.T @ X, X.T @ Y, jnp.dot(Y, Y)
    s2a, La = sigma2_sample_direct(k1, X, Y, w, a1, b1, return_L=True)
    ba = beta_sample_direct(k2, X, Y, w, jnp.sqrt(s2a), L=La)
    s2b, Lb = sigma2_sample_direct(k1, X, Y, w, a1, b1, return_L=True,
                                   XtX=XtX, XtY=XtY, YtY=YtY)
    bb = beta_sample_direct(k2, X, Y, w, jnp.sqrt(s2b), L=Lb, XtX=XtX, XtY=XtY)
    print("precomputed-XtX path identical:",
          bool(jnp.array_equal(s2a, s2b) and jnp.array_equal(ba, bb)))

    # determinism under the same key
    r1 = sigma2_sample(k1, X, Y, w, a1, b1)
    r2 = sigma2_sample(k1, X, Y, w, a1, b1)
    print("deterministic under same key:", bool(jnp.array_equal(r1, r2)))


def bench(fn, args, reps):
    jfn = jax.jit(fn)
    keys = list(jax.random.split(jax.random.PRNGKey(3), reps))
    jax.block_until_ready(jfn(keys[0], *args))  # warm-up (compile)
    t0 = time.perf_counter()
    for k in keys:
        jax.block_until_ready(jfn(k, *args))
    return (time.perf_counter() - t0) / reps


def benchmark():
    a1, b1 = 3.0, 2.0
    print("\nS1 timing per draw (sparse w, s=20); * = fastest in row:")
    for n, p, reps in [(100, 4000, 50), (8000, 500, 20)]:
        g = np.random.default_rng(3)
        X = jnp.asarray(g.standard_normal((n, p)), jnp.float32)
        Y = jnp.asarray(g.standard_normal(n), jnp.float32)
        w = jnp.full((p,), 0.02, jnp.float32).at[:20].set(3.0)
        args = (X, Y, w, a1, b1)
        t1 = bench(sigma2_sample, args, reps)
        t2 = bench(sigma2_sample_direct, args, reps)
        best = min(t1, t2)
        print(f"  CPU float32 n={n:5d}, p={p:5d}: "
              f"n-chol {t1 * 1e3:8.2f}{'*' if t1 == best else ' '} "
              f"p-chol {t2 * 1e3:8.2f}{'*' if t2 == best else ' '} ms")


if __name__ == "__main__":
    distribution_check(sigma2_sample, n=30, p=200)
    distribution_check(sigma2_sample_direct, n=200, p=30)
    sharing_check()
    benchmark()
