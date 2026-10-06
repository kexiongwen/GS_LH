"""Statistical verification for GS_LH_JAX/beta_sample.py (float32).

Reference moments are computed in float64 numpy (test-side only; the
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


def theoretical(X, Y, w, sigma):
    """Exact mu = V X'Y and Sigma = sigma^2 V by direct inversion (float64)."""
    X64, Y64, w64 = (np.asarray(a, np.float64) for a in (X, Y, w))
    A = X64.T @ X64 + np.diag(w64 ** -2)
    V = np.linalg.inv(A)
    s = float(sigma)
    return V @ X64.T @ Y64, s ** 2 * V


def moment_check(fn, n, p, N=200_000, seed=0, **kw):
    g = np.random.default_rng(seed)
    X = jnp.asarray(g.standard_normal((n, p)), jnp.float32)
    Y = jnp.asarray(g.standard_normal(n), jnp.float32)
    w = jnp.asarray(g.uniform(0.1, 2.1, p), jnp.float32)
    sigma = jnp.float32(1.3)

    # exercise the exact path the GS_LH_JAX driver uses for the direct
    # sampler: precomputed XtX / XtY (constant across Gibbs iterations)
    if fn is beta_sample_direct and "XtX" not in kw:
        kw = {"XtX": X.T @ X, "XtY": X.T @ Y, **kw}

    mu, Sigma = theoretical(X, Y, w, sigma)

    @jax.jit
    def accumulate(key):
        def f(carry, k):
            s1, s2 = carry
            b = fn(k, X, Y, w, sigma, **kw)
            return (s1 + b, s2 + jnp.outer(b, b)), None

        keys = jax.random.split(key, N)
        return jax.lax.scan(f, (jnp.zeros(p), jnp.zeros((p, p))), keys)[0]

    s1, s2 = accumulate(jax.random.PRNGKey(seed))
    mean_hat = np.asarray(s1, np.float64) / N
    cov_hat = np.asarray(s2, np.float64) / N - np.outer(mean_hat, mean_hat)

    sd = np.sqrt(np.diag(Sigma))
    mean_z = np.abs((mean_hat - mu) / sd).max()
    cov_err = np.linalg.norm(cov_hat - Sigma) / np.linalg.norm(Sigma)
    # MC error scales: E|cov_hat - Sigma|_F^2 ~ (tr(Sigma)^2 + |Sigma|_F^2) / N
    # (diagonal-dominated Sigma makes it much larger than the naive 1/sqrt(N))
    mean_z_exp = np.sqrt(2 * np.log(2 * p) / N)
    cov_err_exp = np.sqrt((np.trace(Sigma) ** 2 + np.linalg.norm(Sigma) ** 2)
                          / N) / np.linalg.norm(Sigma)
    print(f"{fn.__name__} n={n}, p={p}: max |z| of mean = {mean_z:.4f} "
          f"(~{mean_z_exp:.4f} expected), rel. Frobenius err of cov = {cov_err:.4%} "
          f"(~{cov_err_exp:.4%} expected)")
    ok = mean_z < 1.5 * mean_z_exp and cov_err < 1.5 * cov_err_exp
    print("  PASS" if ok else "  FAIL")


def reuse_check():
    n, p = 30, 200
    g = np.random.default_rng(1)
    X = jnp.asarray(g.standard_normal((n, p)), jnp.float32)
    Y = jnp.asarray(g.standard_normal(n), jnp.float32)
    w = jnp.asarray(g.uniform(0.1, 1.1, p), jnp.float32)
    sigma = jnp.float32(0.9)

    Xw = X * w
    L = jnp.linalg.cholesky(Xw @ Xw.T + jnp.eye(n))

    key = jax.random.PRNGKey(123)
    b1 = beta_sample(key, X, Y, w, sigma)
    b2 = beta_sample(key, X, Y, w, sigma, L=L)
    ok = bool(jnp.array_equal(b1, b2))
    print("precomputed-L draw identical:", ok)
    assert ok

    XtX, XtY = X.T @ X, X.T @ Y
    # B-form system of beta_sample_direct (W(X'X)W + I), as GS_LH builds it
    B = (w[:, None] * XtX) * w[None, :]
    B = B.at[jnp.diag_indices(p)].add(1.0)
    Lp = jnp.linalg.cholesky(B)
    d1 = beta_sample_direct(key, X, Y, w, sigma)
    d2 = beta_sample_direct(key, X, Y, w, sigma, L=Lp, XtX=XtX, XtY=XtY)
    ok = bool(jnp.array_equal(d1, d2))
    print("direct precomputed-path draw identical:", ok)
    assert ok


def benchmark(n=100, p=4000, reps=200):
    g = np.random.default_rng(2)
    X = jnp.asarray(g.standard_normal((n, p)), jnp.float32)
    Y = jnp.asarray(g.standard_normal(n), jnp.float32)
    w = jnp.asarray(g.uniform(0.1, 1.1, p), jnp.float32)
    w = w.at[:20].set(3.0)
    sigma = jnp.float32(1.0)
    XtX, XtY = X.T @ X, X.T @ Y

    for name, fn in [("Algorithm 1 (fast)", beta_sample),
                     ("direct", beta_sample_direct)]:
        kw = {} if fn is beta_sample else {"XtX": XtX, "XtY": XtY}
        jfn = jax.jit(lambda k: fn(k, X, Y, w, sigma, **kw))
        keys = list(jax.random.split(jax.random.PRNGKey(3), reps))
        jax.block_until_ready(jfn(keys[0]))  # warm-up (compile)
        t0 = time.perf_counter()
        for k in keys:
            jax.block_until_ready(jfn(k))
        dt = (time.perf_counter() - t0) / reps
        print(f"{name}: {dt * 1000:.2f} ms per draw (n={n}, p={p})")


if __name__ == "__main__":
    moment_check(beta_sample, n=30, p=200)     # p >> n regime
    moment_check(beta_sample, n=200, p=30)     # n > p regime
    moment_check(beta_sample_direct, n=200, p=30)  # driver path (XtX/XtY)
    moment_check(beta_sample_direct, n=30, p=200)  # p >> n: rank-deficient X'X
    reuse_check()
    benchmark()
