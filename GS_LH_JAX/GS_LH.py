"""
GS_LH.py  (JAX port, single precision)

Main Gibbs sampler driver for Bayesian L1/2 regression
(Ke & Fan, 2024, JCGS, DOI: 10.1080/10618600.2024.2374579).

Model:
    Y = X beta + eps,  eps ~ N_n(0, sigma^2 I_n),
    Y centered; columns of X standardized (||X_j||^2 = n).

Prior (on error-standardized coefficients beta~_j = beta_j / sigma):
    pi(beta_j | lambda, sigma)
        = (lambda^2 / (4 sigma)) exp(-lambda sqrt(|beta_j| / sigma)),
    lambda  ~ Gamma(a, b),          [shape, rate; c, d in the note]
    sigma^2 ~ InvGamma(a1, b1).

Each iteration runs S1-S5 in order, all conditionals using the freshest
values:
    S1: sigma^2 update (collapsed, beta integrated out)   [sigma2_sample]
    S2: beta update (multivariate normal)                 [beta_sample]
    S3: lambda update (Gamma)                             [shrinkage_sample]
    S4: v_j updates (InvGaussian reciprocal)              [shrinkage_sample]
    S5: tau_j^2 updates (InvGaussian reciprocal)          [shrinkage_sample]

Chain structure: the Markov state carried between iterations is only
w = tau / lambda^2 -- S1 draws sigma^2 from w alone, S2 draws beta from
(w, sigma^2), and S3-S5 regenerate (lambda, v, tau) from
beta~ = beta/sigma. Hence beta and sigma^2 are never initialized; the
chain starts from w0 (default: ones, i.e. lambda = v_j = tau_j^2 = 1).

JAX-specific notes
------------------
* All computation is single precision (float32): inputs are coerced on
  entry, and jax's x64 mode is neither required nor enabled by this
  package.
* Randomness is explicit: `seed` seeds one jax.random.PRNGKey that is
  split into one key per iteration up front; the same seed reproduces
  the same chain.
* The chain is executed as jax.lax.scan over the iteration keys, in
  chunks of _CHUNK iterations: each chunk is ONE compiled XLA While loop
  (S1-S5 inside, one Cholesky factorization shared between S1 and S2 per
  iteration), with no Python dispatch or host synchronization inside.
  Between chunks the draws are copied to the host (bounded device memory,
  progress reports). Compilation is cached per (chunk length, shapes,
  method) across GS_LH calls; GS_LH compiles ahead of time
  (lower().compile()) so no compilation is counted in runtime_sec.
* Posterior draws are stored on the host and returned as numpy arrays
  (the JAX analog of the PyTorch version's CPU tensors).

Algorithm pairings (argument `method`; S1's factorization is shared with
S2 within every iteration):
    "fast"    S1 = sigma2_sample (n x n Cholesky of Sigma, shared L),
              S2 = beta_sample (Bhattacharya et al. 2016, Algorithm 1).
              Best when p >= n (dense X). Cost per iteration O(n^2 p + n^3).
    "direct"  S1 = sigma2_sample_direct (p x p Woodbury form, shared L),
              S2 = beta_sample_direct (Prop. 2.1 + Cholesky = Rue 2001).
              Best when n > p (dense X). Cost per iteration O(p^3) after
              a one-time O(n p^2) precompute of X'X, X'Y, Y'Y.
    "auto"    "fast" if p >= n else "direct".
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from .beta_sample import beta_sample, beta_sample_direct
from .shrinkage_sample import shrinkage
from .sigma2_sample import sigma2_sample, sigma2_sample_direct

__all__ = ["GS_LH"]

# Iterations per lax.scan chunk: bounds device-side draw storage to
# _CHUNK x p floats and gives one progress report per chunk.
_CHUNK = 500


# One full Gibbs iteration (S1-S5); S1's Cholesky factor L is shared with
# S2 within the iteration. Scalar hyperparameters are traced, so changing
# them does not trigger recompilation. These stay UN-jitted: they are
# compiled inline into the scan runners below.

def _step_fast(w, key, X, Y, a1, b1, a, b):
    k1, k2, k3 = jax.random.split(key, 3)
    sigma2, L = sigma2_sample(k1, X, Y, w, a1, b1, return_L=True)
    sigma = jnp.sqrt(sigma2)
    beta = beta_sample(k2, X, Y, w, sigma, L=L)
    return shrinkage(k3, beta / sigma, a, b), beta, sigma2


def _step_direct(w, key, X, Y, XtX, XtY, YtY, a1, b1, a, b):
    k1, k2, k3 = jax.random.split(key, 3)
    sigma2, L = sigma2_sample_direct(k1, X, Y, w, a1, b1, return_L=True,
                                     XtX=XtX, XtY=XtY, YtY=YtY)
    sigma = jnp.sqrt(sigma2)
    beta = beta_sample_direct(k2, X, Y, w, sigma, L=L, XtX=XtX, XtY=XtY)
    return shrinkage(k3, beta / sigma, a, b), beta, sigma2


def _scan_runner(step):
    """Wrap a per-iteration step into a jitted lax.scan over a chunk of
    iteration keys. Returns (w_final, (betas, sigma2s)) for the chunk."""

    @jax.jit
    def run(w, keys, *args):
        def f(w, k):
            w, beta, sigma2 = step(w, k, *args)
            return w, (beta, sigma2)

        return jax.lax.scan(f, w, keys)

    return run


_run_fast = _scan_runner(_step_fast)
_run_direct = _scan_runner(_step_direct)


def GS_LH(X, Y, n_iter, burnin=0, a1=1.0, b1=1.0, a=1e-3, b=1e-3,
          method="auto", w0=None, standardize=True, seed=None,
          verbose=False):
    """Run the Bayesian L1/2 Gibbs sampler (JAX, single precision).

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix (dense). Coerced to float32.
    Y : array-like, shape (n,)
        Response vector. Coerced to float32.
    n_iter : int
        Number of posterior draws to KEEP (after burn-in).
    burnin : int
        Number of initial iterations to discard.
    a1, b1 : float
        Prior sigma^2 ~ InvGamma(a1, b1). Default (1, 1): weakly
        informative but vanishing at the origin -- recommended for
        p >> n, where the customary near-Jeffreys choice (1e-3, 1e-3)
        lets the posterior collapse onto near-interpolating solutions
        (sigma^2 -> 0); see the WARNING in sigma2_sample.py.
    a, b : float
        Hyperprior lambda ~ Gamma(a, b) [shape, rate]; weakly informative
        default 1e-3, 1e-3 (the global-rate prior was found to have
        little influence on the sigma^2 behaviour above).
    method : {"auto", "fast", "direct"}
        Algorithm pairing for S1/S2 (see module docstring).
    w0 : array-like, shape (p,), optional
        Initial state w = tau/lambda^2 (default: vector of ones).
    standardize : bool
        If True (default), center Y, center X columns and rescale them to
        ||X_j||^2 = n, as the model assumes. Back-transformation to the
        original scale:  beta_orig_j = beta_j / x_scale_j,
        intercept = y_mean - x_mean . beta_orig  (stats in "preprocess").
        Set False when X is already standardized and Y centered.
    seed : int, optional
        Seed for jax.random.PRNGKey. None draws a random seed.
    verbose : bool
        Print progress about every 10% of iterations.

    Returns
    -------
    dict with keys
        "beta"       (n_iter, p) numpy float32 array of posterior beta draws
        "sigma2"     (n_iter,) numpy float32 array of posterior sigma^2 draws
        "method"     the algorithm pairing actually used
        "preprocess" {"y_mean", "x_mean", "x_scale"} (numpy) or None
        "w_final"    last state w (numpy; for warm restarts via w0)
        "runtime_sec" wall-clock seconds of the scan loop (excludes the
                     ahead-of-time JIT compilation)
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    key = jax.random.PRNGKey(seed)

    # single precision throughout
    X = jnp.asarray(X, dtype=jnp.float32)
    Y = jnp.asarray(Y, dtype=jnp.float32)
    n, p = X.shape

    # --- resolve algorithm pairing ---
    if method == "auto":
        method = "fast" if p >= n else "direct"
    if method not in ("fast", "direct"):
        raise ValueError(f"unknown method {method!r}")

    # --- standardization (model assumption: Y centered, ||X_j||^2 = n) ---
    preprocess = None
    if standardize:
        y_mean = Y.mean()
        Y = Y - y_mean
        x_mean = X.mean(axis=0)
        X = X - x_mean
        x_scale = jnp.linalg.norm(X, axis=0) / np.sqrt(n)
        x_scale = jnp.where(x_scale > 0, x_scale, 1.0)
        X = X / x_scale
        preprocess = {"y_mean": float(y_mean),
                      "x_mean": np.asarray(x_mean),
                      "x_scale": np.asarray(x_scale)}

    # --- chain state: only w = tau/lambda^2 ---
    if w0 is None:
        w = jnp.ones(p, dtype=jnp.float32)
    else:
        w = jnp.asarray(w0, dtype=jnp.float32)
        if bool(jnp.any(w <= 0)):
            raise ValueError("w0 must be strictly positive")

    if method == "fast":
        run, args = _run_fast, (X, Y, a1, b1, a, b)
    else:  # "direct": X'X, X'Y, Y'Y are constant across iterations
        XtX, XtY = X.T @ X, X.T @ Y
        YtY = jnp.dot(Y, Y)
        run, args = _run_direct, (X, Y, XtX, XtY, YtY, a1, b1, a, b)

    total = burnin + n_iter
    keys = jax.random.split(key, total)

    # chunk lengths: min(_CHUNK, remaining) -- at most two distinct
    # lengths, hence at most two XLA compilations per (n, p, method)
    sizes = []
    left = total
    while left > 0:
        sizes.append(min(_CHUNK, left))
        left -= sizes[-1]

    beta_keep = np.empty((n_iter, p), dtype=np.float32)   # stored on host
    sigma2_keep = np.empty(n_iter, dtype=np.float32)

    # ahead-of-time compilation (not timed); the jit cache makes the
    # scan calls below reuse these executables
    start = 0
    for c in set(sizes):
        run.lower(w, keys[start:start + c], *args).compile()
        start += c

    t_start = time.perf_counter()
    print_every = max(1, len(sizes) // 10)
    start = 0
    for ci, c in enumerate(sizes):
        w, (betas, s2s) = run(w, keys[start:start + c], *args)
        s2_host = np.asarray(s2s)

        # --- collect the part of the chunk after burn-in ---
        g0 = max(start, burnin)
        if g0 < start + c:
            beta_keep[g0 - burnin:start + c - burnin] = \
                np.asarray(betas[g0 - start:])
            sigma2_keep[g0 - burnin:start + c - burnin] = s2_host[g0 - start:]

        if verbose and (ci % print_every == 0 or ci == len(sizes) - 1):
            rate = (start + c) / (time.perf_counter() - t_start)
            print(f"[GS_LH] {start + c}/{total} iterations "
                  f"({rate:.0f} it/s), sigma2 = {s2_host[-1]:.4g}")
        start += c

    runtime = time.perf_counter() - t_start

    out = {
        "beta": beta_keep,
        "sigma2": sigma2_keep,
        "method": method,
        "preprocess": preprocess,
        "w_final": np.asarray(w),
        "runtime_sec": runtime,
    }
    return out
