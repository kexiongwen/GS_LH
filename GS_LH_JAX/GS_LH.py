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
  progress reports, one finiteness audit of w and sigma^2 per chunk).
  Compilation is cached per (chunk length, shapes, method) across GS_LH
  calls; GS_LH compiles ahead of time (lower().compile()) so no
  compilation is counted in runtime_sec.
* Posterior draws are stored on the host and returned as numpy arrays
  (the JAX analog of the PyTorch version's CPU tensors).
* jax.random.gamma is a pure XLA primitive -- exact, no host sync -- so
  the chi-square Gamma approximation of chi2.py is never MANDATORY here
  (in the PyTorch version torch._standard_gamma syncs the host, forcing
  chi-square under CUDA-graph capture).  On the GPU backend the driver
  still offers it as an accuracy-for-speed tradeoff: _accept_chi2 applies
  the same rule as the PyTorch version (half-integer rounding of the
  Gamma shapes negligible, e.g. large n/p or integer shapes), and the
  approximation then applies to every chunk runner; otherwise, and on
  every non-GPU backend, exact Gamma is used.  An accepted chi-square
  run is still bitwise-reproducible from `seed` (the dof masks are fixed
  per run), but its chain differs bitwise from the exact-Gamma chain.
* With graph=True (GPU backend only), the chunk runners are compiled
  with XLA command buffers (_GRAPH_COMPILER_OPTIONS): each chunk's whole
  scan While loop is recorded as ONE CUDA graph on first execution and
  replayed afterwards, removing the per-kernel launch overhead between
  iterations -- the JAX analog of the PyTorch version's graph=True, but
  with no manual capture protocol.  The same compiled kernels are
  replayed, so the chain is bitwise-identical to the plain runners for a
  fixed seed; exact Gamma is capturable, hence graph mode is NOT gated
  by the chi-square decision.  On non-GPU backends -- and on JAX
  versions without command-buffer support -- graph=True issues a
  RuntimeWarning and runs with the plain runners.

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
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .beta_sample import beta_sample, beta_sample_direct
from .chi2 import make_chi2_mask
from .shrinkage_sample import shrinkage
from .sigma2_sample import sigma2_sample, sigma2_sample_direct

__all__ = ["GS_LH"]

# Iterations per lax.scan chunk: bounds device-side draw storage to
# _CHUNK x p floats and gives one progress report per chunk.
_CHUNK = 500


def _accept_chi2(p, a, a1, n, tol=1e-2, verbose=False):
    r"""Decide whether the chi-square Gamma approximation is accurate
    enough for this run (see chi2.py; port of the PyTorch version's
    decision rule).  The chi-square path rounds the shapes to the nearest
    half-integer (|delta| <= 0.25 on each shape).  Both Gamma shapes of
    the sampler are constant across the whole run, so this one check
    decides every iteration:

        S3 lambda : alpha_lam = 2p + a,
                    rel_lam = |round(2*alpha_lam)/2 - alpha_lam| / alpha_lam,
        S1 sigma2 : alpha_s2  = a1 + n/2,
                    rel_s2  = |round(2*alpha_s2)/2 - alpha_s2| / alpha_s2.

    Returns True  -- max(rel_lam, rel_s2) <= tol: the relative bias of
    each Gamma draw's mean and variance is at most tol (an O(1/p) resp.
    O(1/n) perturbation, negligible in the regimes this sampler targets;
    (half-)integer shapes give rel = 0) and the approximation is used
    automatically on the GPU backend;
    Returns False -- otherwise: refused; a one-line note is printed when
    verbose=True and exact Gamma is used everywhere.
    """
    alpha_lam = 2 * p + a
    rel_lam = abs(round(2 * alpha_lam) / 2 - alpha_lam) / alpha_lam
    alpha_s2 = a1 + 0.5 * n
    rel_s2 = abs(round(2 * alpha_s2) / 2 - alpha_s2) / alpha_s2
    rel = max(rel_lam, rel_s2)
    if rel > tol:
        if verbose:
            print(f"GS_LH: chi-square approximation refused (shape "
                  f"perturbation {rel:.1e} > {tol:g}); using exact Gamma",
                  flush=True)
        return False
    return True


# One full Gibbs iteration (S1-S5); S1's Cholesky factor L is shared with
# S2 within the iteration. Scalar hyperparameters are traced, so changing
# them does not trigger recompilation. These stay UN-jitted: they are
# compiled inline into the scan runners below.

def _step_fast(w, key, X, Y, a1, b1, a, b, mask_s2, mask_lam):
    k1, k2, k3 = jax.random.split(key, 3)
    sigma2, L = sigma2_sample(k1, X, Y, w, a1, b1, return_L=True,
                              chi2_mask=mask_s2)
    sigma = jnp.sqrt(sigma2)
    beta = beta_sample(k2, X, Y, w, sigma, L=L)
    return shrinkage(k3, beta / sigma, a, b, chi2_mask=mask_lam), beta, sigma2


def _step_direct(w, key, X, Y, XtX, XtY, YtY, a1, b1, a, b,
                 mask_s2, mask_lam):
    k1, k2, k3 = jax.random.split(key, 3)
    sigma2, L = sigma2_sample_direct(k1, X, Y, w, a1, b1, return_L=True,
                                     XtX=XtX, XtY=XtY, YtY=YtY,
                                     chi2_mask=mask_s2)
    sigma = jnp.sqrt(sigma2)
    beta = beta_sample_direct(k2, X, Y, w, sigma, L=L, XtX=XtX, XtY=XtY)
    return shrinkage(k3, beta / sigma, a, b, chi2_mask=mask_lam), beta, sigma2


def _scan_runner(step, compiler_options=None):
    """Wrap a per-iteration step into a jitted lax.scan over a chunk of
    iteration keys. Returns (w_final, (betas, sigma2s)) for the chunk.

    compiler_options are forwarded to jax.jit (used by the graph runners
    below: XLA command buffers record the whole chunk While loop as one
    CUDA graph on the GPU backend).
    """

    @jax.jit(compiler_options=compiler_options)
    def run(w, keys, *args):
        def f(w, k):
            w, beta, sigma2 = step(w, k, *args)
            return w, (beta, sigma2)

        return jax.lax.scan(f, w, keys)

    return run


# XLA:GPU command-buffer (CUDA graph) capture of the whole scan While
# loop: with CONDITIONAL+WHILE enabled, the entire chunk executable is
# recorded as ONE CUDA graph on first execution and replayed afterwards,
# removing the per-kernel launch overhead between iterations.  Every op
# in an iteration (GPU RNG, cuBLAS/cuSOLVER custom calls, fusions) is
# capturable, and the draws are bitwise-identical to the plain runners
# for a fixed seed (the same compiled kernels are replayed).  Passed
# per-compile so graph mode is a per-call choice (no XLA_FLAGS needed
# before importing jax).
_GRAPH_COMPILER_OPTIONS = {
    "xla_gpu_enable_command_buffer":
        "FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE",
    "xla_gpu_graph_min_graph_size": 1,
}

_run_fast = _scan_runner(_step_fast)
_run_direct = _scan_runner(_step_direct)

# Graph runners are built under a try: jax.jit's compiler_options argument
# (per-compile XLA command buffers) needs a recent JAX, and a failure at
# import time would break even plain runs on older versions.  When
# unsupported, graph=True warns and runs with the plain runners.
try:
    _run_fast_graph = _scan_runner(_step_fast, _GRAPH_COMPILER_OPTIONS)
    _run_direct_graph = _scan_runner(_step_direct, _GRAPH_COMPILER_OPTIONS)
    _GRAPH_RUNNERS_OK = True
except TypeError:
    _run_fast_graph = _run_direct_graph = None
    _GRAPH_RUNNERS_OK = False


def GS_LH(X, Y, n_iter, burnin=0, a1=1.0, b1=1.0, a=1e-3, b=1e-3,
          method="auto", w0=None, standardize=True, seed=None,
          verbose=False, graph=False):
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
    graph : bool
        If True and the backend is GPU, compile the chunk runners with
        XLA command buffers (CUDA graphs): each chunk's whole scan While
        loop -- all its iterations, RNG included -- is recorded as one
        CUDA graph on first execution and replayed afterwards, removing
        the per-kernel launch overhead of the scan loop (the JAX analog
        of the PyTorch version's graph=True).  The math is unchanged:
        the same compiled kernels are replayed, so draws remain exact
        and bitwise-identical to the plain runners for a fixed seed.
        Requires a GPU backend and a JAX with command-buffer support
        (jax.jit's compiler_options); elsewhere a RuntimeWarning is
        issued and the run proceeds with the plain runners.

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

    Notes
    -----
    On the GPU backend the two Gamma draws of every iteration (S1
    sigma^2, S3 lambda) automatically use the chi-square path of chi2.py
    when the one-time _accept_chi2 check accepts the half-integer
    rounding of their shapes (relative moment bias <= 1e-2; both shapes,
    2p + a and a1 + n/2, are constant across the run, and (half-)integer
    shapes incur no approximation at all).  Refusal is silent -- exact
    Gamma everywhere (a one-line note when verbose=True); non-GPU
    backends always use exact Gamma.  For (half-)integer shapes the
    chi-square draw is exact -- only the computation changes; otherwise
    each Gamma conditional is perturbed by at most the accepted relative
    bias on its mean and variance.

    After each chunk the chain state is audited for finiteness (the
    chunk's sigma^2 draws and the carried w; one host check per chunk,
    cost negligible -- the JAX analog of the PyTorch version's
    audit_every, unconditional).  A non-finite state raises
    FloatingPointError instead of silently contaminating the chain;
    the check is needed because jnp.linalg.cholesky does not raise on
    failure -- it returns NaN.
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
        if not bool(jnp.isfinite(w).all()) or bool(jnp.any(w <= 0)):
            raise ValueError("w0 must be finite and strictly positive")

    # --- runner selection (graph mode = XLA command buffers on GPU) ---------
    # The graph runners replay the same compiled kernels as the plain
    # ones, so the chain is bitwise-identical either way; graph=True only
    # changes HOW the chunk While loop is launched.  Exact Gamma is
    # capturable in JAX, so the chi-square decision does NOT gate graph
    # mode here (unlike the PyTorch version).
    on_gpu = jax.default_backend() == "gpu"
    use_graph = graph and on_gpu and _GRAPH_RUNNERS_OK
    if graph and not use_graph:
        why = (f"got the '{jax.default_backend()}' backend" if not on_gpu
               else "this JAX lacks jax.jit(compiler_options=...) "
                    "(command buffers)")
        warnings.warn(
            f"graph=True requires a GPU backend with command-buffer "
            f"support ({why}); running with the plain runners",
            RuntimeWarning,
        )
    if method == "fast":
        run, args = (_run_fast_graph if use_graph else _run_fast), \
            (X, Y, a1, b1, a, b)
    else:  # "direct": X'X, X'Y, Y'Y are constant across iterations
        XtX, XtY = X.T @ X, X.T @ Y
        YtY = jnp.dot(Y, Y)
        run, args = (_run_direct_graph if use_graph else _run_direct), \
            (X, Y, XtX, XtY, YtY, a1, b1, a, b)

    # --- chi-square decision (GPU backend only; one-time, decides every
    # iteration since both Gamma shapes are run-time constants) ---
    # Accepted -> the S1/S3 Gamma draws use the chi-square identity with
    # half-integer-rounded dof in every chunk runner (see chi2.py);
    # refused or non-GPU -> exact Gamma everywhere.  Masks built ONCE here.
    chi2_ok = jax.default_backend() == "gpu" and _accept_chi2(
        p, a, a1, n, verbose=verbose)
    mask_lam = make_chi2_mask([2 * p + a]) if chi2_ok else None
    mask_s2 = make_chi2_mask([a1 + 0.5 * n]) if chi2_ok else None
    args = args + (mask_s2, mask_lam)

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

        # --- finiteness audit: one host check per chunk (the JAX analog of
        # the PyTorch version's audit_every, unconditional).  Together with
        # the w check this catches any non-finite beta of the chunk: one at
        # iteration i poisons w at i, hence sigma2 at i + 1, and one at the
        # final iteration poisons the returned w itself.  Needed because
        # jnp.linalg.cholesky does not raise on failure -- it returns NaN.
        if not (np.isfinite(s2_host).all() and bool(jnp.isfinite(w).all())):
            raise FloatingPointError(
                f"non-finite state (w or sigma2) detected in chunk "
                f"{ci + 1}/{len(sizes)} (iterations {start + 1}-"
                f"{start + c}); the chain is numerically contaminated and "
                "must be discarded. This should not happen (all systems "
                "are I + PSD by construction) -- please report the data "
                "and hyperparameters."
            )

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
