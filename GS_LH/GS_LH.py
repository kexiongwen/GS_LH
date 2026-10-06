"""
GS_LH.py

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

Chi-square Gamma draws (ported from the BFM factor-model implementation):
torch._standard_gamma synchronizes the host, so on CUDA both Gamma draws
of a sweep (S1 sigma^2, S3 lambda) go through the synchronization-free
identity Ga(nu/2, rate) = chi2(nu)/(2*rate) whenever a one-time up-front
check (_accept_chi2) accepts the half-integer rounding of the shape
(relative moment bias <= 1e-2).  Both shapes -- 2p + a and a1 + n/2 --
are constant across the run, so the check decides every iteration;
(half-)integer shapes give rel = 0, i.e. an exact draw with no
approximation at all.  Refusal is silent (exact Gamma everywhere), and
CPU always uses exact Gamma.  See chi2.py.

CUDA-graph mode (argument `graph`; ported from the same implementation):
with graph=True on a CUDA device (and the chi-square check accepted) the
whole iteration -- S1, S2, S3-S5 -- is captured ONCE as a single CUDA
graph (graph.py) and replayed thereafter, eliminating all per-kernel
launch overhead; the RNG advances normally across replays.  All sampled
systems are I + PSD by construction, so the modules
run cholesky_ex with no per-solve flag checks; numerical safety is
instead enforced by a periodic finiteness audit (`audit_every`).  On any
capture failure the sampler warns and falls back to the eager loop; on
CPU, or when the chi-square check is refused, graph=True simply runs
eagerly (a note is printed when verbose=True in the latter case).
"""

import time
import warnings

import torch

from .beta_sample import beta_sample, beta_sample_direct
from .graph import _graph_sweep_body
from .shrinkage_sample import shrinkage
from .sigma2_sample import sigma2_sample, sigma2_sample_direct

__all__ = ["GS_LH"]


def _accept_chi2(p, a, a1, n, tol=1e-2):
    r"""Decide whether the chi-square Gamma approximation is accurate
    enough for this run (see chi2.py).

    Both Gamma shapes of the sampler are constant across the whole run,
    so this one check decides every iteration:

        S3 lambda : alpha_lam = 2p + a,
                    rel_lam = |round(2*alpha_lam)/2 - alpha_lam| / alpha_lam,
        S1 sigma2 : alpha_s2  = a1 + n/2,
                    rel_s2  = |round(2*alpha_s2)/2 - alpha_s2| / alpha_s2.

    Returns True  -- max(rel_lam, rel_s2) <= tol: the relative bias of
    each Gamma draw's mean and variance is at most tol (an O(1/p) resp.
    O(1/n) perturbation, negligible in the regimes this sampler targets;
    (half-)integer shapes give rel = 0) and the chi-square path is used
    automatically on CUDA, in eager execution too (it avoids the
    host synchronization of torch._standard_gamma);
    Returns False -- otherwise: the chi-square path is disabled
    everywhere and the sampler silently uses exact Gamma.
    """
    alpha_lam = 2 * p + a
    rel_lam = abs(round(2 * alpha_lam) / 2 - alpha_lam) / alpha_lam
    alpha_s2 = a1 + 0.5 * n
    rel_s2 = abs(round(2 * alpha_s2) / 2 - alpha_s2) / alpha_s2
    return max(rel_lam, rel_s2) <= tol


def GS_LH(X, Y, n_iter, burnin=0, a1=1.0, b1=1.0, a=1e-3, b=1e-3,
          method="auto", w0=None, standardize=True, seed=None,
          verbose=False, graph=False, audit_every=50):
    """Run the Bayesian L1/2 Gibbs sampler.

    Parameters
    ----------
    X : torch.Tensor, shape (n, p)
        Design matrix (dense).
    Y : torch.Tensor, shape (n,)
        Response vector.
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
    w0 : torch.Tensor, shape (p,), optional
        Initial state w = tau/lambda^2 (default: vector of ones).
    standardize : bool
        If True (default), center Y, center X columns and rescale them to
        ||X_j||^2 = n, as the model assumes. Back-transformation to the
        original scale:  beta_orig_j = beta_j / x_scale_j,
        intercept = y_mean - x_mean . beta_orig  (stats in "preprocess").
        Set False when X is already standardized and Y centered.
    seed : int, optional
        Seed for torch's global RNG.
    verbose : bool
        Print progress every ~10% of iterations.
    graph : bool
        If True, capture the whole iteration (S1, S2, S3-S5) as ONE CUDA
        graph and replay it (graph.py; large gains for small/launch-bound
        problems; replays advance the RNG normally and a fixed seed
        replays a run bit-exactly).  Capture is a one-time cost per
        GS_LH call (three warm-up iterations plus the recording, excluded
        from runtime_sec, mirroring the JAX port's AOT exclusion), so
        graph=True pays off best for longer chains.  Requirements: X on a
        CUDA device and the chi-square check (_accept_chi2) accepted --
        the exact Gamma draw is not capturable, so a refused check means
        graph=True runs eagerly (a note is printed when verbose=True), as
        does any capture failure (RuntimeWarning) or a non-CUDA device
        (RuntimeWarning).  See the module docstring.
    audit_every : int
        Every this many iterations, verify that w and sigma2 are finite
        (one scalar host sync per audit, amortized cost negligible).  A
        non-finite state signals a numerical anomaly outside the guarded
        range and raises FloatingPointError instead of silently
        contaminating the chain.  Set to 0 to disable.

    Returns
    -------
    dict with keys
        "beta"       (n_iter, p) CPU tensor of posterior beta draws
        "sigma2"     (n_iter,) CPU tensor of posterior sigma^2 draws
        "method"     the algorithm pairing actually used
        "preprocess" {"y_mean", "x_mean", "x_scale"} or None
        "w_final"    last state w (for warm restarts)
        "runtime_sec" wall-clock seconds of the loop

    Notes
    -----
    On CUDA the two Gamma draws of every iteration (S1 sigma^2, S3
    lambda) automatically use the synchronization-free chi-square path of
    chi2.py when the one-time _accept_chi2 check accepts the half-integer
    rounding of their shapes (relative moment bias <= 1e-2; both shapes,
    2p + a and a1 + n/2, are constant across the run, and (half-)integer
    shapes incur no approximation at all).  Refusal is silent -- exact
    Gamma everywhere; CPU always uses exact Gamma.  For (half-)integer
    shapes the chi-square draw is exact -- only the computation changes;
    otherwise each Gamma conditional is perturbed by at most the accepted
    relative bias on its mean and variance.
    """
    if seed is not None:
        torch.manual_seed(seed)

    n, p = X.shape
    dtype, device = X.dtype, X.device
    Y = Y.to(dtype=dtype, device=device)

    # --- resolve algorithm pairing ---
    if method == "auto":
        method = "fast" if p >= n else "direct"
    if method not in ("fast", "direct"):
        raise ValueError(f"unknown method {method!r}")
    if X.layout != torch.strided:
        raise ValueError("sparse X is not supported; pass a dense design matrix")

    # --- standardization (model assumption: Y centered, ||X_j||^2 = n) ---
    preprocess = None
    if standardize:
        y_mean = Y.mean()
        Y = Y - y_mean
        x_mean = X.mean(dim=0)
        X = X - x_mean
        x_scale = X.norm(dim=0) / (n ** 0.5)
        x_scale = torch.where(x_scale > 0, x_scale, torch.ones_like(x_scale))
        X = X / x_scale
        preprocess = {"y_mean": y_mean.item(),
                      "x_mean": x_mean.cpu(), "x_scale": x_scale.cpu()}

    # --- chain state: only w = tau/lambda^2 ---
    if w0 is None:
        w = torch.ones(p, dtype=dtype, device=device)
    else:
        w = w0.to(dtype=dtype, device=device).clone()
        if not bool(torch.isfinite(w).all()) or bool(torch.any(w <= 0)):
            raise ValueError("w0 must be finite and strictly positive")

    # --- chi-square decision (one-time; decides every iteration) ---
    # Accepted -> both Gamma draws of a sweep avoid the host
    # synchronization of torch._standard_gamma on CUDA (eager included);
    # refused -> exact Gamma everywhere, silently.  CPU: always exact.
    chi2_ok = device.type == "cuda" and _accept_chi2(p, a, a1, n)

    total = burnin + n_iter
    step = max(1, total // 10)

    if method == "fast":
        s1_fn, s2_fn = sigma2_sample, beta_sample
        pre1 = pre2 = {}
    else:  # "direct": X'X, X'Y, Y'Y are constant across iterations
        s1_fn, s2_fn = sigma2_sample_direct, beta_sample_direct
        XtX, XtY = X.T @ X, X.T @ Y
        pre1 = {"XtX": XtX, "XtY": XtY, "YtY": torch.dot(Y, Y)}
        pre2 = {"XtX": XtX, "XtY": XtY}

    # --- chain storage -------------------------------------------------------
    # On CUDA runs: page-locked (pinned) buffers so the per-iteration copy
    # can be issued non-blocking and the D2H transfer overlaps the next
    # iteration's computation (one stream synchronization at the end
    # flushes them); if the OS refuses the page-locked allocation we
    # silently fall back to pageable memory (the copies then simply become
    # synchronous again).
    pinned = device.type == "cuda"
    try:
        beta_keep = torch.empty((n_iter, p), dtype=dtype, pin_memory=pinned)
        sigma2_keep = torch.empty(n_iter, dtype=dtype, pin_memory=pinned)
    except RuntimeError:
        pinned = False
        beta_keep = torch.empty((n_iter, p), dtype=dtype)
        sigma2_keep = torch.empty(n_iter, dtype=dtype)

    # --- optional CUDA-graph capture of the whole iteration ------------------
    # Requires CUDA (the graph replays device kernels) and the chi-square
    # check (the exact Gamma draw synchronizes, hence is not capturable).
    use_graph = graph and device.type == "cuda" and chi2_ok
    if graph and device.type != "cuda":
        warnings.warn("graph=True requires a CUDA device; running eagerly",
                      RuntimeWarning)
    elif graph and not chi2_ok and verbose:
        print("GS_LH: graph=True refused (chi-square shape perturbation "
              "> 1e-2; see _accept_chi2); running eagerly")
    g = None
    if use_graph:
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):  # warmup (also validates the body)
                    _graph_sweep_body(X, Y, w, a1, b1, a, b,
                                      s1_fn, s2_fn, pre1, pre2)
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                ow, obeta, os2 = _graph_sweep_body(
                    X, Y, w, a1, b1, a, b, s1_fn, s2_fn, pre1, pre2)
        except Exception as exc:  # capture is best-effort: eager fallback
            warnings.warn(f"CUDA graph capture failed ({type(exc).__name__}: "
                          f"{exc}); falling back to eager", RuntimeWarning)
            g = None

    # --- periodic finiteness audit --------------------------------------------
    # All sampled systems are I + PSD by construction
    # and the modules run cholesky_ex with no per-solve flag checks, so a
    # non-finite state signals a numerical anomaly OUTSIDE the guarded
    # range.  Audit the carried state every `audit_every` iterations (one
    # scalar host sync per audit, amortized cost negligible) and fail
    # loudly rather than silently contaminating the chain.
    def _audit(it, w_t, s2_t):
        if audit_every and (it + 1) % audit_every == 0 and not (
            bool(torch.isfinite(w_t).all()) and bool(torch.isfinite(s2_t).all())
        ):
            raise FloatingPointError(
                f"non-finite state (w or sigma2) detected at iteration "
                f"{it + 1}; the chain is numerically contaminated and must "
                "be discarded. This should not happen (all systems are "
                "I + PSD by construction) -- please report the data and "
                "hyperparameters."
            )

    # --- run -----------------------------------------------------------------
    t_start = time.perf_counter()
    if g is not None:
        for it in range(total):
            g.replay()
            # carry the new state into the static input buffer
            w.copy_(ow)
            if it >= burnin:
                k = it - burnin
                # non_blocking: async D2H into pinned memory, in-stream
                # ordered before the next replay (no-op on CPU targets)
                beta_keep[k].copy_(obeta, non_blocking=True)
                sigma2_keep[k].copy_(os2, non_blocking=True)
            _audit(it, w, os2)
            if verbose and (it + 1) % step == 0:
                rate = (it + 1) / (time.perf_counter() - t_start)
                print(f"[GS_LH] {it + 1}/{total} iterations "
                      f"({rate:.0f} it/s), sigma2 = {os2.item():.4g}")
    else:
        for it in range(total):
            # --- S1 (collapsed sigma^2) and S2 (beta) ---
            # one factorization shared between S1 and S2
            sigma2, L = s1_fn(X, Y, w, a1, b1, return_L=True,
                              use_chi2=chi2_ok, **pre1)
            sigma = sigma2.sqrt()
            beta = s2_fn(X, Y, w, sigma, L=L, **pre2)

            # --- S3-S5: shrinkage parameters; new state w = tau/lambda^2 ---
            w = shrinkage(beta / sigma, a, b, use_chi2=chi2_ok)

            # --- collect after burn-in ---
            if it >= burnin:
                k = it - burnin
                beta_keep[k].copy_(beta, non_blocking=True)
                sigma2_keep[k].copy_(sigma2, non_blocking=True)

            _audit(it, w, sigma2)
            if verbose and (it + 1) % step == 0:
                rate = (it + 1) / (time.perf_counter() - t_start)
                print(f"[GS_LH] {it + 1}/{total} iterations "
                      f"({rate:.0f} it/s), sigma2 = {sigma2.item():.4g}")

    if device.type == "cuda":
        torch.cuda.current_stream().synchronize()  # flush async D2H copies
    runtime = time.perf_counter() - t_start

    out = {
        "beta": beta_keep,
        "sigma2": sigma2_keep,
        "method": method,
        "preprocess": preprocess,
        "w_final": w.detach().cpu(),
        "runtime_sec": runtime,
    }
    return out
