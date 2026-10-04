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
"""

import time

import torch

from .beta_sample import beta_sample, beta_sample_direct
from .shrinkage_sample import shrinkage
from .sigma2_sample import sigma2_sample, sigma2_sample_direct

__all__ = ["GS_LH"]


def GS_LH(X, Y, n_iter, burnin=0, a1=1.0, b1=1.0, a=1e-3, b=1e-3,
          method="auto", w0=None, standardize=True, seed=None,
          verbose=False):
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

    Returns
    -------
    dict with keys
        "beta"       (n_iter, p) CPU tensor of posterior beta draws
        "sigma2"     (n_iter,) CPU tensor of posterior sigma^2 draws
        "method"     the algorithm pairing actually used
        "preprocess" {"y_mean", "x_mean", "x_scale"} or None
        "w_final"    last state w (for warm restarts)
        "runtime_sec" wall-clock seconds of the loop
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
        if torch.any(w <= 0):
            raise ValueError("w0 must be strictly positive")

    total = burnin + n_iter
    beta_keep = torch.empty((n_iter, p), dtype=dtype)      # stored on CPU
    sigma2_keep = torch.empty(n_iter, dtype=dtype)
    step = max(1, total // 10)

    if method == "fast":
        s1_fn, s2_fn = sigma2_sample, beta_sample
        pre1 = pre2 = {}
    else:  # "direct": X'X, X'Y, Y'Y are constant across iterations
        s1_fn, s2_fn = sigma2_sample_direct, beta_sample_direct
        XtX, XtY = X.T @ X, X.T @ Y
        pre1 = {"XtX": XtX, "XtY": XtY, "YtY": torch.dot(Y, Y)}
        pre2 = {"XtX": XtX, "XtY": XtY}

    t_start = time.perf_counter()
    for it in range(total):
        # --- S1 (collapsed sigma^2) and S2 (beta) ---
        # one factorization shared between S1 and S2
        sigma2, L = s1_fn(X, Y, w, a1, b1, return_L=True, **pre1)
        sigma = sigma2.sqrt()
        beta = s2_fn(X, Y, w, sigma, L=L, **pre2)

        # --- S3-S5: shrinkage parameters; new state w = tau/lambda^2 ---
        w = shrinkage(beta / sigma, a, b)

        # --- collect after burn-in ---
        if it >= burnin:
            k = it - burnin
            beta_keep[k] = beta.cpu()
            sigma2_keep[k] = sigma2.cpu()

        if verbose and (it + 1) % step == 0:
            rate = (it + 1) / (time.perf_counter() - t_start)
            print(f"[GS_LH] {it + 1}/{total} iterations "
                  f"({rate:.0f} it/s), sigma2 = {sigma2.item():.4g}")

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
