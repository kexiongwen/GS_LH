"""
graph.py

CUDA-graph-capturable form of one full Gibbs iteration, used by the
``graph=True`` mode of the GS_LH driver (see GS_LH.py).  Ported from the
BFM factor-model project (its graph.py module).

A CUDA graph may only contain a fixed sequence of kernels on static
addresses, with no host-side branches on device values and no host
synchronization.  The three step calls of an iteration are plain module
calls:

* sigma2_sample / beta_sample(_direct) solve systems that are I + PSD
  by construction, via cholesky_ex with no info-flag read -- no host
  check, no fallback branch;
* shrinkage (S3 lambda) and the collapsed sigma^2 update (S1) draw their
  Gammas through chi2.gamma_sample, which switches to the chi-square
  identity automatically under capture.  Both concentrations are scalars
  (2p + a and a1 + n/2) with host-known dof, so no chi-square mask is
  needed and the whole iteration contains no host read at all.

The driver records this iteration once with ``torch.cuda.graph`` and then
replays it every iteration (RNG advances normally across replays),
carrying the state w between the static input buffer and the graph-pool
outputs after each replay, and audits finiteness periodically (its
``audit_every`` argument) instead of checking factorization flags.
"""

from .shrinkage_sample import shrinkage

__all__ = ["_graph_sweep_body"]


def _graph_sweep_body(X, Y, w, a1, b1, a, b, s1_fn, s2_fn, pre1, pre2):
    r"""One Gibbs iteration in CUDA-graph-capturable form.

    Mirrors the driver loop exactly -- S1 (collapsed sigma^2, sharing its
    Cholesky factor with S2), S2 (beta), S3-S5 (shrinkage; the new state
    w = tau/lambda^2) -- by calling the step functions directly (all are
    capture-safe; see module docstring).

    Parameters
    ----------
    X : torch.Tensor, shape (n, p)
        Design matrix (static buffer, on a CUDA device).
    Y : torch.Tensor, shape (n,)
        Response vector (static buffer, likewise).
    w : torch.Tensor, shape (p,)
        Chain state (static buffer; the driver copies the returned w back
        into it between replays).
    a1, b1, a, b : float
        Prior sigma^2 ~ InvGamma(a1, b1) and hyperprior
        lambda ~ Gamma(a, b) parameters.
    s1_fn, s2_fn : callable
        The resolved algorithm pairing: sigma2_sample / beta_sample for
        "fast", sigma2_sample_direct / beta_sample_direct for "direct".
    pre1, pre2 : dict
        Precomputed-argument dicts of the pairing
        ({"XtX", "XtY", "YtY"} / {"XtX", "XtY"} for "direct", empty for
        "fast"); constant tensors, captured by address.

    Returns
    -------
    w, beta, sigma2 : torch.Tensor
        The newly drawn state (graph-pool tensors; the driver copies out
        what it needs before the next replay).
    """
    # S1: collapsed sigma^2 (use_chi2: also chi-square in warmup)
    sigma2, L = s1_fn(X, Y, w, a1, b1, return_L=True, use_chi2=True, **pre1)
    sigma = sigma2.sqrt()
    # S2: beta, reusing S1's Cholesky factor
    beta = s2_fn(X, Y, w, sigma, L=L, **pre2)
    # S3-S5: shrinkage parameters; new state w = tau/lambda^2
    w = shrinkage(beta / sigma, a, b, use_chi2=True)
    return w, beta, sigma2
