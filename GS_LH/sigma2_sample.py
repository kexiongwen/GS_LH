"""
sigma2_sample.py

Gibbs sampler step S1 for the Bayesian L1/2 regression:
collapsed update of the error variance sigma^2 (beta integrated out) --
the scheme of the accompanying note:
    sigma^2 | tau^2, lambda, Y
        ~ InvGamma(a1 + n/2,  b1 + q/2),   q = Y' Sigma^{-1} Y,
    Sigma = I_n + X S X',   S = diag(w^2),
with w = tau/lambda^2 the output of shrinkage_sample.shrinkage.

WARNING: sigma^2 estimation in the p >> n regime
------------------------------------------------
In experiments (n = 100, p = 500, sigma = 1) the sampler was found to
underestimate sigma^2 severely when the prior is near-Jeffreys (a1 ~ 0):
the posterior itself concentrates on near-interpolating solutions,

    sigma^2  ~  RSS / n  (~= 0.004 << sigma^2_true).

Mechanism: (i) the p >> n design provides p - n interpolation directions;
(ii) with the error-standardized prior (beta_j = sigma * beta~_j) the
effective penalty lambda sqrt(|beta~_j|) rescales perfectly as sigma -> 0,
so the beta-prior offers no resistance; (iii) the near-Jeffreys InvGamma
density (sigma^2)^{a1-1} ~ (sigma^2)^{-1} piles mass at 0 and offers no
resistance either. This happens for the L1/2 prior AND for the DL prior
with fixed rate (which collapses even faster -- the learned lambda drifts
upward and mildly resists), and beta point estimates are unaffected
throughout. It is a model property, not a sampler defect: the standard
conditional (Park-Casella style) update sigma^2 | beta, w, Y, being an
exact conditional of the same joint posterior, lands in the same
degenerate region.

Mitigation (the effective lever is the sigma^2 prior): use an InvGamma
prior that vanishes at the origin, a1 >= 1. In the experiment above the
posterior mean of sigma^2 recovered monotonically with prior strength:
0.018 (a1 = b1 = 1e-3), 0.33 (1,1), 0.41 (2,2), 0.54 (5,5), 0.65 (10,10);
GS_LH therefore defaults to a1 = b1 = 1. Full anchoring in p >> n
requires a fairly informative sigma^2 prior or external knowledge of
sigma^2.

Two efficient ways to compute the quadratic form q, mirroring the two
beta samplers of beta_sample.py; each factorization can be SHARED with
the S2 beta update of the same Gibbs iteration through the L arguments:

1. sigma2_sample -- Cholesky of the n x n marginal covariance Sigma:
       q = ||L^{-1} Y||^2,   Sigma = L L'.
   Cost O(n^2 p + n^3); best when p >= n. The same L is reused by
   beta_sample (Algorithm 1 of Bhattacharya et al., 2016), so ONE
   factorization per iteration covers S1 + S2. Numerically cleanest:
   q is a plain sum of squares, no subtractive cancellation.

2. sigma2_sample_direct -- Woodbury form on the p x p system in the
   congruence-scaled B-form:
       q = Y'Y - (Wg)' B^{-1} (Wg),   g = X'Y,   W = diag(w),
       B = W(X'X)W + I_p = (XW)'(XW) + I_p.
   Cost O(n p^2 + p^3), or O(p^3) per iteration after a one-time
   O(n p^2) precompute of X'X (arguments XtX/XtY/YtY; B = W(X'X)W + I
   is then an O(p^2) rescaling); best when n > p.  The Cholesky factor
   of B is reused by beta_sample_direct. Subtractive cancellation is
   possible when q << Y'Y, so q is clamped at 0.

   Algebraic note: the B-form is the congruence-scaled equivalent of
   the canonical Rue (2001) parametrization A = X'X + W^{-2} (posterior
   precision = X'X + prior precision): A = W^{-1} B W^{-1}, hence
   g' A^{-1} g = (Wg)' B^{-1} (Wg) and beta = A^{-1} b = W B^{-1} (Wb).
   The two forms are cost-equivalent and accuracy-equivalent in
   practice (componentwise-stable solves leave the error dominated by
   the shared X'X formation); the B-form is implemented because every
   quantity in it is polynomial in w -- no reciprocals of w anywhere in
   the S1/S2 pairing -- and B = I_p + (XW)'(XW) shares the I + PSD
   structure (eigenvalues >= 1) of the fast pairing's M = I_n + XW^2X',
   whose nonzero spectrum coincides with B's.  (kappa(B) grows like
   n * max_j w_j^2 versus kappa(A) ~ kappa(X'X) in the sampler's sparse
   regime -- irrelevant at the accuracies involved, but worth knowing
   for adversarial near-collinear designs.)

Note: X'X, X'Y and Y'Y are constant across the whole Gibbs run; the
direct (p x p) path accepts them precomputed via the XtX / XtY / YtY
arguments (GS_LH does this automatically). XtX is cloned before the
congruence scaling, so caller-supplied matrices are never mutated.
"""

import torch

from .chi2 import gamma_sample

__all__ = ["sigma2_sample", "sigma2_sample_direct"]


def _invgamma_sample(concentration, rate, use_chi2=False):
    """Draw from InvGamma(alpha, gamma) with density
    f(y) proportional to y^{-alpha-1} exp(-gamma / y):
    reciprocal of a Gamma(alpha, rate=gamma) draw.

    use_chi2 opts in to the chi-square Gamma path (see chi2.py; the
    scalar concentration needs no mask).
    """
    return 1 / gamma_sample(concentration, rate, use_chi2=use_chi2)


def sigma2_sample(X, Y, w, a1, b1, L=None, return_L=False, use_chi2=False):
    """S1 update via the n x n Cholesky of Sigma = I_n + X diag(w^2) X'
    (method 1 above; default for p >= n).

    Parameters
    ----------
    X : torch.Tensor, shape (n, p)
        Design matrix (dense).
    Y : torch.Tensor, shape (n,)
        Centered response vector.
    w : torch.Tensor, shape (p,)
        Positive local scales w_j = tau_j / lambda^2.
    a1, b1 : float
        Shape and rate of the prior sigma^2 ~ InvGamma(a1, b1).
    L : torch.Tensor, shape (n, n), optional
        Precomputed Cholesky factor of Sigma. If None, Sigma is formed
        and factorized here.
    return_L : bool
        If True, return (sigma2, L) so the driver can pass L to
        beta_sample for the S2 update of the same iteration.
    use_chi2 : bool
        Opt in to the chi-square path for the Gamma draw (see chi2.py;
        the scalar concentration a1 + n/2 needs no mask).  The GS_LH
        driver enables this on CUDA when its one-time _accept_chi2 check
        accepts the rounding perturbation.

    Returns
    -------
    sigma2 : 0-dim torch.Tensor  (or (sigma2, L) if return_L=True)
    """
    n = X.shape[0]

    if L is None:
        Xw = X * w                       # column scaling: (Xw)_{ij} = X_{ij} w_j
        M = Xw @ Xw.T                    # = X diag(w^2) X'
        M.diagonal().add_(1.0)           # + I_n  ->  M = Sigma
        # cholesky_ex without reading info: M = I + PSD by construction,
        # so no per-solve host check (keeps the function CUDA-graph
        # capturable and synchronization-free; the raising cholesky would
        # sync on the info flag).  The GS_LH driver audits finiteness of
        # the chain periodically (audit_every).
        L = torch.linalg.cholesky_ex(M)[0]

    # q = Y' Sigma^{-1} Y = ||L^{-1} Y||^2
    z = torch.linalg.solve_triangular(L, Y.unsqueeze(-1), upper=False).squeeze(-1)
    q = torch.dot(z, z)

    sigma2 = _invgamma_sample(a1 + n / 2, b1 + q / 2, use_chi2=use_chi2)
    if return_L:
        return sigma2, L
    return sigma2


def sigma2_sample_direct(X, Y, w, a1, b1, L=None, return_L=False,
                         XtX=None, XtY=None, YtY=None, use_chi2=False):
    """S1 update via the p x p Woodbury form in the congruence-scaled
    B-form (method 2 above; best for n > p; companion of
    beta_sample_direct).

    q = Y'Y - (Wg)' B^{-1} (Wg),  g = X'Y,  W = diag(w),
    B = W(X'X)W + I_p.

    Parameters: see sigma2_sample; L is the Cholesky factor of B (p x p),
    shared with beta_sample_direct. return_L returns (sigma2, L).
    XtX, XtY, YtY : torch.Tensor, optional
        Precomputed X'X, X'Y, Y'Y (constant across Gibbs iterations), as
        supplied by GS_LH; skips their O(n p^2) / O(np) recomputation.
        XtX is cloned before the congruence scaling and never mutated.
    use_chi2 : bool
        See sigma2_sample.
    """
    n = X.shape[0]

    if L is None:
        B = XtX.clone() if XtX is not None else X.T @ X
        B.mul_(w[:, None]).mul_(w[None, :])   # W (X'X) W
        B.diagonal().add_(1.0)                # + I_p  ->  B = W(X'X)W + I
        # cholesky_ex without reading info: B = I_p + (XW)'(XW) is I + PSD
        # by construction (eigenvalues >= 1), so no per-solve host check
        # (CUDA-graph capturable, synchronization-free).  The GS_LH driver
        # audits finiteness of the chain periodically (audit_every).
        L = torch.linalg.cholesky_ex(B)[0]

    g = XtY if XtY is not None else X.T @ Y
    Wg = w * g
    s = torch.cholesky_solve(Wg.unsqueeze(-1), L).squeeze(-1)
    # q = Y'Y - (Wg)' B^{-1} (Wg) >= 0 mathematically; clamp guards roundoff
    q = ((YtY if YtY is not None else torch.dot(Y, Y))
         - torch.dot(Wg, s)).clamp_min(0)

    sigma2 = _invgamma_sample(a1 + n / 2, b1 + q / 2, use_chi2=use_chi2)
    if return_L:
        return sigma2, L
    return sigma2

