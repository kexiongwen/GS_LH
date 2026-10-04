"""
beta_sample.py

Gibbs sampler step S2 for the Bayesian L1/2 regression:
sample the coefficient vector beta conditional on the fresh sigma^2.

Target distribution:
    beta | tau^2, lambda, sigma^2, Y ~ N_p(mu_hat, sigma^2 V),
    V      = (X'X + S^{-1})^{-1},
    mu_hat = V X'Y,
with S = diag(w^2), where w = tau/lambda^2 is the output of
shrinkage_sample.shrinkage (prior covariance = sigma^2 diag(w^2)).

Two samplers of the same target:

1. beta_sample -- Algorithm 1 of Bhattacharya, Chakraborty & Mallick (2016),
   "Fast sampling with Gaussian scale mixture priors in high-dimensional
   regression", Biometrika 103, 985-991, doi:10.1093/biomet/asw042.
   Data augmentation + one n x n linear system: O(n^2 p + n^3), exact,
   best when p >> n. Its system matrix I_n + X S X' coincides with the
   marginal covariance of the collapsed sigma^2 update (S1), so one
   Cholesky factor can be shared between S1 and S2 (argument `L`).

2. beta_sample_direct -- Proposition 2.1 of Nishimura & Suchard (2022),
   "Prior-Preconditioned Conjugate Gradient Method for Accelerated Gibbs
   Sampling in 'Large n, Large p' Bayesian Sparse Regression", JASA,
   doi:10.1080/01621459.2022.2057859, with a direct Cholesky solve of the
   p x p system A = X'X + diag(w)^{-2}: cost O(n p^2 + p^3), reduced to
   O(p^3) per draw when the caller supplies the precomputed X'X (argument
   XtX). Algebraically the classical Rue (2001) sampler (Cholesky of the
   p x p posterior precision); best when n > p, the mirror image of
   beta_sample (whose costs are driven by n).

Choice guide: p >= n -> beta_sample; n > p -> beta_sample_direct.
"""

import torch

__all__ = ["beta_sample", "beta_sample_direct"]


def beta_sample(X, Y, w, sigma, L=None):
    """Draw beta ~ N_p(V X'Y, sigma^2 V), V = (X'X + diag(w)^{-2})^{-1},
    via Algorithm 1 of Bhattacharya, Chakraborty & Mallick (2016).

    Algorithm 1 applied with Phi = X/sigma, D = sigma^2 S, alpha = Y/sigma
    (then (Phi'Phi + D^{-1})^{-1} = sigma^2 V, Sigma Phi'alpha = V X'Y):

        Step 1  u = sigma (w . z1),  z1 ~ N(0, I_p);   delta ~ N(0, I_n)
        Step 2  v = X (w . z1) + delta          (sigma cancels in Phi u)
        Step 3  solve (I_n + X S X') omega = Y/sigma - v
        Step 4  beta = u + sigma S X' omega

    Parameters
    ----------
    X : torch.Tensor, shape (n, p)
        Design matrix (columns standardized, ||X_j||^2 = n).
    Y : torch.Tensor, shape (n,)
        Centered response vector.
    w : torch.Tensor, shape (p,)
        Positive local scales w_j = tau_j / lambda^2 from
        shrinkage_sample.shrinkage; the prior covariance is
        sigma^2 * diag(w^2).
    sigma : float or 0-dim torch.Tensor
        Error standard deviation (square root of the fresh S1 draw).
    L : torch.Tensor, shape (n, n), optional
        Precomputed Cholesky factor of M = I_n + X diag(w^2) X' (e.g. the
        one used by the S1 sigma^2 update of the same iteration). If None,
        M is formed and factorized here.

    Returns
    -------
    torch.Tensor, shape (p,)
        One exact draw from the conditional posterior of beta.
    """
    n = X.shape[0]

    # Steps 1-2: u ~ N(0, sigma^2 diag(w^2)), delta ~ N(0, I_n);
    #            v = Phi u + delta = X (w . z1) + delta  (sigma cancels)
    z1 = torch.randn_like(w)
    u = sigma * (w * z1)
    v = X @ (w * z1) + torch.randn(n, dtype=X.dtype, device=X.device)

    # Step 3: solve M omega = Y/sigma - v,  M = I_n + X diag(w^2) X'
    if L is None:
        Xw = X * w                       # column scaling: (Xw)_{ij} = X_{ij} w_j
        M = Xw @ Xw.T                    # = X diag(w^2) X'
        M.diagonal().add_(1.0)           # + I_n
        L = torch.linalg.cholesky(M)
    omega = torch.cholesky_solve((Y / sigma - v).unsqueeze(-1), L).squeeze(-1)

    # Step 4: beta = u + D Phi' omega = u + sigma diag(w^2) X' omega
    return u + sigma * (w.square() * (X.T @ omega))


def beta_sample_direct(X, Y, w, sigma, L=None, XtX=None, XtY=None):
    """Draw beta ~ N_p(V X'Y, sigma^2 V), V = (X'X + diag(w)^{-2})^{-1}:
    Proposition 2.1 of Nishimura & Suchard (2022) with a direct Cholesky
    solve of the p x p system.

    On the sigma^2-scaled system A = X'X + diag(w)^{-2}, generate
    b ~ N(X'Y, sigma^2 A) as
        b = X'Y + sigma X' eta + sigma w^{-1} . delta,
        eta ~ N(0, I_n),  delta ~ N(0, I_p),
    then return A^{-1} b via
    the Cholesky factor of A. Since A^{-1} b has mean V X'Y and covariance
    sigma^2 A^{-1} = sigma^2 V, this is exactly the target. It is
    algebraically the classical sampler of Rue (2001) (Cholesky of the
    p x p posterior precision); the b-generation only changes how the
    Gaussian noise is produced, not the output distribution.

    Cost: O(n p^2) to form A = X'X plus O(p^3 / 3) Cholesky -- best when
    n > p (mirror image of beta_sample, whose costs are driven by n).
    With a caller-supplied precomputed X'X (argument XtX, constant across
    Gibbs iterations, as supplied by GS_LH) the per-draw cost drops to
    O(p^3 / 3).

    Parameters
    ----------
    X, Y, w, sigma : see beta_sample.
    L : torch.Tensor, shape (p, p), optional
        Precomputed Cholesky factor of A = X'X + diag(w)^{-2}. If None,
        A is formed and factorized here.
    XtX, XtY : torch.Tensor, optional
        Precomputed X'X and X'Y. XtX is cloned before the diagonal
        update and never mutated.

    Returns
    -------
    torch.Tensor, shape (p,)
        One exact draw from the conditional posterior of beta.
    """
    n, p = X.shape

    # Proposition 2.1 on the scaled system: b ~ N(X'Y, sigma^2 A)
    eta = torch.randn(n, dtype=X.dtype, device=X.device)
    delta = torch.randn_like(w)
    b = ((XtY if XtY is not None else X.T @ Y)
         + sigma * (X.T @ eta) + sigma * (delta / w))

    # Direct solve: A = X'X + diag(w)^{-2}, beta = A^{-1} b
    if L is None:
        A = XtX.clone() if XtX is not None else X.T @ X
        A.diagonal().add_(w.pow(-2))
        L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
