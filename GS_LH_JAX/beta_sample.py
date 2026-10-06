"""
beta_sample.py  (JAX port, single precision)

Gibbs sampler step S2 for the Bayesian L1/2 regression:
sample the coefficient vector beta conditional on the fresh sigma^2.

Target distribution:
    beta | tau^2, lambda, sigma^2, Y ~ N_p(mu_hat, sigma^2 V),
    V      = (X'X + S^{-1})^{-1},
    mu_hat = V X'Y,
with S = diag(w^2), where w = tau/lambda^2 is the output of
shrinkage_sample.shrinkage (prior covariance = sigma^2 diag(w^2)).

Implemented with JAX in single precision (float32); every public function
takes an explicit jax.random key as its first argument.

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
   p x p system in the congruence-scaled B-form
       B = W(X'X)W + I_p = (XW)'(XW) + I_p,   W = diag(w)
   -- the B-form of A = X'X + diag(w)^{-2} (A = W^{-1} B W^{-1}, so
   beta = W B^{-1} h with h ~ N(W X'Y, sigma^2 B); no reciprocals of w
   anywhere, and B = I + Gram shares the I + PSD structure of
   beta_sample's n x n system, whose nonzero spectrum coincides with
   B's): cost O(n p^2 + p^3), reduced to O(p^3) per draw when the caller
   supplies the precomputed X'X (argument XtX). Algebraically the
   classical Rue (2001) sampler (Cholesky of the p x p posterior
   precision); best when n > p, the mirror image of beta_sample (whose
   costs are driven by n).

Choice guide: p >= n -> beta_sample; n > p -> beta_sample_direct.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

__all__ = ["beta_sample", "beta_sample_direct"]


def beta_sample(key, X, Y, w, sigma, L=None):
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
    key : jax.random.PRNGKey
    X : jnp.ndarray, shape (n, p)
        Design matrix (columns standardized, ||X_j||^2 = n).
    Y : jnp.ndarray, shape (n,)
        Centered response vector.
    w : jnp.ndarray, shape (p,)
        Positive local scales w_j = tau_j / lambda^2 from
        shrinkage_sample.shrinkage; the prior covariance is
        sigma^2 * diag(w^2).
    sigma : float or 0-dim jnp.ndarray
        Error standard deviation (square root of the fresh S1 draw).
    L : jnp.ndarray, shape (n, n), optional
        Precomputed Cholesky factor of M = I_n + X diag(w^2) X' (e.g. the
        one used by the S1 sigma^2 update of the same iteration). If None,
        M is formed and factorized here.

    Returns
    -------
    jnp.ndarray, shape (p,)
        One exact draw from the conditional posterior of beta.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    Y = jnp.asarray(Y, dtype=jnp.float32)
    w = jnp.asarray(w, dtype=jnp.float32)
    key_z, key_d = jax.random.split(key)
    n = X.shape[0]

    # Steps 1-2: u ~ N(0, sigma^2 diag(w^2)), delta ~ N(0, I_n);
    #            v = Phi u + delta = X (w . z1) + delta  (sigma cancels)
    z1 = jax.random.normal(key_z, w.shape, dtype=jnp.float32)
    u = sigma * (w * z1)
    v = X @ (w * z1) + jax.random.normal(key_d, (n,), dtype=jnp.float32)

    # Step 3: solve M omega = Y/sigma - v,  M = I_n + X diag(w^2) X'
    if L is None:
        Xw = X * w                       # column scaling: (Xw)_{ij} = X_{ij} w_j
        M = Xw @ Xw.T                    # = X diag(w^2) X'
        M = M.at[jnp.diag_indices(n)].add(1.0)   # + I_n
        L = jnp.linalg.cholesky(M)
    omega = jsla.cho_solve((L, True), (Y / sigma - v)[:, None])[:, 0]

    # Step 4: beta = u + D Phi' omega = u + sigma diag(w^2) X' omega
    return u + sigma * (w ** 2 * (X.T @ omega))


def beta_sample_direct(key, X, Y, w, sigma, L=None, XtX=None, XtY=None):
    """Draw beta ~ N_p(V X'Y, sigma^2 V), V = (X'X + diag(w)^{-2})^{-1}:
    Proposition 2.1 of Nishimura & Suchard (2022) with a direct Cholesky
    solve of the p x p system in the congruence-scaled B-form.

    With W = diag(w) and B = W(X'X)W + I_p = (XW)'(XW) + I_p (the B-form
    of A = X'X + diag(w)^{-2}: A = W^{-1} B W^{-1}, hence V = A^{-1} =
    W B^{-1} W), generate h ~ N(W X'Y, sigma^2 B) as
        h = W X'Y + sigma W X' eta + sigma delta,
        eta ~ N(0, I_n),  delta ~ N(0, I_p),
    then return beta = W B^{-1} h via the Cholesky factor of B. Since
    W B^{-1} h has mean W B^{-1} W X'Y = V X'Y and covariance
    sigma^2 W B^{-1} W = sigma^2 V, this is exactly the target. It is
    algebraically the classical sampler of Rue (2001) (Cholesky of the
    p x p posterior precision); the congruence scaling only rewrites the
    system, and the h-generation only changes how the Gaussian noise is
    produced, not the output distribution. The B-form needs no
    reciprocals of w anywhere; its system B = I_p + (XW)'(XW) is I + PSD
    by construction (eigenvalues >= 1), the same structure as the fast
    pairing's n x n system I_n + XW^2X' (whose nonzero spectrum
    coincides with B's).

    Cost: O(n p^2) to form the Gram part of B plus O(p^3 / 3) Cholesky --
    best when n > p (mirror image of beta_sample, whose costs are driven
    by n). With a caller-supplied precomputed X'X (argument XtX,
    constant across Gibbs iterations, as supplied by GS_LH) B is an
    O(p^2) rescaling of it and the per-draw cost drops to O(p^3 / 3).

    Parameters
    ----------
    key : jax.random.PRNGKey
    X, Y, w, sigma : see beta_sample.
    L : jnp.ndarray, shape (p, p), optional
        Precomputed Cholesky factor of B = W(X'X)W + I_p. If None,
        B is formed and factorized here.
    XtX, XtY : jnp.ndarray, optional
        Precomputed X'X and X'Y. JAX arrays are immutable: caller-supplied
        XtX is never mutated by the congruence scaling.

    Returns
    -------
    jnp.ndarray, shape (p,)
        One exact draw from the conditional posterior of beta.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    Y = jnp.asarray(Y, dtype=jnp.float32)
    w = jnp.asarray(w, dtype=jnp.float32)
    key_e, key_d = jax.random.split(key)
    n, p = X.shape

    # Proposition 2.1 on the congruence-scaled system:
    # h ~ N(W X'Y, sigma^2 B),  W = diag(w),  B = W(X'X)W + I_p
    eta = jax.random.normal(key_e, (n,), dtype=jnp.float32)
    delta = jax.random.normal(key_d, w.shape, dtype=jnp.float32)
    h = (w * (XtY if XtY is not None else X.T @ Y)
         + sigma * (w * (X.T @ eta)) + sigma * delta)

    # Direct solve: B = W(X'X)W + I_p,  beta = W B^{-1} h
    if L is None:
        B = XtX if XtX is not None else X.T @ X
        B = (w[:, None] * B) * w[None, :]
        B = B.at[jnp.diag_indices(p)].add(1.0)
        L = jnp.linalg.cholesky(B)
    u = jsla.cho_solve((L, True), h[:, None])[:, 0]
    return w * u
