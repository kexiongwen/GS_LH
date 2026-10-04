"""
shrinkage_sample.py  (JAX port, single precision)

Gibbs sampler steps S3-S5 for the Bayesian L1/2 regression:
updates of the shrinkage-related parameters
(Ke & Fan, 2024, JCGS, DOI: 10.1080/10618600.2024.2374579).

Implemented with JAX in single precision (float32); every public function
takes an explicit jax.random key as its first argument and coerces its
array inputs to float32. Notation: beta~_j = beta_j / sigma is the
error-standardized coefficient.

S3 (global rate lambda, marginalized over v, tau^2):
    lambda | beta, sigma
        ~ Gamma(2p + a,  b + sum_j sqrt(|beta~_j|)).

S4 (local scale v_j, marginalized over tau^2; sample the reciprocal):
    1/v_j | beta, lambda, sigma
        ~ InvGaussian(1 / (2 * lambda * sqrt(|beta~_j|)),  1/2).

S5 (local scale tau_j^2; sample the reciprocal):
    1/tau_j^2 | beta, lambda, v_j, sigma
        ~ InvGaussian(1 / (lambda^2 * v_j * |beta~_j|),  1 / v_j^2).

Two distributional identities let every draw go through a shape-1
inverse-Gaussian generator only (see Appendix of the accompanying note):

    reciprocal duality : giG(1/2, chi, psi) -> 1/Y ~ InvGaussian(sqrt(psi/chi), psi)
    scaling law        : X ~ InvGaussian(mu, lam0)  =>  cX ~ InvGaussian(c*mu, c*lam0)

Design notes
------------
* S3 -> S4 -> S5 is a marginal-then-conditional chain given (beta, sigma),
  so (lambda, v, tau) need NOT be carried as sampler state; they are
  regenerated from beta~ every iteration inside `shrinkage`.
* `shrinkage` returns w_j = tau_j / lambda^2, the per-coefficient sd scale:
      Var(beta_j | tau, lambda, sigma^2) = sigma^2 * w_j^2.
  Steps S1/S2 then only need S = diag(w^2), i.e.
      V = (X'X + diag(w)^{-2})^{-1},  Sigma = I_n + X diag(w^2) X'.
"""

import jax
import jax.numpy as jnp

__all__ = ["inv_gauss", "shrinkage"]


def inv_gauss(key, mu):
    """Draw from InvGaussian(mean=mu, shape=1), elementwise.

    Michael-Schucany-Haas algorithm specialized to shape lambda0 = 1:
    with nu ~ N(0, 1), y = nu^2, x = mu + mu^2 y/2 - (mu/2) sqrt(4 mu y + mu^2 y^2),
    return x with probability mu/(mu+x), else mu^2/x. Here `a` below is x/mu
    (note (ink+2)^2 - 4 = ink^2 + 4 ink with ink = mu*y), so x = mu*a and
    mu^2/x = mu/a, with acceptance probability 1/(1+a).

    The textbook branch a = 1 + (ink - sqrt(ink^2 + 4 ink))/2 suffers
    catastrophic cancellation for huge ink; we use the identical rationalized
    form
        a = w/(1+w),  w = 2 / (ink * (1 + sqrt(1 + 4/ink))),
    which stays accurate and positive for all ink > 0 in float32.

    ink is floored at 8*tiny: a normal draw can come out an exact 0, and
    ink = 0 makes w NaN via 0 * inf. The floor activates only for
    z^2 < 8*tiny/mu, a region of negligible probability whose draw is
    unchanged in practice (a ~= 1 with or without the floor).

    Parameters
    ----------
    key : jax.random.PRNGKey
    mu : array-like
        Positive means, any shape (coerced to float32).

    Returns
    -------
    jnp.ndarray, float32, same shape as `mu`.
    """
    mu = jnp.asarray(mu, dtype=jnp.float32)
    key_n, key_u = jax.random.split(key)

    ink = mu * jax.random.normal(key_n, mu.shape, dtype=jnp.float32) ** 2
    # Guard the z = 0 draw: 4/ink must stay finite (ink = 0 makes w NaN
    # via 0 * inf).
    ink = jnp.maximum(ink, 8 * jnp.finfo(jnp.float32).tiny)
    w = 2 / (ink * (1 + jnp.sqrt(1 + 4 / ink)))
    a = w / (1 + w)

    u = jax.random.uniform(key_u, mu.shape, dtype=jnp.float32)
    return jnp.where(1 / (1 + a) >= u, mu * a, mu / a)


def shrinkage(key, param, a, b):
    """One pass of steps S3-S5: draw (lambda, v, tau) given beta~ and
    return the composite scale w = tau / lambda^2.

    Parameters
    ----------
    key : jax.random.PRNGKey
    param : jnp.ndarray, shape (p,)
        Error-standardized coefficients beta~ = beta / sigma at the current
        iteration (the driver divides the freshly drawn beta by sigma).
    a, b : float
        Shape and rate of the hyperprior lambda ~ Gamma(a, b).

    Returns
    -------
    jnp.ndarray, shape (p,)
        w_j = tau_j / lambda^2, so that the prior covariance of beta is
        sigma^2 * diag(w^2). Directly plugged into S1/S2 as S = diag(w^2).

    Notes
    -----
    * S4: 1/v_j ~ IG(1/(2 lambda sqrt(|beta~_j|)), 1/2) is drawn as
      X/2 with X ~ IG(1/(lambda sqrt(|beta~_j|)), 1)  (scaling law, c = 1/2).
    * S5: 1/tau_j^2 ~ IG(1/(lambda^2 v_j |beta~_j|), 1/v_j^2) is drawn via
      v_j^2/tau_j^2 ~ IG(v_j/(lambda^2 |beta~_j|), 1)  (scaling law, c = v_j^2),
      hence tau_j = v_j / sqrt(.).
    * Entries of `param` are floored at the smallest positive normal float32
      before the square root, so exact zeros cannot produce 1/0 = inf;
      thanks to the rationalized `inv_gauss` this safeguard is enough to
      keep every downstream draw finite.
    """
    key_lam, key_v, key_tau = jax.random.split(key, 3)
    param = jnp.asarray(param, dtype=jnp.float32)

    # Sample lam (S3): lam ~ Gamma(2p + a, b + sum sqrt(|beta~|))  [shape, rate]
    ink = jnp.sqrt(jnp.maximum(jnp.abs(param), jnp.finfo(jnp.float32).tiny))
    lam = (jax.random.gamma(key_lam, 2 * param.shape[0] + a, dtype=jnp.float32)
           / (jnp.sum(ink) + b))

    ink = ink * lam  # now ink = lam * sqrt(|beta~|)

    # Sample V (S4): 1/v = IG(1/ink, 1) / 2
    v = 2 / inv_gauss(key_v, 1 / ink)

    # Sample tau (S5): v^2/tau^2 = IG(v/ink^2, 1)
    tau = v / jnp.sqrt(inv_gauss(key_tau, v / ink ** 2))

    return tau / lam ** 2
