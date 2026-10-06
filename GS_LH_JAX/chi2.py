"""
chi2.py  (JAX port, single precision)

Chi-square approximation of Gamma draws: Ga(alpha, rate) with the shape
rounded to the nearest half-integer (dof nu = round(2*alpha),
|delta| <= 0.25) drawn via the identity

      Ga(nu/2, rate) = chi2(nu) / (2*rate),   chi2(nu) = sum of nu N(0,1)^2,

composed purely of normal draws and arithmetic.  Ported from the BFM
factor-model project (its Chi2.py module), keeping its role change
relative to the PyTorch version of this package:

* In the PyTorch package torch._standard_gamma synchronizes the host,
  so the chi-square path is MANDATORY under CUDA-graph capture and
  measurably faster in eager GPU execution.
* In JAX, jax.random.gamma is a pure XLA primitive -- no host sync, and
  capturable by XLA command buffers (CUDA graphs) as-is -- so the
  chi-square path is never mandatory.  It is an optional
  accuracy-for-speed tradeoff, an O(1/shape) model-level perturbation
  negligible when the shape is large (lambda step O(1/p), sigma2 step
  O(1/n)).  The driver's _accept_chi2 decides whether the rounding
  perturbation is negligible and opts in automatically on the GPU
  backend; everything else uses exact jax.random.gamma.

Mask convention (JAX): under jit the dof is not host-readable, so the
chi-square path takes a precomputed 0/1 mask from make_chi2_mask(); the
mask WIDTH carries the (static) dof.  In this sampler both Gamma shapes
are scalars -- 2p + a for the lambda step (S3) and a1 + n/2 for the
collapsed sigma2 step (S1) -- so the driver builds two (1, nu) masks,
broadcast over the draw, ONCE on host.  (Adaptation vs the reference
module: the chi2 sum is reshaped back to rate.shape, so a (1, nu) mask
over a 0-dim rate still returns a 0-dim draw.)
"""

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["gamma_sample", "make_chi2_mask"]

_F32 = jnp.float32


def make_chi2_mask(concentration) -> jax.Array:
    """0/1 mask (..., nu_max) for chi-square draws with per-element dof.

    Row k has ones in the first nu_k = round(2 * concentration_k) columns
    and zeros elsewhere; nu_max = max_k nu_k.  Host-side computation:
    call ONCE outside jit (the driver builds it together with the
    chi-square decision).
    """
    # round in float64, matching the driver's _accept_chi2 bound: a
    # float32 cast could round 2*shape the other way when it sits within
    # float32-eps of a .5 boundary
    conc = np.asarray(concentration, dtype=np.float64)
    nu = np.round(2.0 * conc)
    nu_max = int(nu.max())
    mask = np.arange(nu_max)[None, :] < nu.reshape(-1)[:, None]
    return jnp.asarray(mask.astype(np.float32))


def gamma_sample(
    key: jax.Array,
    concentration,
    rate: jax.Array,
    chi2_mask: jax.Array | None = None,
) -> jax.Array:
    r"""Draw from Ga(concentration, rate) (shape-rate), elementwise.

    Parameters
    ----------
    key : jax.random.PRNGKey
    concentration : float or array-like
        Shape parameter(s), scalar or broadcastable with `rate`.
    rate : array-like
        Rate parameter(s), broadcastable with the concentration.
    chi2_mask : array-like or None
        Precomputed mask from make_chi2_mask().  None draws EXACT
        Gamma via jax.random.gamma; a mask selects the chi-square
        path with the shape rounded to the nearest half-integer (see
        module docstring).  Unlike the PyTorch version there is no
        capture-detection case: exact Gamma is always capturable in
        JAX, so opting in is the only way to get the approximation.

    Returns
    -------
    jnp.ndarray, float32, broadcast shape of concentration and rate.
    """
    rate = jnp.asarray(rate, dtype=_F32)

    if chi2_mask is None:
        shape = jnp.broadcast_shapes(jnp.shape(concentration), rate.shape)
        return jax.random.gamma(key, concentration, shape=shape,
                                dtype=_F32) / rate

    # chi-square path: chi2 = sum of nu N(0,1)^2, nu from the mask width
    z = jax.random.normal(key, rate.shape + chi2_mask.shape[-1:], _F32)
    chi2 = (z * z * chi2_mask).sum(-1).reshape(rate.shape)
    return chi2 / (2 * rate)
