"""
chi2.py

Gamma-distribution draws without host synchronization (ported from the
BFM factor-model project -- its Chi2.py module).

torch.distributions.Gamma/Chi2 sample through torch._standard_gamma, whose
CUDA kernel synchronizes the host.  That synchronization both invalidates
CUDA stream capture (cudaErrorStreamCaptureInvalidated) and costs a
measurable share of an eager GPU iteration (~15% of a sweep in the
factor-model sibling project).  gamma_sample() therefore offers a
chi-square path in two situations:

* under CUDA-graph capture (detected via
  torch.cuda.is_current_stream_capturing()) -- mandatory;
* in eager execution when explicitly opted in (a tensor concentration
  given together with `chi2_mask`, or a scalar concentration with
  use_chi2=True) -- e.g. when the driver's _accept_chi2 decision accepts
  the rounding perturbation.

The chi-square path uses the identity

      Ga(nu/2, rate) = chi2(nu) / (2*rate),   chi2(nu) = sum of nu N(0,1)^2,

composed purely of synchronisation-free primitives (randn / square /
sum).  The shape is rounded to the nearest half-integer (dof
nu = round(2*shape), |delta| <= 0.25): an O(1/shape) model-level
approximation, negligible when the shape is large.  In this sampler both
Gamma shapes are run-time constants -- 2p + a for the lambda step (S3)
and a1 + n/2 for the collapsed sigma2 step (S1) -- so the relative
perturbation is O(1/p) resp. O(1/n), and the driver's one-time
_accept_chi2 check (rel <= 1e-2) decides for the whole run.  With the
default hyperparameters the sigma2 shape a1 + n/2 = 1 + n/2 is always
(half-)integer -- the draw is then exact, the same distribution computed
without synchronization -- and the lambda shape 2p + a = 2p + 1e-3 is
perturbed only at the ~1e-6 level.  Anything else uses the exact
torch.distributions.Gamma(concentration, rate).sample().

For a TENSOR concentration (varying dof per element) the 0/1 chi-square
mask must be precomputed OUTSIDE capture with make_chi2_mask() and
passed in, because sizing the mask needs host-side integers and reading
device values synchronizes.  A scalar (python float) concentration needs
no mask: its dof is host-known by construction.  (Both Gamma draws of
this sampler have scalar concentrations; the tensor path is kept for
parity with the reference module.)

Implementation: PyTorch-based, functional style.
"""

import torch
from torch.distributions import Gamma

__all__ = ["gamma_sample", "make_chi2_mask"]


def make_chi2_mask(concentration: torch.Tensor) -> torch.Tensor:
    """0/1 mask (K, nu_max) for chi-square draws with per-element dof.

    Row k has ones in the first nu_k = round(2 * concentration_k) columns
    and zeros elsewhere.  Must be called OUTSIDE CUDA-graph capture (it
    reads the concentration values on host).
    """
    nu = torch.round(2 * concentration)
    nu_max = int(nu.max())
    mask = torch.arange(nu_max, device=concentration.device)[None, :] < nu[:, None]
    return mask.to(concentration.dtype)


def gamma_sample(concentration, rate, chi2_mask=None, use_chi2=False):
    r"""Draw from Ga(concentration, rate) (shape-rate), elementwise.

    Parameters
    ----------
    concentration : float or torch.Tensor
        Shape parameter(s).  A python float (or any non-tensor) needs no
        mask for the chi-square path; a tensor concentration requires
        `chi2_mask` from make_chi2_mask().
    rate : torch.Tensor
        Rate parameter(s), same shape as (or broadcastable with) the
        concentration; the chi-square path requires matching shapes.
    chi2_mask : torch.Tensor or None
        Precomputed mask from make_chi2_mask(concentration) for a tensor
        concentration.  Passing it also OPTS IN to the chi-square path in
        eager execution (it is mandatory under CUDA-graph capture).
    use_chi2 : bool
        Opt in to the chi-square path in eager execution for a scalar
        (non-tensor) concentration, which needs no mask.

    Returns
    -------
    torch.Tensor
        Exact Gamma draw when executing eagerly without opt-in; chi-square
        draw with the shape rounded to the nearest half-integer (see
        module docstring) when capturing OR when opted in.  Truth table:

            state                       tensor conc.    scalar conc.
            eager, no opt-in            exact Gamma     exact Gamma
            eager, mask / use_chi2      chi-square      chi-square
            capture, with mask          chi-square      chi-square
            capture, no mask            ValueError      chi-square
    """
    if not (torch.cuda.is_current_stream_capturing() or use_chi2
            or chi2_mask is not None):
        return Gamma(concentration, rate).sample()

    # ---- chi-square path (capture or explicitly enabled) ---------------------
    if torch.is_tensor(concentration):
        if chi2_mask is None:
            raise ValueError(
                "gamma_sample in the chi-square path with a tensor "
                "concentration requires a precomputed chi2_mask from "
                "make_chi2_mask(concentration)"
            )
        z = torch.randn(*concentration.shape, chi2_mask.size(1),
                        dtype=rate.dtype, device=rate.device)
        chi2 = (z.square() * chi2_mask).sum(-1)
    else:
        nu = int(round(2 * concentration))  # host-side, no sync
        z = torch.randn(*rate.shape, nu, dtype=rate.dtype, device=rate.device)
        chi2 = z.square().sum(-1)
    return chi2 / (2 * rate)
