"""
GS_LH -- Gibbs sampler for Bayesian L1/2 regression
(Ke & Fan, 2024, JCGS, DOI: 10.1080/10618600.2024.2374579).

GS_LH       : main driver (steps S1-S5)
beta_sample : S2, beta | ...  (Algorithm 1 of Bhattacharya et al. 2016)
beta_sample_direct : S2 via the p x p Cholesky (Prop. 2.1 = Rue 2001)
sigma2_sample      : S1, collapsed sigma^2 (n x n Cholesky)
sigma2_sample_direct : S1 via the p x p Woodbury form
shrinkage   : S3-S5, lambda / v / tau^2 updates, returns w = tau/lambda^2
inv_gauss   : shape-1 inverse-Gaussian generator (Michael-Schucany-Haas)
chi2        : synchronization-free Gamma draws (chi-square identity) with
              the driver's one-time acceptance check (_accept_chi2)
graph       : CUDA-graph sweep body (graph=True mode of the driver)
"""
from .GS_LH import GS_LH
from .beta_sample import beta_sample, beta_sample_direct
from .chi2 import gamma_sample, make_chi2_mask
from .shrinkage_sample import inv_gauss, shrinkage
from .sigma2_sample import sigma2_sample, sigma2_sample_direct

__all__ = [
    "GS_LH",
    "beta_sample",
    "beta_sample_direct",
    "gamma_sample",
    "inv_gauss",
    "make_chi2_mask",
    "shrinkage",
    "sigma2_sample",
    "sigma2_sample_direct",
]
