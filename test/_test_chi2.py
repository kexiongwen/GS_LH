"""Verification for chi2.py and the driver's _accept_chi2 auto-decision.

Rule: on CUDA, both Gamma draws of a sweep (S1 sigma^2 shape a1 + n/2,
S3 lambda shape 2p + a) use the synchronization-free chi-square path iff
max(rel_lam, rel_s2) <= 1e-2, where rel = |round(2*alpha)/2 - alpha| /
alpha bounds the relative bias of the draw's mean and variance.  Refusal
is silent (exact Gamma everywhere); CPU always uses exact Gamma.

The CUDA sections use a monkeypatched forced-refusal to prove, bitwise,
(a) that acceptance really engages the approximation (chain differs from
the exact-Gamma chain) and (b) that refusal really means exact Gamma
(chain identical to the forced-exact chain).  Sections needing a CUDA
device print a skip note without one.
"""
import importlib
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH import GS_LH, gamma_sample, make_chi2_mask
from GS_LH.sigma2_sample import sigma2_sample
from GS_LH.shrinkage_sample import shrinkage

# the package __init__ re-exports the GS_LH FUNCTION, which shadows the
# submodule attribute: `import GS_LH.GS_LH as drv` binds the function,
# and monkeypatching drv._accept_chi2 would be a silent no-op (see README).
drv = importlib.import_module("GS_LH.GS_LH")

dtype = torch.float64
on_cuda = torch.cuda.is_available()
results = []


def check(tag, ok):
    results.append((tag, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}: {tag}")


print("=== 1. _accept_chi2 unit checks (any device) ===")
# defaults: alpha_lam = 1000.001 (rel ~ 1e-6), alpha_s2 = 51 (rel = 0)
check("default hyperparameters accepted (p=500, n=100)",
      drv._accept_chi2(500, 1e-3, 1.0, 100) is True)
# alpha_lam = 10.3: delta = 0.2 -> rel ~ 1.9e-2 > 1e-2
check("tiny fractional lambda shape refused (p=5, a=0.3)",
      drv._accept_chi2(5, 0.3, 1.0, 100) is False)
# alpha_s2 = 5.3: delta = 0.2 -> rel ~ 3.8e-2 > 1e-2
check("tiny fractional sigma2 shape refused (n=10, a1=0.3)",
      drv._accept_chi2(500, 1e-3, 0.3, 10) is False)
# half-integer shapes are exact: rel = 0 whatever the size
check("half-integer shapes exact at any size (p=2, a=0.5)",
      drv._accept_chi2(2, 0.5, 1.5, 3) is True)

print("=== 2. chi-square identity: moment checks (any device) ===")
# integer shape nu/2: chi2 path is EXACTLY Ga(nu/2, rate); compare to theory
torch.manual_seed(0)
N = 200_000
for alpha, rate in [(51.0, 2.3), (124.5, 1.0)]:
    rate_t = torch.tensor(rate, dtype=dtype)
    draws = torch.stack([gamma_sample(alpha, rate_t, use_chi2=True)
                         for _ in range(N)])
    m_t, v_t = alpha / rate, alpha / rate ** 2
    m, v = draws.mean().item(), draws.var().item()
    print(f"  Ga({alpha}, {rate}) via chi2: mean {m:.4f} vs {m_t:.4f} "
          f"({m / m_t - 1:+.2%}), var {v:.4f} vs {v_t:.4f} ({v / v_t - 1:+.2%})")
    check(f"moments within 1% of theory (alpha={alpha})",
          abs(m / m_t - 1) < 0.01 and abs(v / v_t - 1) < 0.01)

# non-half-integer shape: moments land on the ROUNDED shape's theory,
# i.e. the perturbation is exactly the rel the rule bounds
alpha, rate = 51.2, 2.3
alpha_r = round(2 * alpha) / 2
rel = abs(alpha_r - alpha) / alpha
rate_t = torch.tensor(rate, dtype=dtype)
draws = torch.stack([gamma_sample(alpha, rate_t, use_chi2=True)
                     for _ in range(N)])
m = draws.mean().item()
print(f"  Ga({alpha}, {rate}): rounded shape {alpha_r}, rel = {rel:.2e}; "
      f"mean {m:.4f} vs rounded theory {alpha_r / rate:.4f} "
      f"({m / (alpha_r / rate) - 1:+.2%})")
check("rounded-shape theory matched within 1% (rel = "
      f"{rel:.1e} <= tol)", abs(m / (alpha_r / rate) - 1) < 0.01)

print("=== 3. gamma_sample dispatch truth table ===")
rate_t = torch.tensor(2.0, dtype=dtype)
torch.manual_seed(1)
exact = gamma_sample(51.0, rate_t)
torch.manual_seed(1)
from torch.distributions import Gamma
check("eager no opt-in == torch.distributions.Gamma (bitwise)",
      torch.equal(exact, Gamma(51.0, rate_t).sample()))
torch.manual_seed(1)
chi2 = gamma_sample(51.0, rate_t, use_chi2=True)
check("eager opt-in engages the chi-square algorithm (draw differs)",
      not torch.equal(exact, chi2))
try:
    gamma_sample(torch.tensor([51.0, 52.0]), torch.ones(2, dtype=dtype),
                 use_chi2=True)
    check("tensor concentration without mask -> ValueError", False)
except ValueError:
    check("tensor concentration without mask -> ValueError", True)
mask = make_chi2_mask(torch.tensor([51.0, 52.0]))
d = gamma_sample(torch.tensor([51.0, 52.0]), torch.ones(2, dtype=dtype),
                 chi2_mask=mask)
check("tensor concentration with mask works", d.shape == (2,)
      and torch.isfinite(d).all().item())

print("=== 4. CPU: the decision never applies (forced-accept regression) ===")
n, p = 60, 80
torch.manual_seed(3)
Xc = torch.randn(n, p, dtype=dtype)
Yc = torch.randn(n, dtype=dtype)
kw = dict(n_iter=60, burnin=10, method="fast", standardize=False, seed=5)
base = GS_LH(Xc, Yc, **kw)
orig = drv._accept_chi2
drv._accept_chi2 = lambda *a, **k: True   # forced acceptance
try:
    forced = GS_LH(Xc, Yc, **kw)
finally:
    drv._accept_chi2 = orig
same = (torch.equal(base["beta"], forced["beta"])
        and torch.equal(base["sigma2"], forced["sigma2"]))
check("CPU chain bitwise identical even with _accept_chi2 forced True",
      same)

print("=== 5. CUDA sections ===")
if not on_cuda:
    print("  skipped (no CUDA device)")
else:
    dev = torch.device("cuda")

    # shared integration problem
    n, p = 100, 500
    g = torch.Generator().manual_seed(4)
    X = torch.randn(n, p, generator=g, dtype=dtype)
    X = X * (n ** 0.5 / X.norm(dim=0))
    beta_true = torch.zeros(p, dtype=dtype)
    for j, v in {0: 2.0, 1: -1.5, 2: 1.2, 3: 0.9, 4: -0.7}.items():
        beta_true[j] = v
    Y = X @ beta_true + torch.randn(n, generator=g, dtype=dtype)
    Xg, Yg = X.to(dev), Y.to(dev)

    print("  -- 5a. acceptance really engages the approximation --")
    kw = dict(n_iter=200, burnin=50, method="fast", standardize=False,
              seed=42)
    drv._accept_chi2 = lambda *a, **k: False   # forced refusal -> exact
    try:
        ref_exact = GS_LH(Xg, Yg, **kw)
    finally:
        drv._accept_chi2 = orig
    chi2_a = GS_LH(Xg, Yg, **kw)
    chi2_b = GS_LH(Xg, Yg, **kw)
    engaged = not torch.equal(chi2_a["beta"], ref_exact["beta"])
    repro = (torch.equal(chi2_a["beta"], chi2_b["beta"])
             and torch.equal(chi2_a["sigma2"], chi2_b["sigma2"]))
    check("accepted chain != exact-Gamma chain (approximation engaged)",
          engaged)
    check("accepted chain reproducible under same seed", repro)

    print("  -- 5b. refusal really means exact Gamma (bitwise) --")
    # n=12, p=8, a=0.3, a1=0.3: rel_lam ~ 1.2e-2, rel_s2 ~ 3.2e-2 -> refused
    gt = torch.Generator().manual_seed(6)
    Xt = torch.randn(12, 8, generator=gt, dtype=dtype)
    Yt = torch.randn(12, generator=gt, dtype=dtype)
    kwt = dict(n_iter=40, burnin=10, method="fast", standardize=False,
               seed=7, a=0.3, a1=0.3)
    assert drv._accept_chi2(8, 0.3, 0.3, 12) is False
    refused = GS_LH(Xt.to(dev), Yt.to(dev), **kwt)
    drv._accept_chi2 = lambda *a, **k: False
    try:
        refused_forced = GS_LH(Xt.to(dev), Yt.to(dev), **kwt)
    finally:
        drv._accept_chi2 = orig
    same = (torch.equal(refused["beta"], refused_forced["beta"])
            and torch.equal(refused["sigma2"], refused_forced["sigma2"]))
    check("refused chain identical to forced-exact chain", same)

    print("  -- 5c. end-to-end quality with the approximation engaged --")
    out = GS_LH(Xg, Yg, n_iter=2000, burnin=1000, method="fast",
                standardize=False, seed=42)
    bm = out["beta"].mean(0).cpu()
    s2m = out["sigma2"].mean().item()
    signals = {0: 2.0, 1: -1.5, 2: 1.2, 3: 0.9, 4: -0.7}
    recovered = all(abs(bm[j].item() - v) < 0.35
                    for j, v in signals.items())
    noise_max = bm[5:].abs().max().item()
    print(f"  sigma2 mean {s2m:.3f} (true 1.0; ~0.33 expected in p>>n), "
          f"signals {[round(bm[j].item(), 2) for j in signals]}, "
          f"noise max {noise_max:.3f}")
    check("recovery with chi-square engaged", recovered
          and noise_max < 0.2 and s2m > 0.2)

    print("  -- 5d. speed: exact Gamma vs chi-square (informational) --")
    Xs = Xg.float()
    Ys = Yg.float()
    kws = dict(n_iter=1000, burnin=200, method="fast", standardize=False,
               seed=1)
    drv._accept_chi2 = lambda *a, **k: False
    try:
        t0 = time.perf_counter()
        GS_LH(Xs, Ys, **kws)
        t_exact = time.perf_counter() - t0
    finally:
        drv._accept_chi2 = orig
    t0 = time.perf_counter()
    GS_LH(Xs, Ys, **kws)
    t_chi2 = time.perf_counter() - t0
    print(f"  eager exact Gamma: {kws['n_iter'] / t_exact:.0f} it/s   "
          f"eager chi-square: {kws['n_iter'] / t_chi2:.0f} it/s   "
          f"({t_exact / t_chi2 - 1:+.0%})")

print()
failed = [t for t, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed"
      + (f"; FAILED: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
