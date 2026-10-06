"""Validation of graph=True mode: quality, fallbacks, reproducibility.

Mirrors the BFM factor-model graph tests, adapted to the regression
sampler.  Requires a CUDA device for the GPU sections (they print a skip
note without one); the CPU fallback checks run everywhere.
"""
import importlib
import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH import GS_LH

# the package __init__ re-exports the GS_LH FUNCTION, which shadows the
# submodule attribute: monkeypatching needs the real module (see README).
drv = importlib.import_module("GS_LH.GS_LH")

dtype = torch.float64
on_cuda = torch.cuda.is_available()
results = []


def check(tag, ok):
    results.append((tag, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}: {tag}")


def simulate(n, p, signals, sigma=1.0, seed=0):
    """X columns standardized to ||X_j||^2 = n; sparse true beta."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, p, generator=g, dtype=dtype)
    X = X * (n ** 0.5 / X.norm(dim=0))
    beta_true = torch.zeros(p, dtype=dtype)
    for j, v in signals.items():
        beta_true[j] = v
    Y = X @ beta_true + sigma * torch.randn(n, generator=g, dtype=dtype)
    return X, Y, beta_true


def quality(out, signals):
    bm = out["beta"].mean(0).cpu()
    s2m = out["sigma2"].mean().item()
    recovered = all(abs(bm[j].item() - v) < 0.35 for j, v in signals.items())
    noise_max = bm[max(signals) + 1:].abs().max().item()
    fin = (torch.isfinite(out["beta"]).all().item()
           and torch.isfinite(out["sigma2"]).all().item())
    return recovered, noise_max, s2m, fin, bm


print("=== 1. CPU: graph=True -> warn + eager fallback (bitwise == eager) ===")
n, p = 60, 80
Xc, Yc, _ = simulate(n, p, {0: 1.5}, sigma=1.0, seed=3)
kw = dict(n_iter=60, burnin=10, method="fast", standardize=False, seed=5)
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")
    out_g = GS_LH(Xc, Yc, graph=True, **kw)
warned = any("CUDA device" in str(w.message) for w in ws)
out_e = GS_LH(Xc, Yc, **kw)
same = (torch.equal(out_g["beta"], out_e["beta"])
        and torch.equal(out_g["sigma2"], out_e["sigma2"]))
check("RuntimeWarning issued", warned)
check("ran to completion", out_g["beta"].shape[0] == 60)
check("CPU fallback chain bitwise identical to eager", same)

print("=== 2. GPU sections ===")
if not on_cuda:
    print("  skipped (no CUDA device)")
else:
    dev = torch.device("cuda")
    signals = {0: 2.0, 1: -1.5, 2: 1.2, 3: 0.9, 4: -0.7}
    X, Y, _ = simulate(100, 500, signals, sigma=1.0, seed=4)
    Xg, Yg = X.to(dev), Y.to(dev)

    print("  -- 2a. eager vs graph, quality + speed (fast pairing) --")
    outs = {}
    for name, gkw in [("eager", {}), ("graph", {"graph": True})]:
        t0 = time.perf_counter()
        outs[name] = GS_LH(Xg, Yg, n_iter=2000, burnin=1000, method="fast",
                           standardize=False, seed=42, **gkw)
        el = time.perf_counter() - t0
        rec, nm, s2m, fin, bm = quality(outs[name], signals)
        print(f"  {name:6s}: {3000 / el:6.0f} it/s  finite={fin}  "
              f"sigma2={s2m:.3f}  signals={[round(bm[j].item(), 2) for j in signals]}")
        check(f"{name}: recovery", rec and nm < 0.2 and s2m > 0.2 and fin)
    dmax = (outs["eager"]["beta"].mean(0) - outs["graph"]["beta"].mean(0)) \
        .abs().max().item()
    print(f"  max |posterior mean diff| eager vs graph = {dmax:.4f}")
    check("graph statistically equivalent to eager (max diff < 0.05)",
          dmax < 0.05)

    print("  -- 2b. graph with the direct pairing --")
    Xd, Yd, _ = simulate(500, 100, {0: 1.5, 1: -1.0, 2: 0.8}, sigma=0.8,
                         seed=5)
    out_d = GS_LH(Xd.to(dev), Yd.to(dev), n_iter=1500, burnin=500,
                  method="direct", standardize=False, seed=7, graph=True)
    rec, nm, s2m, fin, bm = quality(out_d, {0: 1.5, 1: -1.0, 2: 0.8})
    print(f"  direct graph: finite={fin}  sigma2={s2m:.3f}  "
          f"signals={[round(bm[j].item(), 2) for j in [0, 1, 2]]}")
    check("direct pairing: recovery under graph", rec and nm < 0.2 and fin)

    print("  -- 2c. reproducibility: same seed bitwise, different seed differs --")
    kw = dict(n_iter=200, burnin=50, method="fast", standardize=False,
              graph=True)
    o1 = GS_LH(Xg, Yg, seed=123, **kw)
    o2 = GS_LH(Xg, Yg, seed=123, **kw)
    o3 = GS_LH(Xg, Yg, seed=124, **kw)
    check("two graph runs, same seed: bitwise identical",
          torch.equal(o1["beta"], o2["beta"])
          and torch.equal(o1["sigma2"], o2["sigma2"]))
    check("different seed differs", not torch.equal(o1["beta"], o3["beta"]))

    print("  -- 2d. refused chi-square check -> graph silently refused --")
    # n=12, p=8, a=0.3, a1=0.3: rel_lam ~ 1.2e-2, rel_s2 ~ 3.2e-2 -> refused
    Xt, Yt, _ = simulate(12, 8, {0: 1.0}, sigma=1.0, seed=6)
    kwt = dict(n_iter=40, burnin=10, method="fast", standardize=False,
               a=0.3, a1=0.3, seed=7)
    assert drv._accept_chi2(8, 0.3, 0.3, 12) is False
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        refused = GS_LH(Xt.to(dev), Yt.to(dev), graph=True, **kwt)
    orig = drv._accept_chi2
    drv._accept_chi2 = lambda *a, **k: False
    try:
        forced = GS_LH(Xt.to(dev), Yt.to(dev), **kwt)  # eager, exact Gamma
    finally:
        drv._accept_chi2 = orig
    same = (torch.equal(refused["beta"], forced["beta"])
            and torch.equal(refused["sigma2"], forced["sigma2"]))
    check("no warnings (silent refusal)", len(ws) == 0)
    check("refused graph chain identical to forced-exact eager chain", same)

    print("  -- 2e. capture failure -> warn + eager fallback --")
    def _boom(*args):
        raise RuntimeError("synthetic capture failure")
    orig_body = drv._graph_sweep_body
    drv._graph_sweep_body = _boom
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            out_f = GS_LH(Xg, Yg, n_iter=60, burnin=10, method="fast",
                          standardize=False, seed=11, graph=True)
    finally:
        drv._graph_sweep_body = orig_body
    warned = any("graph capture failed" in str(w.message) for w in ws)
    fin = (torch.isfinite(out_f["beta"]).all().item()
           and torch.isfinite(out_f["sigma2"]).all().item())
    check("RuntimeWarning mentions the capture failure", warned)
    check("fallback ran to completion, finite draws",
          out_f["beta"].shape[0] == 60 and fin)

    print("  -- 2f. audit smoke (audit_every=10 and 0 under graph) --")
    out_a = GS_LH(Xg, Yg, n_iter=100, burnin=20, method="fast",
                  standardize=False, seed=13, graph=True, audit_every=10)
    out_a0 = GS_LH(Xg, Yg, n_iter=100, burnin=20, method="fast",
                   standardize=False, seed=13, graph=True, audit_every=0)
    check("audit_every=10 runs clean under graph",
          torch.isfinite(out_a["beta"]).all().item())
    check("audit_every=0 (disabled) runs clean under graph",
          torch.isfinite(out_a0["beta"]).all().item())

print()
failed = [t for t, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed"
      + (f"; FAILED: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
