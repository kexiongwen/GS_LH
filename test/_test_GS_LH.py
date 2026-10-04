"""End-to-end verification for the GS_LH driver."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH import GS_LH

dtype = torch.float64


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


def test_recovery():
    print("=== Test 1: recovery (method='fast') ===")
    n, p = 100, 500
    signals = {0: 2.0, 1: -1.5, 2: 1.2, 3: 0.9, 4: -0.7}
    X, Y, bt = simulate(n, p, signals, sigma=1.0, seed=1)
    out = GS_LH(X, Y, n_iter=4000, burnin=1000, method="fast",
                standardize=False, seed=42)
    bm, bs = out["beta"].mean(0), out["beta"].std(0)
    s2m = out["sigma2"].mean().item()
    # signal recovery within tolerance (shrinkage priors bias the
    # posterior mean toward 0 by design, and the deflated sigma^2 in
    # p >> n deflates posterior sds -- so 2-sd interval coverage is
    # NOT the right correctness criterion here; see note S4.5)
    recovered = all(abs(bm[j].item() - v) < 0.35 for j, v in signals.items())
    noise_max = bm[5:].abs().max().item()
    # NOTE: with the default a1=b1=1 the posterior mean of sigma^2
    # partially recovers (~0.33 expected at n=100, p=500) -- the
    # residual underestimation is a documented model property in
    # p >> n (see Section 4.5 of the note / sigma2_sample WARNING).
    print(f"  sigma2 mean {s2m:.3f} (true 1.0; ~0.33 expected in p>>n, "
          f"see note S4.5), signals: {[round(bm[j].item(), 2) for j in signals]}, "
          f"noise max |mean| {noise_max:.3f}, recovered: {recovered}, "
          f"runtime {out['runtime_sec']:.1f}s")
    ok = recovered and noise_max < 0.2 and s2m > 0.2
    print(f"  {'PASS' if ok else 'FAIL'}")


def test_method_agreement():
    print("=== Test 2: two methods agree ===")
    n, p = 80, 120
    signals = {0: 1.5, 1: -1.0, 2: 0.8}
    X, Y, bt = simulate(n, p, signals, sigma=0.8, seed=2)
    means, s2s = {}, {}
    for m in ["fast", "direct"]:
        out = GS_LH(X, Y, n_iter=2000, burnin=500, method=m,
                    standardize=False, seed=7)
        means[m] = out["beta"].mean(0)
        s2s[m] = out["sigma2"].mean().item()
        print(f"  {m:7s}: signal means "
              f"{[round(means[m][j].item(), 3) for j in signals]}, "
              f"sigma2 {s2s[m]:.3f}")
    dmax = max((means[a] - means[b]).abs().max().item()
               for a in means for b in means)
    s2_spread = max(s2s.values()) / min(s2s.values()) - 1
    print(f"  max |mean diff| across methods = {dmax:.4f} (< 0.15 required), "
          f"sigma2 spread = {s2_spread:.2%} (< 10% required)")
    print("  PASS" if dmax < 0.15 and s2_spread < 0.10 else "  FAIL")


def test_standardize():
    print("=== Test 3: standardize=True with intercept ===")
    g = torch.Generator().manual_seed(3)
    n, p = 200, 100
    beta_true = torch.zeros(p, dtype=dtype)
    beta_true[[0, 1, 2]] = torch.tensor([1.5, -1.0, 0.8], dtype=dtype)
    X = torch.randn(n, p, generator=g, dtype=dtype)
    X = X * (torch.arange(p, dtype=dtype) % 5 + 1) + 0.7   # arbitrary scales/means
    Y = 2.5 + X @ beta_true + 0.5 * torch.randn(n, generator=g, dtype=dtype)
    out = GS_LH(X, Y, n_iter=3000, burnin=1000, method="fast",
                standardize=True, seed=5)
    pre = out["preprocess"]
    bm = out["beta"].mean(0)
    beta_orig = bm / pre["x_scale"]
    intercept = pre["y_mean"] - pre["x_mean"] @ beta_orig
    print(f"  recovered beta_orig[:3] = "
          f"{[round(beta_orig[j].item(), 3) for j in range(3)]} "
          f"(true [1.5, -1.0, 0.8])")
    print(f"  intercept = {intercept:.3f} (true 2.5)")
    err = (beta_orig[:3] - beta_true[:3]).abs().max().item()
    ok = err < 0.2 and abs(intercept - 2.5) < 0.25
    print("  PASS" if ok else "  FAIL")


def test_gpu_and_speed():
    print("=== Test 4: GPU smoke + throughput ===")
    if torch.cuda.is_available():
        n, p = 100, 2000
        X, Y, _ = simulate(n, p, {0: 1.5, 1: -1.0}, sigma=1.0, seed=4)
        out = GS_LH(X.cuda(), Y.cuda(), n_iter=200, burnin=100,
                    method="fast", standardize=False, seed=9)
        print(f"  GPU float64 fast: beta finite "
              f"{torch.isfinite(out['beta']).all().item()}, "
              f"sigma2 mean {out['sigma2'].mean():.3f}")
    else:
        print("  (no CUDA)")

    n, p = 100, 4000
    X, Y, _ = simulate(n, p, {0: 1.5, 1: -1.0, 2: 0.8}, sigma=1.0, seed=6)
    out = GS_LH(X, Y, n_iter=500, burnin=100, method="fast",
                standardize=False, seed=11)
    rate = 600 / out["runtime_sec"]
    print(f"  CPU (n=100, p=4000): {rate:.0f} it/s "
          f"-> 10k iterations ~ {10000 / rate:.0f}s")


if __name__ == "__main__":
    test_recovery()
    test_method_agreement()
    test_standardize()
    test_gpu_and_speed()
