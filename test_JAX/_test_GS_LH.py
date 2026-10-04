"""End-to-end verification for the GS_LH_JAX driver (float32)."""
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH_JAX import GS_LH


def simulate(n, p, signals, sigma=1.0, seed=0):
    """X columns standardized to ||X_j||^2 = n; sparse true beta.
    Returns numpy float32 arrays (usable by both the JAX and the
    PyTorch drivers; see _test_vs_torch.py)."""
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, p)).astype(np.float32)
    X *= (n ** 0.5 / np.linalg.norm(X, axis=0))
    beta_true = np.zeros(p, dtype=np.float32)
    for j, v in signals.items():
        beta_true[j] = v
    Y = (X @ beta_true + sigma * g.standard_normal(n)).astype(np.float32)
    return X, Y, beta_true


def test_recovery():
    print("=== Test 1: recovery (method='fast') ===")
    n, p = 100, 500
    signals = {0: 2.0, 1: -1.5, 2: 1.2, 3: 0.9, 4: -0.7}
    X, Y, bt = simulate(n, p, signals, sigma=1.0, seed=1)
    out = GS_LH(X, Y, n_iter=4000, burnin=1000, method="fast",
                standardize=False, seed=42)
    bm, bs = out["beta"].mean(0), out["beta"].std(0)
    s2m = out["sigma2"].mean()
    # signal recovery within tolerance (shrinkage priors bias the
    # posterior mean toward 0 by design, and the deflated sigma^2 in
    # p >> n deflates posterior sds -- so 2-sd interval coverage is
    # NOT the right correctness criterion here; see note S4.5)
    recovered = all(abs(bm[j] - v) < 0.35 for j, v in signals.items())
    noise_max = np.abs(bm[5:]).max()
    # NOTE: with the default a1=b1=1 the posterior mean of sigma^2
    # partially recovers (~0.33 expected at n=100, p=500) -- the
    # residual underestimation is a documented model property in
    # p >> n (see Section 4.5 of the note / sigma2_sample WARNING).
    print(f"  sigma2 mean {s2m:.3f} (true 1.0; ~0.33 expected in p>>n, "
          f"see note S4.5), signals: {[round(float(bm[j]), 2) for j in signals]}, "
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
        s2s[m] = out["sigma2"].mean()
        print(f"  {m:7s}: signal means "
              f"{[round(float(means[m][j]), 3) for j in signals]}, "
              f"sigma2 {s2s[m]:.3f}")
    dmax = max(np.abs(means[a] - means[b]).max()
               for a in means for b in means)
    s2_spread = max(s2s.values()) / min(s2s.values()) - 1
    print(f"  max |mean diff| across methods = {dmax:.4f} (< 0.15 required), "
          f"sigma2 spread = {s2_spread:.2%} (< 10% required)")
    print("  PASS" if dmax < 0.15 and s2_spread < 0.10 else "  FAIL")


def test_standardize():
    print("=== Test 3: standardize=True with intercept ===")
    g = np.random.default_rng(3)
    n, p = 200, 100
    beta_true = np.zeros(p, dtype=np.float32)
    beta_true[[0, 1, 2]] = [1.5, -1.0, 0.8]
    X = g.standard_normal((n, p)).astype(np.float32)
    X = X * (np.arange(p, dtype=np.float32) % 5 + 1) + 0.7   # arbitrary scales/means
    Y = (2.5 + X @ beta_true
         + 0.5 * g.standard_normal(n)).astype(np.float32)
    out = GS_LH(X, Y, n_iter=3000, burnin=1000, method="fast",
                standardize=True, seed=5)
    pre = out["preprocess"]
    bm = out["beta"].mean(0)
    beta_orig = bm / pre["x_scale"]
    intercept = pre["y_mean"] - pre["x_mean"] @ beta_orig
    print(f"  recovered beta_orig[:3] = "
          f"{[round(float(beta_orig[j]), 3) for j in range(3)]} "
          f"(true [1.5, -1.0, 0.8])")
    print(f"  intercept = {intercept:.3f} (true 2.5)")
    err = np.abs(beta_orig[:3] - beta_true[:3]).max()
    ok = err < 0.2 and abs(intercept - 2.5) < 0.25
    print("  PASS" if ok else "  FAIL")


def test_speed():
    print("=== Test 4: backend + throughput ===")
    print(f"  JAX devices: {jax.devices()}")
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
    test_speed()
