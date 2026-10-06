"""Verification for sigma2_sample.py: distribution moments, L-sharing, benchmarks."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH.beta_sample import beta_sample, beta_sample_direct
from GS_LH.sigma2_sample import sigma2_sample, sigma2_sample_direct

dtype = torch.float64


def exact_q(X, Y, w):
    """q = Y' Sigma^{-1} Y by brute force."""
    n = X.shape[0]
    Xw = X * w
    M = Xw @ Xw.T
    M.diagonal().add_(1.0)
    return torch.cholesky_solve(Y.unsqueeze(-1), torch.linalg.cholesky(M)).squeeze(-1) @ Y


def distribution_check(fn, n, p, a1=3.0, b1=2.0, N=200_000, seed=0, **kw):
    """Sample sigma^2 with fixed (X, Y, w); compare to InvGamma theory."""
    torch.manual_seed(seed)
    X = torch.randn(n, p, dtype=dtype)
    Y = torch.randn(n, dtype=dtype)
    w = torch.rand(p, dtype=dtype) * 2 + 0.1

    q = exact_q(X, Y, w)
    alpha, gamma = a1 + n / 2, b1 + q / 2
    mean_true = gamma / (alpha - 1)
    var_true = gamma**2 / ((alpha - 1) ** 2 * (alpha - 2))

    draws = torch.stack([fn(X, Y, w, a1, b1, **kw) for _ in range(N)])
    m, v = draws.mean().item(), draws.var().item()
    print(f"{fn.__name__} (n={n}, p={p}): mean {m:.5f} vs {mean_true.item():.5f} "
          f"({(m / mean_true.item() - 1):+.2%}), var {v:.5f} vs {var_true.item():.5f} "
          f"({(v / var_true.item() - 1):+.2%})")


def sharing_check():
    n, p = 30, 200
    torch.manual_seed(1)
    X = torch.randn(n, p, dtype=dtype)
    Y = torch.randn(n, dtype=dtype)
    w = torch.rand(p, dtype=dtype) + 0.1
    a1, b1 = 3.0, 2.0

    # n x n factor shared S1 -> S2 (Algorithm 1)
    s2, L = sigma2_sample(X, Y, w, a1, b1, return_L=True)
    b = beta_sample(X, Y, w, s2.sqrt(), L=L)
    ok = bool(torch.isfinite(b).all())
    print("n x n L shared with beta_sample: beta finite:", ok)
    assert ok

    # p x p factor shared S1 -> S2 (direct)
    s2, Lp = sigma2_sample_direct(X, Y, w, a1, b1, return_L=True)
    b = beta_sample_direct(X, Y, w, s2.sqrt(), L=Lp)
    ok = bool(torch.isfinite(b).all())
    print("p x p L shared with beta_sample_direct: beta finite:", ok)
    assert ok

    # precomputed XtX/XtY/YtY path (as driven by GS_LH): identical draws,
    # caller's XtX not mutated by the congruence scaling
    XtX, XtY, YtY = X.T @ X, X.T @ Y, torch.dot(Y, Y)
    torch.manual_seed(11)
    s2a, La = sigma2_sample_direct(X, Y, w, a1, b1, return_L=True)
    ba = beta_sample_direct(X, Y, w, s2a.sqrt(), L=La)
    torch.manual_seed(11)
    s2b, Lb = sigma2_sample_direct(X, Y, w, a1, b1, return_L=True,
                                   XtX=XtX, XtY=XtY, YtY=YtY)
    bb = beta_sample_direct(X, Y, w, s2b.sqrt(), L=Lb, XtX=XtX, XtY=XtY)
    ok = torch.equal(s2a, s2b) and torch.equal(ba, bb)
    print("precomputed-XtX path identical:", ok)
    assert ok
    ok = torch.equal(XtX, X.T @ X)
    print("caller XtX not mutated:", ok)
    assert ok

    # determinism
    torch.manual_seed(7)
    r1 = sigma2_sample(X, Y, w, a1, b1)
    torch.manual_seed(7)
    r2 = sigma2_sample(X, Y, w, a1, b1)
    ok = torch.equal(r1, r2)
    print("deterministic under same seed:", ok)
    assert ok


def bench(fn, args, reps, cuda):
    fn(*args)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def benchmark():
    a1, b1 = 3.0, 2.0
    print("\nS1 timing per draw (sparse w, s=20); * = fastest in row:")
    for dev in [torch.device("cpu"), torch.device("cuda")]:
        if dev.type == "cuda" and not torch.cuda.is_available():
            continue
        cuda = dev.type == "cuda"
        for dt in ([torch.float64] if not cuda else [torch.float64, torch.float32]):
            for n, p, reps in [(100, 4000, 20), (8000, 500, 10)]:
                torch.manual_seed(3)
                X = torch.randn(n, p, dtype=dt, device=dev)
                Y = torch.randn(n, dtype=dt, device=dev)
                w = torch.full((p,), 0.02, dtype=dt, device=dev)
                w[:20] = 3.0
                args = (X, Y, w, a1, b1)
                t1 = bench(sigma2_sample, args, reps, cuda)
                t2 = bench(sigma2_sample_direct, args, reps, cuda)
                best = min(t1, t2)
                tag = f"{'GPU' if cuda else 'CPU'} {str(dt)[6:]}"
                print(f"  {tag:14s} n={n:5d}, p={p:5d}: "
                      f"n-chol {t1 * 1e3:8.2f}{'*' if t1 == best else ' '} "
                      f"p-chol {t2 * 1e3:8.2f}{'*' if t2 == best else ' '} ms")


if __name__ == "__main__":
    distribution_check(sigma2_sample, n=30, p=200)
    distribution_check(sigma2_sample_direct, n=200, p=30)
    sharing_check()
    benchmark()
