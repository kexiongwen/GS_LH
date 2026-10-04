"""Benchmarks: beta_sample_direct (Prop 2.1 + Cholesky = Rue) vs beta_sample."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH.beta_sample import beta_sample, beta_sample_direct

CUDA = torch.cuda.is_available()


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


def make_dense(n, p, dtype, dev, s=20, seed=3):
    torch.manual_seed(seed)
    X = torch.randn(n, p, dtype=dtype, device=dev)
    Y = torch.randn(n, dtype=dtype, device=dev)
    w = torch.full((p,), 0.02, dtype=dtype, device=dev)
    w[:s] = 3.0
    sigma = torch.tensor(1.0, dtype=dtype, device=dev)
    return X, Y, w, sigma


def check_direct(dev):
    """Mean sanity check for beta_sample_direct."""
    torch.manual_seed(0)
    n, p, N = 30, 200, 20000
    X, Y, w, sigma = make_dense(n, p, torch.float64, dev)
    A = (X.T @ X + torch.diag(w.pow(-2))).cpu()
    mu = torch.linalg.solve(A, (X.T @ Y).cpu())
    s1 = torch.zeros(p, dtype=torch.float64, device=dev)
    for _ in range(N):
        s1 += beta_sample_direct(X, Y, w, sigma)
    z = ((s1 / N).cpu() - mu).abs().max().item()
    print(f"direct on {dev}: max |mean - mu| = {z:.2e} (MC scale ~3e-03)")


def run_suite(dev, dtype):
    cuda = dev.type == "cuda"
    tag = f"{'GPU' if cuda else 'CPU'} {str(dtype)[6:]}"
    sizes = [(100, 4000, 10), (2000, 2000, 5), (4000, 4000, 2), (8000, 500, 5)]
    for n, p, reps in sizes:
        X, Y, w, sigma = make_dense(n, p, dtype, dev)
        # constant across Gibbs iterations; precomputed once by GS_LH
        XtX, XtY = X.T @ X, X.T @ Y
        t1 = bench(beta_sample, (X, Y, w, sigma), reps, cuda)
        td = bench(lambda X, Y, w, s: beta_sample_direct(
            X, Y, w, s, XtX=XtX, XtY=XtY), (X, Y, w, sigma), reps, cuda)
        best = min(t1, td)
        print(f"  {tag:14s} n={n:5d}, p={p:5d}: "
              f"Alg1 {t1 * 1000:8.2f}{'*' if t1 == best else ' '} "
              f"direct {td * 1000:8.2f}{'*' if td == best else ' '} ms")


if __name__ == "__main__":
    check_direct(torch.device("cpu"))
    if CUDA:
        check_direct(torch.device("cuda"))
    print("\nTiming per draw (sparse w, s=20); * = fastest in row:")
    run_suite(torch.device("cpu"), torch.float64)
    if CUDA:
        run_suite(torch.device("cuda"), torch.float64)
        run_suite(torch.device("cuda"), torch.float32)
