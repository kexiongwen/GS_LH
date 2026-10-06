"""Statistical verification for beta_sample.py (not part of the package)."""
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH.beta_sample import beta_sample, beta_sample_direct

dtype = torch.float64


def theoretical(X, Y, w, sigma):
    """Exact mu = V X'Y and Sigma = sigma^2 V by direct inversion."""
    p = X.shape[1]
    A = X.T @ X + torch.diag(w.pow(-2))
    V = torch.linalg.inv(A)
    return V @ X.T @ Y, sigma**2 * V


def moment_check(fn, n, p, N=200_000, seed=0, **kw):
    torch.manual_seed(seed)
    X = torch.randn(n, p, dtype=dtype)
    Y = torch.randn(n, dtype=dtype)
    w = torch.rand(p, dtype=dtype) * 2 + 0.1
    sigma = torch.tensor(1.3, dtype=dtype)

    # exercise the exact path the GS_LH driver uses for the direct
    # sampler: precomputed XtX / XtY (constant across Gibbs iterations)
    if fn is beta_sample_direct and "XtX" not in kw:
        kw = {"XtX": X.T @ X, "XtY": X.T @ Y, **kw}

    mu, Sigma = theoretical(X, Y, w, sigma)

    # running accumulators (avoid storing N x p)
    s1 = torch.zeros(p, dtype=dtype)
    s2 = torch.zeros(p, p, dtype=dtype)
    for _ in range(N):
        b = fn(X, Y, w, sigma, **kw)
        s1 += b
        s2 += torch.outer(b, b)
    mean_hat = s1 / N
    cov_hat = s2 / N - torch.outer(mean_hat, mean_hat)

    sd = Sigma.diagonal().sqrt()
    mean_z = ((mean_hat - mu) / sd).abs().max().item()
    cov_err = ((cov_hat - Sigma).norm() / Sigma.norm()).item()
    # MC error scales: E|cov_hat - Sigma|_F^2 ~ (tr(Sigma)^2 + |Sigma|_F^2) / N
    # (diagonal-dominated Sigma makes it much larger than the naive 1/sqrt(N))
    mean_z_exp = (2 * math.log(2 * p) / N) ** 0.5
    cov_err_exp = (((Sigma.trace() ** 2 + Sigma.norm() ** 2) / N).sqrt()
                   / Sigma.norm()).item()
    print(f"{fn.__name__} n={n}, p={p}: max |z| of mean = {mean_z:.4f} "
          f"(~{mean_z_exp:.4f} expected), rel. Frobenius err of cov = {cov_err:.4%} "
          f"(~{cov_err_exp:.4%} expected)")
    ok = mean_z < 1.5 * mean_z_exp and cov_err < 1.5 * cov_err_exp
    print("  PASS" if ok else "  FAIL")


def chol_factor(X, w):
    Xw = X * w
    M = Xw @ Xw.T
    M.diagonal().add_(1.0)
    return torch.linalg.cholesky(M)


def reuse_check():
    n, p = 30, 200
    torch.manual_seed(1)
    X = torch.randn(n, p, dtype=dtype)
    Y = torch.randn(n, dtype=dtype)
    w = torch.rand(p, dtype=dtype) + 0.1
    sigma = torch.tensor(0.9, dtype=dtype)
    L = chol_factor(X, w)

    torch.manual_seed(123)
    b1 = beta_sample(X, Y, w, sigma)
    torch.manual_seed(123)
    b2 = beta_sample(X, Y, w, sigma, L=L)
    ok = torch.equal(b1, b2)
    print("precomputed-L draw identical:", ok)
    assert ok


def rue_sample(X, Y, w, sigma):
    """Naive O(p^3) sampler: Cholesky of the p x p precision (Rue 2001)."""
    A = X.T @ X + torch.diag(w.pow(-2))
    Lp = torch.linalg.cholesky(A)
    mu = torch.cholesky_solve((X.T @ Y).unsqueeze(-1), Lp).squeeze(-1)
    z = torch.randn(X.shape[1], dtype=dtype)
    # cov = sigma^2 A^{-1} = sigma^2 L^{-T} L^{-1}: sample via solve(L.T, z)
    return mu + sigma * torch.linalg.solve_triangular(Lp.T, z.unsqueeze(-1), upper=True).squeeze(-1)


def benchmark(n=100, p=4000, reps=20):
    torch.manual_seed(2)
    X = torch.randn(n, p, dtype=dtype)
    Y = torch.randn(n, dtype=dtype)
    w = torch.rand(p, dtype=dtype) + 0.1
    sigma = torch.tensor(1.0, dtype=dtype)

    for name, fn in [("Algorithm 1 (fast)", beta_sample), ("Rue Cholesky", rue_sample)]:
        fn(X, Y, w, sigma)  # warm-up
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(X, Y, w, sigma)
        dt = (time.perf_counter() - t0) / reps
        print(f"{name}: {dt * 1000:.1f} ms per draw (n={n}, p={p})")


if __name__ == "__main__":
    moment_check(beta_sample, n=30, p=200)         # p >> n regime
    moment_check(beta_sample, n=200, p=30)         # n > p regime (exact for all n, p)
    moment_check(beta_sample_direct, n=200, p=30)  # its regime; driver path (XtX/XtY)
    moment_check(beta_sample_direct, n=30, p=200)  # p >> n: rank-deficient X'X + diag(w^-2)
    reuse_check()
    benchmark()
