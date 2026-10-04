"""torch.linalg.solve (LU) vs cholesky + cholesky_solve on an SPD system."""
import time

import torch


def bench(fn, reps, cuda):
    fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def run(dev, dtype, p=4000, reps=10):
    cuda = dev.type == "cuda"
    torch.manual_seed(0)
    # SPD matrix in the same style as our sampler: A = X'X + diag(w)^{-2}
    n = p // 2
    X = torch.randn(n, p, dtype=dtype, device=dev)
    w = torch.rand(p, dtype=dtype, device=dev) + 0.1
    A = X.T @ X
    A.diagonal().add_(w.pow(-2))
    b = torch.randn(p, 1, dtype=dtype, device=dev)

    def by_solve():
        return torch.linalg.solve(A, b)

    def by_cholesky():
        L = torch.linalg.cholesky(A)
        return torch.cholesky_solve(b, L)

    t_solve = bench(by_solve, reps, cuda)
    t_chol = bench(by_cholesky, reps, cuda)
    diff = (by_solve() - by_cholesky()).abs().max().item()
    print(f"{'GPU' if cuda else 'CPU'} {str(dtype)[6:]:8s} p={p}: "
          f"solve(LU) {t_solve * 1000:8.2f} ms | chol+solve {t_chol * 1000:8.2f} ms "
          f"| speedup {t_solve / t_chol:4.2f}x | max|diff| {diff:.2e}")


for dev in [torch.device("cpu"), torch.device("cuda")]:
    for dtype in [torch.float64, torch.float32]:
        run(dev, dtype)
