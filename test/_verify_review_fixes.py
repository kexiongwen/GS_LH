"""Verification for the review fixes (w0 validation, JAX audit, cosmetics)."""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from GS_LH import GS_LH as GS_LH_torch
from GS_LH_JAX import GS_LH as GS_LH_jax

n, p = 50, 80
g = np.random.default_rng(0)
X = g.standard_normal((n, p))
Y = g.standard_normal(n)
Xt = torch.tensor(X, dtype=torch.float64)
Yt = torch.tensor(Y, dtype=torch.float64)

results = []


def expect_raise(tag, fn, exc):
    try:
        fn()
    except exc as e:
        print(f"  PASS: {tag} -> {type(e).__name__}: {str(e)[:60]}...")
        results.append(True)
    except Exception as e:
        print(f"  FAIL: {tag} -> wrong exception {type(e).__name__}: {e}")
        results.append(False)
    else:
        print(f"  FAIL: {tag} -> no exception raised")
        results.append(False)


print("=== 1. w0 validation rejects NaN / inf / negative ===")
for bad, name in [(np.full(p, np.nan), "NaN"),
                  (np.full(p, np.inf), "inf"),
                  (np.full(p, -1.0), "negative")]:
    expect_raise(f"torch w0 {name:9s}",
                 lambda b=bad: GS_LH_torch(
                     Xt, Yt, 5, burnin=2,
                     w0=torch.tensor(b, dtype=torch.float64)), ValueError)
    expect_raise(f"jax   w0 {name:9s}",
                 lambda b=bad: GS_LH_jax(X, Y, 5, burnin=2, w0=b), ValueError)

print("=== 2. non-finite start (w0 = 1e38) fails loudly, never silently ===")
# torch CPU exact-Gamma path: the NaN rate trips the distribution's
# parameter validation (ValueError) before the audit; torch CUDA chi-square
# path: NaN propagates silently to sigma2 and the audit raises
# FloatingPointError.  Both are loud failures -- the chain is never
# silently contaminated.
expect_raise("torch audit (CPU float32, w0=1e38)",
             lambda: GS_LH_torch(
                 torch.tensor(X, dtype=torch.float32),
                 torch.tensor(Y, dtype=torch.float32),
                 10, burnin=0, audit_every=5,
                 w0=torch.full((p,), 1e38, dtype=torch.float32)),
             (ValueError, FloatingPointError))
if torch.cuda.is_available():
    expect_raise("torch audit (CUDA float32, w0=1e38)",
                 lambda: GS_LH_torch(
                     torch.tensor(X, dtype=torch.float32).cuda(),
                     torch.tensor(Y, dtype=torch.float32).cuda(),
                     10, burnin=0, audit_every=5,
                     w0=torch.full((p,), 1e38, dtype=torch.float32).cuda()),
                 FloatingPointError)
# jax float32: the new per-chunk audit must catch it
expect_raise("jax   audit (w0=1e38)",
             lambda: GS_LH_jax(X, Y, 10, burnin=0,
                               w0=np.full(p, 1e38, dtype=np.float32)),
             FloatingPointError)

print("=== 3. normal runs unaffected ===")
out_t = GS_LH_torch(Xt, Yt, 50, burnin=10, seed=1)
out_j = GS_LH_jax(X, Y, 50, burnin=10, seed=1)
ok_t = torch.isfinite(out_t["beta"]).all().item() and \
    torch.isfinite(out_t["sigma2"]).all().item()
ok_j = bool(np.isfinite(out_j["beta"]).all()) and \
    bool(np.isfinite(out_j["sigma2"]).all())
print(f"  torch run finite: {ok_t}, jax run finite: {ok_j}")
results.append(ok_t and ok_j)

print("=== 4. torch graph-mode verbose line includes sigma2 (GPU) ===")
if torch.cuda.is_available():
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        GS_LH_torch(Xt.cuda(), Yt.cuda(), 20, burnin=5, seed=2,
                    graph=True, verbose=True)
    line = buf.getvalue().strip().splitlines()[-1]
    ok = "sigma2 =" in line
    print(f"  {'PASS' if ok else 'FAIL'}: graph verbose line: {line}")
    results.append(ok)
else:
    print("  (no CUDA)")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
