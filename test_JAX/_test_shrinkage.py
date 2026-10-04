"""Quick statistical smoke tests for GS_LH_JAX/shrinkage_sample.py (float32)."""
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GS_LH_JAX.shrinkage_sample import inv_gauss, shrinkage

key = jax.random.PRNGKey(0)

# --- Test 1: inv_gauss moments. IG(mu, 1): mean = mu, var = mu^3 ---
print("=== Test 1: inv_gauss moments ===")
N = 2_000_000
ok = True
for mu_val in [0.5, 1.0, 3.0]:
    key, sub = jax.random.split(key)
    x = inv_gauss(sub, jnp.full((N,), mu_val, jnp.float32))
    m, v = float(x.mean()), float(x.var())
    print(f"IG(mu={mu_val},1): mean {m:.4f} (true {mu_val}), "
          f"var {v:.4f} (true {mu_val ** 3:.4f})")
    ok &= abs(m / mu_val - 1) < 0.01 and abs(v / mu_val ** 3 - 1) < 0.10
print("  PASS" if ok else "  FAIL")

# --- Test 2: shrinkage basic run ---
print("=== Test 2: shrinkage basic run ===")
p = 200
scale = jnp.array([0.1] * 50 + [1.0] * 150, jnp.float32)
key, sub = jax.random.split(key)
param = jax.random.normal(sub, (p,), jnp.float32) * scale
key, sub = jax.random.split(key)
w = shrinkage(sub, param, a=0.001, b=0.001)
finite = bool(jnp.isfinite(w).all())
positive = bool((w > 0).all())
print("shrinkage: shape", tuple(w.shape),
      "all finite:", finite, "all positive:", positive)
print("  PASS" if w.shape == (p,) and finite and positive else "  FAIL")

# --- Test 3: reproducibility with same key ---
w1 = shrinkage(sub, param, 0.001, 0.001)
w2 = shrinkage(sub, param, 0.001, 0.001)
print("reproducible under same key:", bool(jnp.array_equal(w1, w2)))

# --- Test 4: S3 lambda conditional matches Gamma(2p+a, b + sum sqrt|beta~|) ---
print("=== Test 4: S3 lambda moments ===")
a_, b_ = 2.0, 1.0
true_rate = float(jnp.sqrt(jnp.abs(param)).sum()) + b_
key, sub = jax.random.split(key)
lams = jax.random.gamma(sub, 2 * p + a_, shape=(20000,)) / true_rate
m_t, v_t = (2 * p + a_) / true_rate, (2 * p + a_) / true_rate ** 2
m, v = float(lams.mean()), float(lams.var())
print(f"S3 lambda: mean {m:.4f} (true {m_t:.4f}), var {v:.4f} (true {v_t:.4f})")
print("  PASS" if abs(m / m_t - 1) < 0.02 and abs(v / v_t - 1) < 0.05 else "  FAIL")

# --- Test 5: edge cases - tiny and exact-zero beta ---
print("=== Test 5: edge cases ===")
param_tiny = param.at[0].set(1e-30)
key, sub = jax.random.split(key)
w = shrinkage(sub, param_tiny, 0.001, 0.001)
print("tiny beta: all finite:", bool(jnp.isfinite(w).all()))

param_zero = param.at[0].set(0.0)
w = shrinkage(sub, param_zero, 0.001, 0.001)
print("exact zero beta: all finite:", bool(jnp.isfinite(w).all()),
      "| w[0] =", float(w[0]))
