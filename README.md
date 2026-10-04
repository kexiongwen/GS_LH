# GS_LH — Gibbs Sampler for Bayesian L1/2 Regression

Fast Gibbs samplers for the Bayesian $L_{1/2}$ regression of
**Ke & Fan (2024, JCGS)**, in **PyTorch** (`GS_LH/`) and **JAX**
(`GS_LH_JAX/`), together with a mathematical note
(`On the Connection between Dirichlet–Laplace Prior and L1_2 Prior.md`)
showing that the $L_{1/2}$ prior is exactly the Dirichlet–Laplace prior
of Bhattacharya et al. (2015) at concentration $a = 3/2$.

## Model

$$Y = X\beta + \varepsilon, \qquad \varepsilon \sim \mathcal N_n(0, \sigma^2 I_n),$$

with the $L_{1/2}$ prior placed on the error-standardized coefficients
$\tilde\beta_j = \beta_j/\sigma$:

$$\pi(\tilde\beta_j \mid \lambda) = \frac{\lambda^2}{4}\exp\!\big(-\lambda\sqrt{|\tilde\beta_j|}\,\big),
\qquad \lambda \sim \mathrm{Gamma}(a, b), \qquad \sigma^2 \sim \mathrm{InvGamma}(a_1, b_1).$$

Up to a constant, the negative log-prior is
$\lambda \sum_j \sqrt{|\beta_j|/\sigma}$ — the $L_{1/2}$ penalty applied to
error-standardized coefficients — so posterior modes coincide with
$L_{1/2}$-penalized estimates. High-dimensional designs ($p > n$) are
supported throughout.

## Features

- **Exact Gibbs sampler** (S1–S5) with a **collapsed $\sigma^2$ update**
  ($\beta$ integrated out) forming a joint $(\sigma^2, \beta)$ block update.
- **One Cholesky factorization per iteration**, shared between the
  collapsed $\sigma^2$ update and the $\beta$ update.
- **Two exact algorithm pairings**, auto-selected by the $n$ vs $p$ regime:
  - `fast` ($p \ge n$): Bhattacharya, Chakraborty & Mallick (2016)
    Algorithm 1 on the $n \times n$ system — $O(n^2 p + n^3)$ per iteration;
  - `direct` ($n > p$): Nishimura & Suchard (2022) Prop. 2.1 = Rue (2001)
    on the $p \times p$ system — $O(p^3)$ per iteration after a one-time
    precompute of $X^\top X$.
- **No general giG sampler needed**: every shrinkage draw (S3–S5) goes
  through a single numerically stabilized shape-1 inverse-Gaussian
  generator (reciprocal duality + scaling law; rationalized
  Michael–Schucany–Haas with a zero guard, safe in float32 and float64).
- **Minimal chain state**: only $w = \tau/\lambda^2$ is carried between
  iterations; $\lambda, v, \tau$ are regenerated each pass, and
  $\beta, \sigma^2$ are never initialized.
- **JAX version**: fixed single precision (float32); the whole chain runs
  as chunked `jax.lax.scan` — each chunk is one compiled XLA While loop
  with no Python dispatch or host synchronization inside.
- **`p >> n` $\sigma^2$ pathology handled by default** (see the caveat
  below): the $\sigma^2$ prior defaults to $\mathrm{InvGamma}(1, 1)$.

## The Dirichlet–Laplace connection

The accompanying note proves that at Dirichlet concentration $a = 3/2$ the
$L_{1/2}$ prior and the Dirichlet–Laplace prior coincide layer by layer
under the change of variables $\nu_j = v_j/\lambda^2$,
$\psi_j = \tau_j^2/v_j^2$, and their Gibbs samplers are *one sampler* —
except for a single step: the DL prior fixes the global rate at $1/2$,
while the $L_{1/2}$ prior learns $r = \lambda^2/4$ through the hyperprior
$\lambda \sim \mathrm{Gamma}(c, d)$. At $a = 3/2$ the DL marginal is the
unique elementary member of its family (a pure exponential in
$\sqrt{|\tilde\beta|}$, free of Bessel functions). See
[`On the Connection between Dirichlet–Laplace Prior and L1_2 Prior.md`](On%20the%20Connection%20between%20Dirichlet–Laplace%20Prior%20and%20L1_2%20Prior.md).

## Installation

No build step — pure Python. Dependencies:

```bash
pip install numpy torch        # for GS_LH (PyTorch version)
pip install numpy jax          # for GS_LH_JAX (JAX version)
```

Developed and tested with Python 3 (conda), `torch 2.14`, `jax 0.11`.
The JAX version needs no GPU (tested CPU-only); the PyTorch version runs
on CPU or CUDA.

## Quick start

### PyTorch

```python
import torch
from GS_LH import GS_LH

n, p = 100, 500
X = torch.randn(n, p)
y = torch.randn(n)

out = GS_LH(X, y, n_iter=4000, burnin=1000, seed=42)
beta_mean  = out["beta"].mean(0)     # posterior mean of beta
sigma2_hat = out["sigma2"].mean()    # posterior mean of sigma^2
```

### JAX (fixed float32)

```python
import numpy as np
from GS_LH_JAX import GS_LH

X = np.random.randn(100, 500).astype(np.float32)
y = np.random.randn(100).astype(np.float32)

out = GS_LH(X, y, n_iter=4000, burnin=1000, seed=42)
out["beta"]      # (4000, 500) numpy array of posterior draws
```

Both drivers share the same interface and return a dict:

| key | content |
| --- | --- |
| `"beta"` | `(n_iter, p)` posterior $\beta$ draws (CPU tensor / numpy array) |
| `"sigma2"` | `(n_iter,)` posterior $\sigma^2$ draws |
| `"method"` | the algorithm pairing actually used (`"fast"` / `"direct"`) |
| `"preprocess"` | standardization stats `{"y_mean", "x_mean", "x_scale"}` or `None` |
| `"w_final"` | last chain state $w$ (pass as `w0` for warm restarts) |
| `"runtime_sec"` | wall-clock seconds of the sampling loop |

### Main arguments

| argument | default | meaning |
| --- | --- | --- |
| `n_iter` | — | posterior draws to keep (after burn-in) |
| `burnin` | `0` | initial iterations to discard |
| `a1, b1` | `1.0, 1.0` | prior $\sigma^2 \sim \mathrm{InvGamma}(a_1, b_1)$ — see caveat below |
| `a, b` | `1e-3, 1e-3` | hyperprior $\lambda \sim \mathrm{Gamma}(a, b)$ (shape, rate) |
| `method` | `"auto"` | `"fast"` ($p \ge n$) / `"direct"` ($n > p$) / `"auto"` |
| `w0` | `None` | initial state $w = \tau/\lambda^2$ (default: ones) |
| `standardize` | `True` | center $Y$, center and rescale $X$ columns to $\|X_j\|^2 = n$ |
| `seed` | `None` | RNG seed (fully reproducible chains) |
| `verbose` | `False` | progress every ~10% of iterations |

### PyTorch vs JAX at a glance

| | `GS_LH` (PyTorch) | `GS_LH_JAX` (JAX) |
| --- | --- | --- |
| precision | follows input dtype (float64 recommended/tested) | fixed float32 |
| device | CPU, CUDA | CPU (Windows), GPU where JAX supports it |
| sampling loop | eager Python loop | chunked `jax.lax.scan`, whole chain XLA-compiled |
| RNG | `torch.manual_seed` | explicit `jax.random.PRNGKey` |
| draws returned as | CPU `torch.Tensor` | `numpy.ndarray` |

### Importing submodules

Both packages re-export the driver via `from .GS_LH import GS_LH`, so on
the package object the name `GS_LH` — the driver **function** — shadows
the submodule of the same name. Importing the driver and the step
functions is unaffected:

```python
from GS_LH import GS_LH                      # driver function (usual way)
from GS_LH.beta_sample import beta_sample    # step-function modules
from GS_LH.GS_LH import beta_sample          # also fine
```

The only pitfall is binding the submodule object itself:
`import GS_LH.GS_LH as m` assigns the **function** (the `as` form
resolves through the shadowed package attribute). To reach the actual
module, use `sys.modules`:

```python
import sys
m = sys.modules["GS_LH.GS_LH"]
```

(The same applies to `GS_LH_JAX`.)

## Caveat: $\sigma^2$ estimation in the $p \gg n$ regime

With the customary near-Jeffreys choice $a_1 \approx 0$, the **posterior
itself** (not the sampler) concentrates on near-interpolating solutions:
$\sigma^2$ collapses toward $\mathrm{RSS}/n$, distorting uncertainty
quantification while $\beta$ point estimates remain essentially correct.
This is shared by the $L_{1/2}$ and Dirichlet–Laplace priors (note, §4.5).
The effective lever is the $\sigma^2$ prior — use one that *vanishes at
the origin* ($a_1 \ge 1$). Posterior mean of $\sigma^2$ in an experiment
with $n = 100$, $p = 500$, $\sigma = 1$:

| $\mathrm{InvGamma}(a_1, b_1)$ | $(10^{-3}, 10^{-3})$ | $(1, 1)$ | $(2, 2)$ | $(5, 5)$ | $(10, 10)$ |
| --- | --- | --- | --- | --- | --- |
| posterior mean of $\sigma^2$ | 0.018 | 0.33 | 0.41 | 0.54 | 0.65 |

Hence the default `a1 = b1 = 1`. Full anchoring in $p \gg n$ requires a
fairly informative $\sigma^2$ prior or external knowledge of $\sigma^2$.

## Performance

End-to-end throughput (iterations/sec; CPU, float32, 16-core machine,
torch MKL 8 threads, JAX CPU-only; benchmark script:
`test_JAX/_bench_cpu_vs_torch.py`):

| regime | method | PyTorch | JAX (`lax.scan`) | JAX / torch |
| --- | --- | --- | --- | --- |
| $n{=}100,\ p{=}500$ | fast | 1163 | 3704 | **3.2×** |
| $n{=}100,\ p{=}4000$ | fast | 793 | 1008 | **1.3×** |
| $n{=}800,\ p{=}800$ | fast | 229 | 332 | **1.4×** |
| $n{=}2000,\ p{=}500$ | direct | 742 | 374 | 0.5× |

JAX wins where per-iteration work is small/medium (jit fusion of the
S3–S5 elementwise chain); PyTorch's multithreaded MKL keeps the edge in
the large-`direct` regime dominated by big Cholesky factorizations. The
PyTorch version additionally offers a CUDA path for large problems.

## Repository layout

```
GS_LH/          PyTorch package (driver + S1/S2/S3-S5 step modules)
GS_LH_JAX/      JAX package (float32, chunked lax.scan driver)
test/           PyTorch test suite & benchmarks
test_JAX/       JAX test suite, JAX-vs-PyTorch cross-check & CPU benchmark
On the Connection between Dirichlet–Laplace Prior and L1_2 Prior.md
                the mathematical note (DL at a = 3/2  <=>  L1/2)
```

## Running the tests

Plain scripts (no pytest needed), run from the repo root:

```bash
python test/_test_GS_LH.py          # PyTorch end-to-end (recovery, methods, GPU smoke)
python test_JAX/_test_GS_LH.py      # JAX end-to-end
python test_JAX/_test_beta.py       # 200k-draw moment checks of the beta sampler
python test_JAX/_test_sigma2.py     # 200k-draw InvGamma moment checks
python test_JAX/_test_vs_torch.py   # JAX (float32) vs PyTorch (float64) cross-validation
python test_JAX/_bench_cpu_vs_torch.py   # CPU float32 benchmark
```

## References

- Ke, X., & Fan, Y. (2024). Bayesian L1/2 Regression. *Journal of
  Computational and Graphical Statistics*. DOI: 10.1080/10618600.2024.2374579.
- Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2015).
  Dirichlet–Laplace priors for optimal shrinkage. *JASA*, 110(512), 1479–1490.
- Gruber, L., Kastner, G., Bhattacharya, A., Pati, D., Pillai, N. S., &
  Dunson, D. B. (2025). Correction: A note on simulation methods for the
  Dirichlet-Laplace prior. *JASA*, 120(551), 2011–2014.
- Bhattacharya, A., Chakraborty, A., & Mallick, B. K. (2016). Fast sampling
  with Gaussian scale mixture priors in high-dimensional regression.
  *Biometrika*, 103(4), 985–991.
- Nishimura, A., & Suchard, M. A. (2022). Prior-preconditioned conjugate
  gradient method for accelerated Gibbs sampling in "large n, large p"
  Bayesian sparse regression. *JASA*, 118(544), 2468–2481.

## License

MIT — see [LICENSE](LICENSE).
