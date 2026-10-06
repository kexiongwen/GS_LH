# GS_LH — Gibbs Sampler for Bayesian L1/2 Regression

Fast Gibbs samplers for the Bayesian $L_{1/2}$ regression of
**Ke & Fan (2024, JCGS)**, in **PyTorch** (`GS_LH/`) and **JAX**
(`GS_LH_JAX/`), together with a mathematical note
(`On the Connection between Dirichlet–Laplace Prior and L1_2 Prior.md`)
showing that the $L_{1/2}$ prior is exactly the Dirichlet–Laplace prior
of Bhattacharya et al. (2015) at concentration $a = 3/2$.

A walkthrough with executed examples (simulation, posterior summaries,
both algorithm pairings, standardization, warm restarts, JAX) is in
[`demo.ipynb`](demo.ipynb).

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
- **Synchronization-free Gamma draws on CUDA, decided automatically**:
  both Gamma draws of an iteration (S1 $\sigma^2$, S3 $\lambda$) use the
  identity $\mathrm{Ga}(\nu/2, r) = \chi^2(\nu)/(2r)$ — no host
  synchronization, ~30% faster eager GPU iterations — whenever a one-time
  check (`_accept_chi2`) accepts the half-integer rounding of their
  shapes ($2p + a$ and $a_1 + n/2$, both constant across the run; the
  relative moment bias is then $\le 10^{-2}$, and exactly zero for
  (half-)integer shapes). Refusal is silent (exact Gamma everywhere);
  CPU always uses exact Gamma. Ported from the
  [BFM](https://github.com/kexiongwen/BFM) factor-model implementation.
- **CUDA-graph replay (`graph=True`)**: the whole iteration (S1, S2,
  S3–S5) is captured once as a single CUDA graph and replayed —
  3–9× faster GPU iterations (largest gains at launch-bound small
  regimes; ~4000 it/s at $n{=}100,\ p{=}500$). All sampled systems are
  I + PSD by construction, so the modules run `cholesky_ex` with no
  per-solve flag checks; a periodic finiteness audit (`audit_every`)
  guards the chain instead. Replays advance the RNG normally — a fixed
  seed replays a run bit-exactly. Requires CUDA with the chi-square
  check accepted; on CPU, on refusal, or on any capture failure the
  sampler falls back to the eager loop. Also ported from BFM.
- **JAX version**: fixed single precision (float32); the whole chain runs
  as chunked `jax.lax.scan` — each chunk is one compiled XLA While loop
  with no Python dispatch or host synchronization inside. The chi-square
  Gamma path exists here too, with a role change: `jax.random.gamma` is
  a pure XLA primitive (exact, no host sync), so the approximation is
  never mandatory — on the GPU backend the same `_accept_chi2` rule
  engages it as an accuracy-for-speed tradeoff (~1.6–2.8× faster GPU
  iterations, replacing the exact sampler's per-element rejection
  while-loop inside the scan); non-GPU backends always use exact Gamma.
  With `graph=True` (GPU backend), each chunk's scan While loop is
  additionally recorded as one CUDA graph via XLA command buffers —
  bitwise-identical chains, a further regime-dependent speedup.
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

Developed and tested with Python 3 (conda), `torch 2.14`, `jax 0.11.2`
(identical JAX version on Windows/CPU and WSL2/CUDA). The JAX version
runs CPU-only on native Windows; for GPU use `pip install "jax[cuda12]"`
under WSL2/Linux (verified on an RTX 3060 Ti — see Performance). The
PyTorch version runs on CPU or CUDA.

## Quick start

### PyTorch

```python
import torch
from GS_LH import GS_LH

n, p = 100, 500
X = torch.randn(n, p, dtype=torch.float64)   # float64 recommended
y = torch.randn(n, dtype=torch.float64)

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
| `graph` | `False` | replay the iteration(s) as CUDA graphs (PyTorch: one graph per iteration, needs the chi-square check accepted; JAX: one graph per scan chunk via XLA command buffers; both need a CUDA device and fall back otherwise) |
| `audit_every` | `50` | PyTorch: periodic finiteness audit of the chain state (0 disables); JAX: one finiteness audit per scan chunk (unconditional) |

### PyTorch vs JAX at a glance

| | `GS_LH` (PyTorch) | `GS_LH_JAX` (JAX) |
| --- | --- | --- |
| precision | follows input dtype (float64 recommended/tested) | fixed float32 |
| device | CPU, CUDA | CPU (native Windows); GPU via CUDA — e.g. WSL2 `jax[cuda12]` (tested: RTX 3060 Ti) |
| sampling loop | eager Python loop, or one CUDA-graph replay (`graph=True`) | chunked `jax.lax.scan`, whole chain XLA-compiled, optionally one CUDA graph per chunk (`graph=True`) |
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

All numbers are **end-to-end driver throughput** (kept Gibbs iterations
per second) at **single precision (float32)**: identical regimes and
iteration counts in every column — on CPU both frameworks also run the
same simulated data in one process; throughput does not depend on the
data values — warm-up excluded on both
sides (PyTorch: 30-iteration warm-up run; JAX: AOT compilation not
counted in `runtime_sec`). Machine: 16-core CPU (torch MKL 8 threads),
RTX 3060 Ti 8 GB. The GPU columns are cross-environment on the same
physical card — PyTorch on native Windows CUDA, JAX on WSL2 with
`jax[cuda12]` (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) — so the WSL2
layer only adds overhead to the JAX side. Benchmark scripts:
`test_JAX/_bench_cpu_vs_torch.py` (CPU), `test/_bench_gpu_driver.py`
(PyTorch GPU), `test_JAX/_bench_gpu_driver.py` (JAX GPU).

### 1. CPU (float32)

| regime | method | PyTorch | JAX | JAX / PyTorch |
| --- | --- | --- | --- | --- |
| $n{=}100,\ p{=}500$ | fast | 1116 | 3596 | **3.2×** |
| $n{=}100,\ p{=}4000$ | fast | 755 | 987 | **1.3×** |
| $n{=}800,\ p{=}800$ | fast | 231 | 331 | **1.4×** |
| $n{=}2000,\ p{=}500$ | direct | 697 | 344 | 0.49× |
| $n{=}8000,\ p{=}500$ | direct | 521 | 296 | 0.57× |

### 2. GPU (float32, RTX 3060 Ti)

All `test_JAX/` suites pass unchanged on the GPU backend; posterior
means agree with the CPU/float64 reference chains. The two PyTorch
columns are one process, one card: the eager loop, and `graph=True`
(CUDA-graph replay; the one-time capture is excluded from
`runtime_sec`, mirroring the JAX AOT exclusion). The two JAX columns
are the plain chunked-scan runners and `graph=True` (XLA command
buffers: each chunk's scan While loop recorded as one CUDA graph —
bitwise-identical chains). All GPU columns include the chi-square
Gamma path (auto-accepted at these $n/p$).

| regime | method | PyTorch eager | PyTorch graph | JAX plain | JAX graph |
| --- | --- | --- | --- | --- | --- |
| $n{=}100,\ p{=}500$ | fast | ~440 | ~4000 | ~4200 | ~5100 |
| $n{=}100,\ p{=}4000$ | fast | ~440 | ~3300 | ~3400 | ~3850 |
| $n{=}800,\ p{=}800$ | fast | ~430 | ~1270 | ~950 | ~1420 |
| $n{=}2000,\ p{=}500$ | direct | ~445 | ~2050 | ~1250 | ~1170 |
| $n{=}8000,\ p{=}500$ | direct | ~450 | ~1930 | ~900 | ~1680 |

### Reading the tables

- **CPU, framework vs framework:** JAX leads at small/medium
  per-iteration work — XLA fuses the S3–S5 elementwise chain inside
  `lax.scan` (the standalone shrinkage step alone is ~7× faster than
  eager PyTorch). PyTorch's multithreaded MKL keeps the edge in the
  large-`direct` regime, which is dominated by one big Cholesky
  factorization per iteration.
- **GPU, framework vs framework:** with both frameworks in their graph
  modes it is essentially level: JAX-graph leads at the small `fast`
  regimes (~1.1–1.3×), PyTorch-graph leads at the `direct` regimes
  (~1.2–1.8×). All GPU columns include the chi-square Gamma path
  (auto-accepted at these $n/p$): on the JAX side it replaces the exact
  sampler's per-element rejection while-loop inside the scan (~1.6–2.8×
  over exact Gamma); the eager PyTorch figures include it plus the
  sync-free `cholesky_ex` factorizations (~1.2–1.3× over the
  exact-Gamma, raising-cholesky path they replaced).
- **GPU, graph modes:** the two `graph=True`s are not symmetric in
  spirit. The eager PyTorch driver is launch-latency bound — a flat
  ~440 it/s at every regime — so CUDA-graph replay is transformative
  (3–9×, largest at launch-bound small regimes; at $n{=}800,\ p{=}800$
  the $n \times n$ Cholesky dominates and the gain is smallest). The
  JAX plain runners already execute each chunk as one compiled XLA
  While loop, so XLA command buffers only shave the intra-loop launch
  gaps: a modest, regime-dependent gain (about $+5\%$–$+55\%$ at the
  `fast` regimes, $\sim 2\times$ at $n{=}8000,\ p{=}500$ direct, and
  slightly negative at $n{=}2000,\ p{=}500$ direct) — but with
  bitwise-identical chains, so it costs nothing to try per call.
- **GPU vs CPU:** with the chi-square path engaged, JAX-GPU now matches
  JAX-CPU even at the smallest regime ($n{=}100,\ p{=}500$) and beats
  it 2.9–3.8× at the larger regimes; PyTorch-GPU with `graph=True` is
  above PyTorch-CPU at every regime.
- GPU throughputs vary ±10–15% between runs (clock/thermal state);
  "~" marks the mean of repeated runs. On a partly occupied 8 GB card,
  export `XLA_PYTHON_CLIENT_PREALLOCATE=false` to skip XLA's default
  75 %-of-memory preallocation (the allocator's startup retries are
  noisy but harmless).

## Repository layout

```
GS_LH/          PyTorch package (driver + S1/S2/S3-S5 step modules)
GS_LH_JAX/      JAX package (float32, chunked lax.scan driver)
test/           PyTorch test suite & benchmarks
test_JAX/       JAX test suite, JAX-vs-PyTorch cross-check & benchmarks
demo.ipynb      executed walkthrough (simulation, summaries, both methods, JAX)
On the Connection between Dirichlet–Laplace Prior and L1_2 Prior.md
                the mathematical note (DL at a = 3/2  <=>  L1/2)
LICENSE         MIT
```

## Running the tests

Plain scripts (no pytest needed), run from the repo root:

```bash
python test/_test_GS_LH.py          # PyTorch end-to-end (recovery, methods, GPU smoke)
python test/_test_beta.py           # PyTorch 200k-draw moment checks of both beta samplers
python test/_test_sigma2.py         # PyTorch 200k-draw InvGamma moment checks
python test/_test_shrinkage.py      # PyTorch inverse-Gaussian / S3-S5 smoke checks
python test/_test_chi2.py           # chi-square Gamma path + _accept_chi2 auto-decision
python test/_test_graph.py          # CUDA-graph mode: quality, fallbacks, reproducibility
python test/_verify_review_fixes.py # w0 validation, non-finite-state audits, graph verbose line
python test_JAX/_test_GS_LH.py      # JAX end-to-end
python test_JAX/_test_beta.py       # 200k-draw moment checks of the beta sampler
python test_JAX/_test_sigma2.py     # 200k-draw InvGamma moment checks
python test_JAX/_test_vs_torch.py   # JAX (float32) vs PyTorch (float64) cross-validation
python test_JAX/_test_chi2.py       # JAX chi-square path + _accept_chi2 (GPU sections need jax[cuda12], e.g. WSL2)
python test_JAX/_test_graph.py      # JAX graph mode (XLA command buffers): bitwise identity, fallbacks
python test_JAX/_bench_cpu_vs_torch.py   # CPU float32 benchmark (torch vs JAX)
python test/_bench_gpu_driver.py         # PyTorch-GPU float32 benchmark (CUDA)
python test_JAX/_bench_gpu_driver.py     # JAX-GPU float32 benchmark (jax[cuda12], e.g. WSL2)
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
