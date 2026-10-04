# On the Connection between the Dirichlet–Laplace Prior and the $L_{1/2}$ Prior

> Priors: the Dirichlet–Laplace prior of Bhattacharya, Pati, Pillai & Dunson (2015, *JASA*) and the Bayesian $L_{1/2}$ prior of Ke, X. & Fan, Y. (2024). Bayesian L1/2 Regression. *Journal of Computational and Graphical Statistics*. DOI: 10.1080/10618600.2024.2374579.
> Sampler: the $\lambda$-reparameterization of Gruber, Kastner, Bhattacharya, Pati, Pillai & Dunson (2025, *JASA* Correction; their $\lambda_j$ written $\nu_j$), with an additional collapsed update for $\sigma^2$ ($\beta$ integrated out).
> This note shows that at Dirichlet concentration $a = 3/2$ the two priors — and their Gibbs samplers — coincide under the explicit change of variables $\nu_j = v_j/\lambda^2$, $\psi_j = \tau_j^2/v_j^2$. The only substantive difference is the treatment of the global shrinkage rate: fixed at $1/2$ in the DL prior, learned through the hyperprior $\lambda \sim \mathrm{Gamma}(c, d)$ in the $L_{1/2}$ prior.

---

## 1. Model Setting

Throughout this note, both priors are placed on the same high-dimensional linear regression model

$$
Y = X\beta + \varepsilon, \qquad
\varepsilon \sim \mathcal N_n(0,\, \sigma^2 I_n),
$$

i.e., $Y \mid \beta, \sigma^2 \sim \mathcal N_n(X\beta,\, \sigma^2 I_n)$, where:

- $X$ is the $n \times p$ design matrix; $p > n$ is allowed (the high-dimensional regime, where OLS is infeasible);
- $Y$ is the $n$-vector of responses;
- $\beta = (\beta_1,\dots,\beta_p)^\top$ is the vector of regression coefficients; $\sigma^2$ is the error variance.

We write $\tilde\beta_j = \beta_j/\sigma$ for the error-standardized coefficient. The Bayesian $L_{1/2}$ variant additionally assumes that $Y$ is centered (no intercept is needed) and that the columns of $X$ are standardized, $\|X_j\|^2 = n$ ($j = 1,\dots,p$).

---

## 2. Bayesian Dirichlet–Laplace Regression

The Dirichlet–Laplace (DL) prior of Bhattacharya et al. (2015), in the variable-selection form of Zhang & Bondell (2018); the sampler below is the $\lambda$-reparameterization of Gruber et al. (2025) — their $\lambda_j$ written $\nu_j$ — with an additional collapsed update for $\sigma^2$ ($\beta$ integrated out).

### 2.1 The Dirichlet–Laplace prior

For $j = 1,\dots,p$:

$$
\beta_j \mid \varphi_j, \kappa, \sigma \sim \mathrm{DE}(\sigma\varphi_j\kappa), \qquad
(\varphi_1, \dots, \varphi_p) \sim \mathrm{Dir}(a, \dots, a), \qquad
\kappa \sim \mathrm{Ga}(pa,\, 1/2),
$$

together with the standard conjugate prior on the error variance,

$$
\sigma^2 \sim \mathrm{IG}(a_1, b_1).
$$

The concentration parameter $a$ controls sparsity (small $a$ pushes most $\varphi_j$ toward $0$), $\varphi_j$ is the local scale of coefficient $j$, and $\kappa$ is the global scale controlling overall shrinkage toward the origin.

*Gamma–Dirichlet reparameterization.* If $(\varphi_1,\dots,\varphi_p) \sim \mathrm{Dir}(a,\dots,a)$ independently of $\kappa \sim \mathrm{Ga}(pa, 1/2)$, then

$$
\nu_j = \varphi_j \, \kappa \;\overset{iid}{\sim}\; \mathrm{Ga}(a,\, 1/2), \qquad j = 1, \dots, p;
$$

conversely, $\kappa = \sum_j \nu_j$ and $\varphi_j = \nu_j / \kappa$.

*Working hierarchy.* Via "normal–exponential mixture = Laplace", the prior can be rewritten in an equivalent normal scale-mixture form in which $\varphi$ and $\kappa$ no longer appear:

$$
\beta_j \mid \sigma^2, \psi_j, \nu_j \sim \mathcal N(0,\, \sigma^2 \psi_j \nu_j^2), \qquad
\psi_j \sim \mathrm{Exp}(1/2), \qquad
\nu_j \sim \mathrm{Ga}(a, 1/2), \qquad
\sigma^2 \sim \mathrm{IG}(a_1, b_1).
$$

Integrating out $\psi_j$ recovers the Laplace kernel $\beta_j \mid \sigma, \nu_j \sim \mathrm{DE}(\sigma\nu_j)$; further summing the $\nu_j$ into $(\varphi, \kappa)$ recovers the original DL form above. Since the prior variance of $\beta_j$ is proportional to $\sigma^2$, shrinkage is expressed relative to the error scale — a practical consequence being that $\sigma^2$ cancels out of the matrix $V$ in the $\beta$ update (S2), which is also what makes the collapsed $\sigma^2$ update (S1) possible.

### 2.2 The Gibbs sampler

Notation: $S = \mathrm{diag}(\psi_1\nu_1^2,\dots,\psi_p\nu_p^2)$, $V = (X^{\top}X + S^{-1})^{-1}$, $\hat\mu = VX^\top Y$. At each iteration, run S1–S4 in order.

**S1. Update $\sigma^2$** (inverse-Gamma; **collapsed** — $\beta$ integrated out)
$$
\sigma^2 \mid \nu, \psi, Y
\sim \mathrm{IG}\Big(a_1 + \frac{n}{2},\; b_1 + \frac12\, Y^\top \Sigma^{-1} Y \Big),
\qquad \Sigma = I_n + X S X^\top .
$$

*Derivation.* Integrating $\beta \sim \mathcal N_p(0,\, \sigma^2 S)$ out of the likelihood gives $Y \mid \sigma^2, \nu, \psi \sim \mathcal N_n(0,\, \sigma^2 \Sigma)$, the key point being that $\Sigma$ does not involve $\sigma^2$; combining with the prior $\sigma^2 \sim \mathrm{IG}(a_1, b_1)$ yields the inverse-Gamma kernel above.

*Computation.* By the Woodbury identity, $\Sigma^{-1} = I_n - XVX^\top$, so no $n \times n$ matrix is needed:
$$
Y^\top \Sigma^{-1} Y = Y^\top Y - Y^\top X \hat\mu,
$$
and the inverted matrix $V$ is the same one used in S2, so the collapsed update is essentially free.

**S2. Sample $\beta$** (multivariate normal, conditional on the freshly drawn $\sigma^2$)
$$
\beta \mid \nu, \psi, \sigma^2, Y
\sim \mathcal N_p\Big(
\hat\mu,\;
\sigma^2 V
\Big).
$$
Steps S1+S2 form a **joint update of the block $(\sigma^2, \beta)$** via
$$
p(\sigma^2, \beta \mid \nu, \psi, Y) = p(\sigma^2 \mid \nu, \psi, Y)\; p(\beta \mid \sigma^2, \nu, \psi, Y)
$$
— marginal first, hence S1 must precede S2.

**S3. Update $\nu_j$**, $j = 1,\dots,p$ (giG; **marginalized over $\psi_j$**)
$$
\nu_j \mid \beta_j, \sigma^2
\sim \mathrm{giG}\Big( a - 1,\; \frac{2|\beta_j|}{\sigma},\; 1 \Big).
$$

*Derivation.* Integrating out $\psi_j$ gives $\tilde\beta_j \mid \nu_j \sim \mathrm{DE}(\nu_j)$, hence
$$
p(\nu_j \mid \tilde\beta_j) \propto \frac{1}{2\nu_j} e^{-|\tilde\beta_j|/\nu_j} \cdot \nu_j^{a-1} e^{-\nu_j/2}
= \nu_j^{(a-1)-1} \exp\Big(-\frac{2|\tilde\beta_j|/\nu_j + \nu_j}{2}\Big),
$$
the giG kernel above.

*Special case $a = 1/2$.* Since $\mathrm{giG}(-\tfrac12, \chi, \psi) = \mathrm{InvGaussian}\big(\sqrt{\chi/\psi},\, \chi\big)$, S3 degenerates to an inverse-Gaussian draw,
$$
\nu_j \mid \beta_j, \sigma^2 \sim \mathrm{InvGaussian}\Big(\sqrt{\frac{2|\beta_j|}{\sigma}},\, \frac{2|\beta_j|}{\sigma}\Big),
$$
and no giG sampler is needed at all.

**S4. Update $\psi_j$**, $j = 1,\dots,p$ (sample the reciprocal from an InvGaussian; **conditional on the freshly drawn $\nu_j$**)
$$
\frac{1}{\psi_j} \,\Big|\, \beta_j, \nu_j, \sigma^2
\sim \mathrm{InvGaussian}\Big( \frac{\sigma \nu_j}{|\beta_j|},\, 1 \Big),
\qquad \text{take the reciprocal to obtain } \psi_j.
$$

*Derivation.* The reciprocal transformation $W = 1/\psi_j$ applied to
$$
p(\psi_j \mid \tilde\beta_j, \nu_j) \propto \psi_j^{-1/2} \exp\Big(-\frac{\tilde\beta_j^2}{2\psi_j\nu_j^2} - \frac{\psi_j}{2}\Big)
$$
yields precisely $W \sim \mathrm{InvGaussian}\big(\nu_j/|\tilde\beta_j|,\, 1\big) = \mathrm{InvGaussian}\big(\sigma\nu_j/|\beta_j|,\, 1\big)$.

Steps S3+S4 form a **joint update of the block $(\nu, \psi)$** via
$$
p(\nu, \psi \mid \beta, \sigma^2) = p(\nu \mid \beta, \sigma^2)\; p(\psi \mid \nu, \beta, \sigma^2)
$$
— marginal first, hence S3 must precede S4 (the sub-steps are not interchangeable).

*Initialization.* Only $\nu$ and $\psi$ need initial values (e.g., ones): S1 samples $\sigma^2$ given $(\nu, \psi)$ only, and S2 then samples $\beta$ given the fresh $\sigma^2$, so $\beta$ and $\sigma^2$ are never initialized.

*(Optional) Recovering the original DL parameters.* Setting, at each iteration,
$$
\kappa = \sum_{j=1}^{p} \nu_j, \qquad \varphi_j = \frac{\nu_j}{\kappa},
$$
yields $(\kappa, \varphi)$ values that automatically follow the correct marginal posteriors $p(\kappa \mid \beta, \sigma^2)$ and $p(\varphi \mid \beta, \sigma^2)$.

---

## 3. Bayesian $L_{1/2}$ Regression

This section condenses Ke & Fan (2024): the Bayesian $L_{1/2}$ prior, placed on the error-standardized coefficients $\tilde\beta_j = \beta_j/\sigma$, and the resulting five-step Gibbs sampler.

### 3.1 The $L_{1/2}$ prior

**Marginal prior.** For $j = 1,\dots,p$, iid:

$$
\pi(\beta_j \mid \lambda, \sigma) = \frac{\lambda^2}{4\sigma}\exp\!\Big(-\lambda\sqrt{\frac{|\beta_j|}{\sigma}}\Big),
$$

i.e., the error-standardized coefficient $\tilde\beta_j = \beta_j/\sigma$ follows the original $L_{1/2}$ prior of Ke & Fan (2024), $\pi(\tilde\beta_j \mid \lambda) = \frac{\lambda^2}{4}\exp\!\big(-\lambda\sqrt{|\tilde\beta_j|}\,\big)$. Up to an additive constant, the induced negative log-prior is $\lambda\sum_j\sqrt{|\beta_j|/\sigma}$ — the $L_{1/2}$ penalty $\lambda\sum_j\sqrt{|\beta_j|}$ applied to the error-standardized coefficients — so posterior modes coincide with $L_{1/2}$-penalized estimates.

**Laplace mixture form.**
$$
\beta_j \mid v_j, \lambda, \sigma \sim \mathrm{DE}\!\Big(\frac{\sigma v_j}{\lambda^2}\Big),
\qquad
v_j \sim \mathrm{Gamma}\!\Big(\frac32,\, \frac14\Big).
$$

**Equivalent normal scale-mixture form** ("normal–exponential mixture = Laplace")
$$
\beta_j \mid \tau_j^2, \lambda, \sigma^2 \sim \mathcal N\!\Big(0,\, \frac{\sigma^2\tau_j^2}{\lambda^{4}}\Big),
\qquad
\tau_j^2 \mid v_j \sim \mathrm{Exp}\!\Big(\frac{1}{2v_j^2}\Big),
\qquad
v_j \sim \mathrm{Gamma}\!\Big(\frac32,\, \frac14\Big).
$$

- Integrating out $\tau_j^2$ recovers the Laplace mixture form; further integrating out $v_j$ recovers the marginal prior above;
- $\tau_j$ is the **local** shrinkage parameter and $1/\lambda^{2}$ is the **global** shrinkage parameter;
- The prior variance of $\beta_j$ is proportional to $\sigma^2$, so shrinkage is expressed relative to the error scale. A practical consequence is that $\sigma^2$ cancels out of the matrix inverse in the $\beta$ update (S2), which is also what makes the collapsed $\sigma^2$ update of S1 possible.

**Hyperpriors.**
$$
\lambda \sim \mathrm{Gamma}(c,\, d),
\qquad
\sigma^2 \sim \mathrm{InvGamma}(a_1,\, b_1).
$$

- $\lambda$ controls global shrinkage and is critical to the prior's performance; placing a hyperprior on it makes global shrinkage self-adaptive (a weakly informative default is a small $c, d$);
- The Jeffreys prior $\pi(\sigma^2) \propto 1/\sigma^2$ is recovered as the limiting case $a_1 = b_1 = 0$ (a weakly informative proper default is $a_1 = b_1 = 0.001$).

### 3.2 The Gibbs sampler

Notation: $\tau^2 = (\tau_1^2,\dots,\tau_p^2)^\top$, $v = (v_1,\dots,v_p)^\top$,
$D_{\tau^2} = \mathrm{diag}(\tau_1^2,\dots,\tau_p^2)$.
Only $\lambda,\ v,\ \tau^2$ need initial values (e.g., ones): S1 samples $\sigma^2$ given $(\tau^2, \lambda)$ only, and S2 then samples $\beta$ given the fresh $\sigma^2$. At each iteration run S1–S5 in order.

**S1. Update $\sigma^2$** (inverse-Gamma; **collapsed** — $\beta$ integrated out)
$$
\sigma^2 \mid \tau^2, \lambda, Y
\sim \mathrm{InvGamma}\Big( a_1 + \frac{n}{2},\; b_1 + \frac12\, Y^\top \Sigma^{-1} Y \Big),
\qquad \Sigma = I_n + \lambda^{-4} X D_{\tau^2} X^\top .
$$

*Derivation.* Given $(\sigma^2, \tau^2, \lambda)$, $\beta \sim \mathcal N_p\big(0,\, \sigma^2 \lambda^{-4} D_{\tau^2}\big)$; integrating $\beta$ out of the likelihood gives $Y \mid \sigma^2, \tau^2, \lambda \sim \mathcal N_n\big(0,\, \sigma^2 \Sigma\big)$, and $\Sigma$ does not involve $\sigma^2$, so the $\mathrm{InvGamma}(a_1, b_1)$ prior is conjugate.

*Computation.* By the Woodbury identity no $n \times n$ matrix is needed:
$$
Y^\top \Sigma^{-1} Y = Y^\top Y - (X^\top Y)^\top \big(X^\top X + \lambda^{4} D_{\tau^2}^{-1}\big)^{-1} (X^\top Y),
$$
and the inverted matrix is the same one used in S2, so the collapsed update is essentially free.

**S2. Sample $\beta$** (multivariate normal, conditional on the freshly drawn $\sigma^2$)
$$
\beta \mid \tau^2, \lambda, \sigma^2, Y
\sim \mathcal N_p\Big(
\big(X^\top X + \lambda^{4} D_{\tau^2}^{-1}\big)^{-1} X^\top Y,\;
\sigma^2 \big(X^\top X + \lambda^{4} D_{\tau^2}^{-1}\big)^{-1}
\Big).
$$

Note that $\sigma^2$ does not appear inside the inverted matrix — the same structure as $V = (X^\top X + S^{-1})^{-1}$ in the DL sampler. Steps S1+S2 form a **joint update of the block $(\sigma^2, \beta)$** via
$$
p(\sigma^2, \beta \mid \tau^2, \lambda, Y) = p(\sigma^2 \mid \tau^2, \lambda, Y)\; p(\beta \mid \sigma^2, \tau^2, \lambda, Y)
$$
— marginal first, hence S1 must precede S2.

**S3. Update $\lambda$** (Gamma; marginalized over $v, \tau^2$)
$$
\lambda \mid \beta, \sigma \sim \mathrm{Gamma}\Big( 2p + c,\; d + \sum_{j=1}^p \sqrt{\frac{|\beta_j|}{\sigma}} \Big).
$$

*Derivation.* Multiplying the marginal prior of Section 3.1 by the hyperprior,
$$
p(\lambda \mid \beta, \sigma) \propto \lambda^{c-1}e^{-d\lambda}\prod_{j=1}^p \frac{\lambda^2}{4\sigma}\, e^{-\lambda\sqrt{|\beta_j|/\sigma}} = \lambda^{2p+c-1}\exp\!\Big(-\Big(d + \sum_{j=1}^p \sqrt{|\beta_j|/\sigma}\Big)\lambda\Big).
$$

**S4. Update $v_j$**, $j = 1,\dots,p$ (sample the reciprocal from an InvGaussian; marginalized over $\tau^2$)
$$
\frac{1}{v_j} \,\Big|\, \beta, \lambda, \sigma
\sim \mathrm{InvGaussian}\Big( \frac{\sqrt{\sigma}}{2\lambda\sqrt{|\beta_j|}},\, \frac{1}{2} \Big),
\qquad \text{take the reciprocal to obtain } v_j.
$$

*Derivation.* From the Laplace mixture form, $p(v_j \mid \beta_j, \lambda, \sigma) \propto v_j^{-1/2}\exp\!\big(-\lambda^2|\beta_j|/(\sigma v_j) - v_j/4\big)$; the reciprocal transformation $W = 1/v_j$ turns this giG kernel into the InvGaussian above.

**S5. Update $\tau_j^2$**, $j = 1,\dots,p$ (sample the reciprocal from an InvGaussian)
$$
\frac{1}{\tau_j^2} \,\Big|\, \beta, \lambda, v_j, \sigma
\sim \mathrm{InvGaussian}\Big( \frac{\sigma}{\lambda^{2} v_j\, |\beta_j|},\, \frac{1}{v_j^2} \Big),
\qquad \text{take the reciprocal to obtain } \tau_j^2.
$$

*Derivation.* $p(\tau_j^2 \mid \beta_j, v_j, \lambda, \sigma) \propto (\tau_j^2)^{-1/2}\exp\!\big(-\lambda^4\beta_j^2/(2\sigma^2\tau_j^2) - \tau_j^2/(2v_j^2)\big)$, a giG kernel in $\tau_j^2$; the reciprocal transformation $W = 1/\tau_j^2$ again yields the InvGaussian above.

---

## 4. The Connection at $a = 3/2$

This section makes the relationship between the two priors precise: on the error-standardized scale $\tilde\beta_j = \beta_j/\sigma$, the $L_{1/2}$ prior is exactly the Dirichlet–Laplace prior at concentration $a = 3/2$, and the two Gibbs samplers are one sampler — except for a single step, the update of the global shrinkage rate.

### 4.1 A common Laplace–gamma mixture

Integrating the exponential mixing variable out of the normal scale-mixture form ("normal–exponential mixture = Laplace"), both priors become Laplace mixtures of the standardized coefficient. For the DL prior, integrating out $\psi_j \sim \mathrm{Exp}(1/2)$ gives

$$
\tilde\beta_j \mid \nu_j \sim \mathrm{DE}(\nu_j), \qquad \nu_j \sim \mathrm{Ga}(a,\, 1/2);
$$

for the $L_{1/2}$ prior, the Laplace mixture form reads

$$
\tilde\beta_j \mid v_j, \lambda \sim \mathrm{DE}\Big(\frac{v_j}{\lambda^2}\Big), \qquad v_j \sim \mathrm{Gamma}\Big(\frac32,\, \frac14\Big).
$$

Now change variables on the $L_{1/2}$ hierarchy, at fixed $\lambda$:

$$
\nu_j = \frac{v_j}{\lambda^2}, \qquad \psi_j = \frac{\tau_j^2}{v_j^2}.
$$

By Gamma scaling (scaling by $1/\lambda^2$ divides the rate by $1/\lambda^2$, so the rate becomes $(1/4)/(1/\lambda^2) = \lambda^2/4$),

$$
\nu_j = \frac{v_j}{\lambda^2} \sim \mathrm{Ga}\Big(\frac32,\, \frac{\lambda^2}{4}\Big);
$$

and $\tau_j^2 \mid v_j \sim \mathrm{Exp}\big(1/(2v_j^2)\big)$ implies $\psi_j = \tau_j^2 / v_j^2 \sim \mathrm{Exp}(1/2)$, independently of $v_j$. The $L_{1/2}$ hierarchy thus coincides, layer by layer, with the DL hierarchy at concentration $a = 3/2$ and gamma rate $r = \lambda^2/4$ in place of $1/2$. The normal kernels agree, since

$$
\psi_j \nu_j^2 = \frac{\tau_j^2}{v_j^2}\cdot\frac{v_j^2}{\lambda^4} = \frac{\tau_j^2}{\lambda^4},
\qquad\text{i.e.,}\qquad
S = \lambda^{-4} D_{\tau^2}.
$$

**Conclusion.** The $L_{1/2}$ prior is the Dirichlet–Laplace prior with concentration $a = 3/2$ and gamma rate $r = \lambda^2/4$ on $\nu_j$. Summing the iid scales by the Gamma–Dirichlet relationship — $\kappa = \sum_{j=1}^p \nu_j \sim \mathrm{Ga}(3p/2,\, \lambda^2/4)$, $\varphi_j = \nu_j/\kappa$ — this reads, in the original DL parameters,

$$
\tilde\beta_j \mid \varphi_j, \kappa \sim \mathrm{DE}(\varphi_j \kappa), \qquad
\varphi \sim \mathrm{Dir}\Big(\frac32, \dots, \frac32\Big), \qquad
\kappa \sim \mathrm{Ga}\Big(\frac{3p}{2},\, \frac{\lambda^2}{4}\Big).
$$

### 4.2 The marginal prior: why $a = 3/2$ is special

For general shape $a$ and rate $r$, mixing $\mathrm{DE}(\nu)$ over $\nu \sim \mathrm{Ga}(a, r)$ gives the Bessel-K marginal

$$
\pi(\tilde\beta) = \int_0^\infty \frac{1}{2\nu}\, e^{-|\tilde\beta|/\nu}\, \frac{r^a}{\Gamma(a)}\, \nu^{a-1} e^{-r\nu}\, d\nu
= \frac{r^a}{\Gamma(a)} \Big(\frac{|\tilde\beta|}{r}\Big)^{(a-1)/2} K_{a-1}\Big(2\sqrt{r |\tilde\beta|}\,\Big),
$$

where the integral is the giG normalizing constant and $K$ is the modified Bessel function of the second kind. This marginal is generally not elementary. Only half-integer orders reduce at all, via $K_{1/2}(z) = \sqrt{\pi/(2z)}\, e^{-z}$ and the recursion in the order; and only at $a = 3/2$ does every algebraic factor cancel: since $\Gamma(3/2) = \sqrt{\pi}/2$,

$$
\pi(\tilde\beta)
= \frac{r^{3/2}}{\Gamma(3/2)} \Big(\frac{|\tilde\beta|}{r}\Big)^{1/4}
\sqrt{\frac{\pi}{4\sqrt{r|\tilde\beta|}}}\; e^{-2\sqrt{r|\tilde\beta|}}
= r\, e^{-2\sqrt{r|\tilde\beta|}}.
$$

Setting $r = \lambda^2/4$ recovers exactly the $L_{1/2}$ prior of Ke & Fan (2024):

$$
\pi(\tilde\beta) = \frac{\lambda^2}{4}\, e^{-\lambda\sqrt{|\tilde\beta|}},
\qquad\text{i.e.,}\qquad
-\log \pi(\tilde\beta) = \mathrm{const} + \lambda \sqrt{|\tilde\beta|}.
$$

Thus the $L_{1/2}$ prior is the unique Dirichlet–Laplace member whose marginal density is elementary — a pure exponential in $\sqrt{|\tilde\beta|}$, free of any algebraic prefactor — and the $L_{1/2}$ penalty is the signature of $a = 3/2$.

*Remark.* The exponential tail is shared by the whole family: $K_\nu(z) \sim \sqrt{\pi/(2z)}\, e^{-z}$ as $z \to \infty$, so every member decays like $e^{-c\sqrt{|\beta|}}$ — heavier than the Lasso's $e^{-c|\beta|}$. The shape $a$ controls the behaviour at the origin. Contrast $a = 1/2$ (DL$_{1/2}$, at the standard rate $r = 1/2$), where $K_{-1/2} = K_{1/2}$ gives

$$
\pi(\tilde\beta) = \frac{1}{2\sqrt{2}}\, |\tilde\beta|^{-1/2}\, e^{-\sqrt{2|\tilde\beta|}}
\;\propto\; |\tilde\beta|^{-1/2}\, e^{-\sqrt{2|\tilde\beta|}}:
$$

the same $e^{-c\sqrt{|\beta|}}$ tail, but an unbounded pole at zero instead of the finite cusp of $a = 3/2$ ($\pi(0) = r < \infty$).

### 4.3 Equivalence of the Gibbs samplers

Set $a = 3/2$ in the DL sampler of Section 2 and allow a general rate $r$ on $\nu_j$ (Section 2 uses $r = 1/2$; the only step affected is S3, whose third giG parameter $1$ becomes $2r$). Under the change of variables of Section 4.1 — a bijection $(v_j, \tau_j^2) \leftrightarrow (\nu_j, \psi_j)$ at fixed $\lambda$ — the two samplers are one sampler.

**S1/S2: identical.** Since $S = \lambda^{-4} D_{\tau^2}$, equivalently $S^{-1} = \lambda^{4} D_{\tau^2}^{-1}$, the inverted matrices coincide:

$$
V = (X^\top X + S^{-1})^{-1} = \big(X^\top X + \lambda^{4} D_{\tau^2}^{-1}\big)^{-1},
\qquad
\Sigma = I_n + X S X^\top = I_n + \lambda^{-4} X D_{\tau^2} X^\top,
$$

so the collapsed $\sigma^2$ update (S1) and the $\beta$ update (S2) are the same distribution written in two notations.

**DL-S3 = $L_{1/2}$-S4.** At $a = 3/2$ with rate $r$, the DL-S3 derivation (marginalized over $\psi_j$) gives

$$
\nu_j \mid \tilde\beta_j \sim \mathrm{giG}\Big(\frac12,\; 2|\tilde\beta_j|,\; 2r\Big).
$$

*Reciprocal duality.* $Y \sim \mathrm{giG}(1/2, \chi, \psi)$ implies $1/Y \sim \mathrm{giG}(-1/2, \psi, \chi) = \mathrm{InvGaussian}\big(\sqrt{\psi/\chi},\, \psi\big)$; with $\chi = 2|\tilde\beta_j|$, $\psi = 2r$,

$$
\frac{1}{\nu_j}\,\Big|\,\tilde\beta_j \sim \mathrm{InvGaussian}\Big(\sqrt{\frac{r}{|\tilde\beta_j|}},\; 2r\Big).
$$

*Scaling law.* $X \sim \mathrm{InvGaussian}(\mu, \lambda_0)$ implies $cX \sim \mathrm{InvGaussian}(c\mu, c\lambda_0)$. Since $1/v_j = \lambda^{-2}\cdot(1/\nu_j)$, applying the scaling law with $c = \lambda^{-2}$ and substituting $r = \lambda^2/4$,

$$
\frac{1}{v_j}\,\Big|\,\tilde\beta_j, \lambda
\sim \mathrm{InvGaussian}\Big(\frac{1}{\lambda^2}\sqrt{\frac{r}{|\tilde\beta_j|}},\; \frac{2r}{\lambda^2}\Big)
= \mathrm{InvGaussian}\Big(\frac{1}{2\lambda\sqrt{|\tilde\beta_j|}},\; \frac12\Big)
= \mathrm{InvGaussian}\Big(\frac{\sqrt{\sigma}}{2\lambda\sqrt{|\beta_j|}},\; \frac12\Big),
$$

which is exactly $L_{1/2}$-S4.

**DL-S4 = $L_{1/2}$-S5.** Since $1/\psi_j = v_j^2 \cdot (1/\tau_j^2)$, applying the same scaling law with $c = v_j^2$ to $L_{1/2}$-S5,

$$
\frac{1}{\psi_j}\,\Big|\,\tilde\beta_j, \nu_j
\sim \mathrm{InvGaussian}\Big(v_j^2 \cdot \frac{\sigma}{\lambda^{2} v_j\, |\beta_j|},\; v_j^2 \cdot \frac{1}{v_j^2}\Big)
= \mathrm{InvGaussian}\Big(\frac{\nu_j}{|\tilde\beta_j|},\; 1\Big)
= \mathrm{InvGaussian}\Big(\frac{\sigma \nu_j}{|\beta_j|},\; 1\Big),
$$

which is exactly DL-S4.

The update order also coincides: both samplers draw the marginal before the conditional within each block — S1 then S2 for $(\sigma^2, \beta)$, and DL-S3/S4, respectively $L_{1/2}$-S4/S5, for $(\nu, \psi)$, respectively $(v, \tau^2)$.

| DL sampler at $a = 3/2$ (rate $r$) | $L_{1/2}$ sampler | relation |
| --- | --- | --- |
| S1: $\sigma^2 \mid \nu, \psi, Y$, collapsed IG | S1: $\sigma^2 \mid \tau^2, \lambda, Y$, collapsed InvGamma | identical, since $S = \lambda^{-4} D_{\tau^2}$ |
| S2: $\beta \mid \nu, \psi, \sigma^2, Y \sim \mathcal N_p(\hat\mu, \sigma^2 V)$ | S2: $\beta \mid \tau^2, \lambda, \sigma^2, Y \sim \mathcal N_p(\hat\mu, \sigma^2 V)$ | same $V$, same $\hat\mu = V X^\top Y$ |
| S3: $\nu_j \mid \tilde\beta_j \sim \mathrm{giG}\big(\frac12,\, 2|\tilde\beta_j|,\, 2r\big)$ | S4: $1/v_j \mid \tilde\beta_j, \lambda \sim \mathrm{InvGaussian}\big(\frac{1}{2\lambda\sqrt{|\tilde\beta_j|}},\, \frac12\big)$ | $\nu_j = v_j/\lambda^2$; reciprocal duality + scaling law |
| S4: $1/\psi_j \mid \tilde\beta_j, \nu_j \sim \mathrm{InvGaussian}\big(\frac{\nu_j}{|\tilde\beta_j|},\, 1\big)$ | S5: $1/\tau_j^2 \mid \tilde\beta_j, \lambda, v_j \sim \mathrm{InvGaussian}\big(\frac{1}{\lambda^2 v_j |\tilde\beta_j|},\, \frac{1}{v_j^2}\big)$ | $1/\psi_j = v_j^2/\tau_j^2$; scaling law |
| — (rate $r$ fixed at $1/2$) | S3: $\lambda \mid \beta, \sigma \sim \mathrm{Gamma}\big(2p + c,\; d + \sum_{j=1}^p \sqrt{|\tilde\beta_j|}\,\big)$ | the one genuinely new step |

**The one new step: $L_{1/2}$-S3.** This is the update of $\lambda$, i.e., of the global rate $r = \lambda^2/4$, which the DL sampler holds fixed at $1/2$ (equivalently, $\lambda \equiv \sqrt{2}$). Marginalizing the $L_{1/2}$ hierarchy down to the elementary marginal of Section 4.2,

$$
p(\lambda \mid \beta, \sigma) \propto \lambda^{2p}\, e^{-\lambda \sum_j \sqrt{|\tilde\beta_j|}}\cdot \lambda^{c-1} e^{-d\lambda},
$$

the Gamma update above (here $c, d$ are the hyperprior parameters of the $L_{1/2}$ prior). Conversely, the DL prior at $a = 3/2$ with a hyperprior on its gamma rate — $\lambda = 2\sqrt{r} \sim \mathrm{Gamma}(c, d)$ — is exactly the $L_{1/2}$ model.

### 4.4 The one substantive difference

Stripped of notation, the two models differ only in the treatment of the global shrinkage rate:

- **DL** fixes the rate of $\nu_j$ at $1/2$ and carries the global scale through $\kappa = \sum_{j=1}^p \nu_j \sim \mathrm{Ga}(pa,\, 1/2)$, with $\varphi = \nu/\kappa$ on the simplex, $\varphi \sim \mathrm{Dir}(a, \dots, a)$; $\kappa$ is learned implicitly through the $\nu_j$ draws (and recovered by summation), but the prior rate $1/2$ is never updated.
- **$L_{1/2}$** frees the rate $r = \lambda^2/4$ and learns it through the hyperprior $\lambda \sim \mathrm{Gamma}(c, d)$ — self-adaptive global shrinkage — at the price of the extra step S3.

*Practical corollary.* At $a = 3/2$ the $\mathrm{giG}(1/2, \cdot, \cdot)$ draw in DL-S3 needs no general giG sampler either: by the reciprocal duality of Section 4.3, for $\nu_j \sim \mathrm{giG}(1/2, \chi, \psi)$ draw

$$
\frac{1}{\nu_j} \sim \mathrm{InvGaussian}\Big(\sqrt{\frac{\psi}{\chi}},\, \psi\Big)
\qquad\text{and invert;}
$$

with $\chi = 2|\tilde\beta_j|$, $\psi = 2r$ this is $\mathrm{InvGaussian}\big(\sqrt{r/|\tilde\beta_j|},\, 2r\big)$ — exactly what $L_{1/2}$-S4 does with $r = \lambda^2/4$ (at the standard DL rate $r = 1/2$: $\mathrm{InvGaussian}\big(1/\sqrt{2|\tilde\beta_j|},\, 1\big)$). This complements the known $a = 1/2$ case, where $\mathrm{giG}(-1/2)$ is directly an InvGaussian and $\nu_j$ itself is drawn without inversion.

---

## 4.5 A practical caveat: the error variance in the $p \gg n$ regime

Both priors of this note couple the coefficient scale to the error scale: $\beta_j = \sigma\tilde\beta_j$ with $\tilde\beta_j$ following a $\sigma$-free density. A consequence worth stating explicitly is that **as $\sigma \to 0$ the effective penalty on the standardized scale rescales perfectly** — the $\beta$-prior offers no resistance to $\sigma^2$ collapsing. If in addition the customary near-Jeffreys choice $\sigma^2 \sim \mathrm{IG}(a_1, b_1)$ with $a_1, b_1 \approx 0$ is used, whose density diverges at the origin, the $\sigma^2$ prior offers no resistance either. Since the $p \gg n$ design provides $p - n$ interpolation directions, the posterior can then concentrate on *near-interpolating solutions*:

$$
\sigma^2 \;\approx\; \frac{\|Y - X\beta\|^2}{n} \;\ll\; \sigma^2_{\mathrm{true}},
$$

while $\beta$ point estimates remain essentially correct — the pathology distorts $\sigma^2$ and uncertainty quantification, not $\hat\beta$. This is a property of the posterior itself (the soft version of the impropriety of the Jeffreys posterior when $p \ge n$), not of any sampler: both the collapsed update (S1) and the standard conditional update $\sigma^2 \mid \beta, w, Y \sim \mathrm{IG}\big(a_1 + \frac{n+p}{2},\, b_1 + \frac{1}{2}(\|Y - X\beta\|^2 + \sum_j \beta_j^2/w_j^2)\big)$ are exact conditionals of the same joint posterior and land in the same degenerate region.

Two remarks on what does *not* cause it, in the spirit of Section 4.4:

- *The learned global rate is not the culprit.* Fixing $\lambda \equiv \sqrt{2}$ (i.e., the DL member at $a = 3/2$, rate $1/2$) collapses *faster*; the learned $\lambda$ of the $L_{1/2}$ prior drifts upward from $\sqrt{2}$ and mildly resists the collapse. The equivalence of Section 4 therefore survives the pathology — but it is shared by both models, and so is the caveat.
- *The initialization does not matter either*; starting the chain at strongly shrinking scales ($w_0 \ll 1$) leads to the same degenerate region.

The effective lever is the $\sigma^2$ prior. In a numerical experiment ($n = 100$, $p = 500$, five signals of magnitudes $0.7$–$2.0$, $\sigma = 1$; $4000$ kept draws after $1000$ burn-in, $\lambda \sim \mathrm{Gamma}(10^{-3}, 10^{-3})$) the posterior mean of $\sigma^2$ was

| $\sigma^2$ prior $\mathrm{IG}(a_1, b_1)$ | $(10^{-3}, 10^{-3})$ | $(1, 1)$ | $(2, 2)$ | $(5, 5)$ | $(10, 10)$ |
| --- | --- | --- | --- | --- | --- |
| posterior mean of $\sigma^2$ | $0.018$ | $0.33$ | $0.41$ | $0.54$ | $0.65$ |

tracking $\mathrm{RSS}/n$ throughout. A weakly informative prior that *vanishes* at the origin ($a_1 \ge 1$, e.g., $a_1 = b_1 = 1$) is therefore advisable in the $p \gg n$ regime; only a fairly informative $\sigma^2$ prior — or external information about $\sigma^2$ — fully anchors the error variance.

---

## Appendix: Distributional Conventions

- $\mathrm{DE}(b)$ (Laplace / double-exponential with scale $b$): $f(y) = \dfrac{1}{2b}e^{-|y|/b}$;
- $\mathrm{Ga}(\alpha, r)$, also written $\mathrm{Gamma}(\alpha, r)$: Gamma with shape $\alpha$ and rate $r$, $f(y) \propto y^{\alpha-1}e^{-ry}$, $y > 0$;
- $\mathrm{Exp}(r) = \mathrm{Ga}(1, r)$;
- $\mathrm{IG}(\alpha, \gamma)$, also written $\mathrm{InvGamma}(\alpha, \gamma)$: inverse-Gamma, $f(y) \propto y^{-\alpha-1}e^{-\gamma/y}$, $y > 0$;
- **Inverse Gaussian**: $Y \sim \mathrm{InvGaussian}(\mu, \lambda_0)$ means
$$
f(y) = \sqrt{\frac{\lambda_0}{2\pi y^3}} \exp\Big(-\frac{\lambda_0 (y-\mu)^2}{2\mu^2 y}\Big), \qquad y > 0;
$$
- **giG** (three-parameter generalized inverse Gaussian): $Y \sim \mathrm{giG}(p, \chi, \psi)$ means
$$
f(y) \propto y^{p-1} \exp\Big(-\frac{\chi/y + \psi y}{2}\Big), \qquad y > 0;
$$
- $\mathrm{Dir}(a,\dots,a)$: Dirichlet with all concentration parameters equal to $a$;
- **Half-integer Bessel identity** (used in Section 4.2): $K_{1/2}(z) = \sqrt{\pi/(2z)}\, e^{-z}$; more generally $K_{-\nu}(z) = K_\nu(z)$ for every order $\nu$.

## References

- Ke, X., & Fan, Y. (2024). Bayesian L1/2 Regression. *Journal of Computational and Graphical Statistics*. DOI: 10.1080/10618600.2024.2374579.
- Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2015). Dirichlet–Laplace priors for optimal shrinkage. *JASA*, 110(512), 1479–1490.
- Gruber, L., Kastner, G., Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2025). Correction: A note on simulation methods for the Dirichlet-Laplace prior. *JASA*, 120(551), 2011–2014.
