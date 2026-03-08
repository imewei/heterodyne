# CMC Fitting Architecture

## Overview

The CMC (Consensus Monte Carlo) subsystem provides Bayesian posterior inference
for the heterodyne model parameters using NumPyro's NUTS sampler. It is the
second stage of the analysis pipeline, warm-started from the NLSQ result. A
Z-space reparameterization reduces posterior correlation for the three
power-law parameter pairs (D0/alpha, v0/beta), and a runtime backend selector
dispatches to sequential CPU, multi-process CPU, or parallel GPU execution.

---

## Component Map

```
optimization/cmc/
├── core.py               # fit_cmc_jax(): main entry point
├── model.py              # NumPyro model definitions (reparam and non-reparam)
├── reparameterization.py # Z-space transforms for power-law pairs
├── priors.py             # Prior construction from ParameterRegistry
├── sampler.py            # NUTS configuration wrappers
├── diagnostics.py        # validate_convergence() via ArviZ
├── data_prep.py          # c2 preparation and sigma estimation
├── config.py             # CMCConfig dataclass
├── results.py            # CMCResult dataclass
├── io.py                 # Posterior serialization
└── backends/
    ├── base.py           # MCMCBackend protocol + select_backend()
    ├── cpu_backend.py    # CPUBackend: sequential NUTS chains
    ├── gpu_backend.py    # GPUBackend: pmap parallel chains
    └── worker_pool.py    # WorkerPoolBackend (importable, not auto-selected)
```

---

## Execution Flow

```
fit_cmc_jax(model, c2_data, phi_angle, config, nlsq_result)
        │
        ├─ estimate_sigma(c2_data)     # diagonal noise estimate if sigma=None
        │
        ├─ select_backend(config)
        │     ├── GPUBackend           if GPU detected
        │     └── CPUBackend           otherwise (default)
        │
        ├─ ReparamConfig + compute_t_ref(dt, t_max)
        │     reference time = geometric mean of dt and t_max
        │
        ├─ transform_nlsq_to_reparam_space(nlsq_params, reparam_config)
        │     converts NLSQ solution to Z-space for warm start
        │
        ├─ get_heterodyne_model_reparam(...)   # NumPyro model (reparameterized)
        │   or get_heterodyne_model(...)       # NumPyro model (physics space)
        │
        ├─ backend.run(numpyro_model, config, rng_key, init_params)
        │     runs NUTS sampling
        │
        ├─ transform_to_physics_space(samples, reparam_config)
        │     converts Z-space samples back to physics parameters
        │
        └─ validate_convergence(idata)
              R-hat, ESS, BFMI checks via ArviZ
```

---

## NumPyro Model

The NumPyro model is constructed in `model.py`. Priors are read directly from
`ParameterRegistry` for each varying parameter:

```python
# For each varying parameter:
sample = numpyro.sample(name, dist.Uniform(info.min_bound, info.max_bound))
```

Parameters with `log_space=True` in the registry (D0_ref, D0_sample, v0) are
sampled in log space to cover multiple orders of magnitude efficiently.

The likelihood is a Normal observation:

```python
c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
numpyro.sample("obs", dist.Normal(c2_model, sigma), obs=c2_data)
```

---

## Z-Space Reparameterization

Power-law pairs (D0, alpha) form banana-shaped posteriors because `D0 * t^alpha`
is approximately constant at the data's characteristic time scale. NUTS
struggles to explore this geometry efficiently.

The reparameterization replaces (D0, alpha) with (D_ref, alpha) where:

```
D_ref = D0 * t_ref^alpha
```

`t_ref` is the geometric mean of `dt` and `t_max`. At this reference time the
product D_ref is well-constrained by data, making the posterior approximately
elliptical and NUTS-friendly.

Three pairs are reparameterized independently, controlled by `ReparamConfig`:

| Flag | Pair |
|---|---|
| `enable_d_ref` | D0_ref / alpha_ref |
| `enable_d_sample` | D0_sample / alpha_sample |
| `enable_v_ref` | v0 / beta |

The transforms `transform_nlsq_to_reparam_space` and
`transform_to_physics_space` handle the forward and inverse mappings so that
the NLSQ warm start and returned posterior samples are both expressed in the
original physics-space units.

---

## Backend Selection

`select_backend(config)` in `backends/base.py` inspects `jax.devices()` at
runtime:

```
jax.devices()
    │
    ├── any GPU present? → GPUBackend
    └── CPU?             → CPUBackend
```

| Backend | Strategy | When to use |
|---|---|---|
| `CPUBackend` | Sequential NUTS per chain | CPU (default) |
| `GPUBackend` | `pmap` across GPU devices | GPU available |
| `WorkerPoolBackend` | Multi-process, one process per chain | Manual selection only |

All backends implement the `MCMCBackend` protocol, which exposes a single
`run(model, config, rng_key, init_params)` method returning a dict of sample
arrays shaped `(num_samples * num_chains,)`.

---

## CMCConfig

| Field | Default | Description |
|---|---|---|
| `num_chains` | 4 | Number of independent MCMC chains |
| `num_samples` | 1000 | Posterior samples per chain |
| `num_warmup` | 500 | NUTS warm-up (adaptation) steps |
| `target_accept_prob` | 0.8 | NUTS dual-averaging target |
| `max_tree_depth` | 10 | NUTS tree depth limit |
| `use_reparameterization` | True | Enable Z-space reparameterization |

---

## Convergence Diagnostics

`validate_convergence(idata)` converts NumPyro samples to an ArviZ
`InferenceData` object and runs:

| Diagnostic | Threshold | Meaning |
|---|---|---|
| R-hat | > 1.01 → WARNING | Chain mixing; 1.0 is perfect convergence |
| ESS (bulk) | < 100 → WARNING | Effective sample size |
| ESS (tail) | < 100 → WARNING | Tail effective sample size |
| BFMI | < 0.2 → WARNING | Bayesian Fraction of Missing Information (divergences) |

---

## CMCResult

| Field | Type | Description |
|---|---|---|
| `samples` | `dict[str, np.ndarray]` | Posterior samples per parameter |
| `parameter_names` | `list[str]` | Names in canonical order |
| `diagnostics` | `CMCDiagnostics` | R-hat, ESS, divergences, wall time |
| `inference_data` | `az.InferenceData` | Full ArviZ object for downstream analysis |
| `converged` | `bool` | True if all R-hat and ESS thresholds passed |
| `nlsq_result` | `NLSQResult \| None` | Source warm-start (if provided) |
