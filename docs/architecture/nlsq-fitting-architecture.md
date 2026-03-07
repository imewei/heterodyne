# NLSQ Fitting Architecture

## Overview

The NLSQ subsystem provides GPU-accelerated non-linear least squares fitting
of the heterodyne c2 model. It is the warm-start stage of the full pipeline,
producing parameter estimates and uncertainties that initialize the subsequent
Bayesian (CMC) sampling. The subsystem is layered: a validation pass bookends
the fit, and a retry wrapper handles convergence failures automatically.

---

## Component Map

```
optimization/nlsq/
├── wrapper.py              # NLSQWrapper: retry-with-perturbation
├── adapter.py              # ScipyNLSQAdapter, NLSQAdapter (JAX)
├── adapter_base.py         # NLSQAdapterBase protocol
├── config.py               # NLSQConfig, NLSQValidationConfig
├── results.py              # NLSQResult dataclass
├── multistart.py           # Multi-start grid search
└── validation/
    ├── input_validator.py  # InputValidator (pre-fit)
    ├── result.py           # ResultValidator (post-fit), ValidationReport
    └── fit_quality.py      # FitQualityValidator (chi2, bounds proximity)
```

---

## Execution Flow

```
InputValidator.validate(data, params, bounds)
        │  checks: empty data, NaN/Inf values, bounds shape,
        │          inverted bounds, initial params out of bounds
        │  returns ValidationReport; abort if is_valid=False
        ▼
NLSQWrapper.fit(residual_fn, initial_params, bounds, config)
        │
        ├─ attempt 0: original params (no perturbation)
        ├─ attempt 1..max_retries: perturbed params
        │     noise = Uniform(-1,1) * perturb_scale * attempt * (upper-lower)
        │     seed = 42 + attempt (reproducible)
        │     params clipped to bounds after perturbation
        │
        └─ returns best NLSQResult (lowest final_cost across all attempts)
                │
                ▼
        ResultValidator.validate(result)
                │  checks: convergence flag, chi2 thresholds,
                │          relative uncertainty, parameter correlations, NaN/Inf
                ▼
        FitQualityValidator.validate(result)
                │  checks: reduced chi2 (warn >= 10, fail >= 100)
                │          bounds proximity (< 0.5% of span = WARNING)
                ▼
        NLSQResult
```

---

## Backend Adapters

`NLSQWrapper` delegates to one of two adapters selected via `use_jax`:

| Adapter | Backend | Notes |
|---|---|---|
| `ScipyNLSQAdapter` | `scipy.optimize.least_squares` | Default; robust, no JIT |
| `NLSQAdapter` | `nlsq` library (JAX-accelerated) | `use_jax=True`; GPU-capable |

Both implement `NLSQAdapterBase` which exposes a single `fit()` method with
the same signature, making the wrapper backend-agnostic.

---

## NLSQWrapper Retry Logic

```python
total_attempts = max_retries + 1          # default: 4 attempts
for attempt in range(total_attempts):
    params = _perturbed_params(initial, lower, upper, attempt)
    result = adapter.fit(residual_fn, params, bounds, config)
    best_result = update_best(best_result, result)
    if result.success:
        return result
return best_result                        # best by final_cost, even if all failed
```

Key design decisions:
- Attempt 0 always uses the unperturbed initial parameters.
- Perturbation is proportional to the parameter-space width `(upper - lower)`,
  so it adapts to each parameter's scale automatically.
- The random seed is deterministic (`42 + attempt`), ensuring reproducibility.
- Even if all attempts fail, the result with the lowest `final_cost` is
  returned, so callers always receive a result rather than an exception.

---

## NLSQResult Dataclass

Key fields returned from every fit:

| Field | Type | Description |
|---|---|---|
| `parameters` | `np.ndarray` | Fitted parameter values |
| `parameter_names` | `list[str]` | Names in canonical order |
| `uncertainties` | `np.ndarray \| None` | 1-sigma from covariance diagonal |
| `covariance` | `np.ndarray \| None` | Full parameter covariance matrix |
| `final_cost` | `float \| None` | Residual sum of squares at solution |
| `reduced_chi_squared` | `float \| None` | chi² / degrees of freedom |
| `success` | `bool` | Whether optimizer converged |
| `message` | `str` | Optimizer status message |
| `n_iterations` | `int \| None` | Number of optimizer iterations |

---

## Validation Severity Levels

All validators produce a `ValidationReport` containing `ValidationIssue`
instances with one of three severity levels:

| Severity | Effect | Examples |
|---|---|---|
| `ERROR` | Sets `is_valid = False` | NaN in data, inverted bounds, optimization failed |
| `WARNING` | Logged but does not block | Reduced chi2 > 10, large relative uncertainty, near-bound solution |
| `INFO` | Logged as informational | Good fit quality (chi2 in expected range) |

---

## Configuration

`NLSQConfig` controls the optimizer:

| Field | Default | Description |
|---|---|---|
| `max_iterations` | 1000 | Maximum optimizer iterations |
| `tolerance` | 1e-8 | Gradient/function tolerance |
| `method` | `"trf"` | Trust-region reflective (scipy) |
| `loss` | `"linear"` | Residual loss function |

`NLSQValidationConfig` controls post-fit thresholds:

| Field | Default | Description |
|---|---|---|
| `chi2_warn_high` | 5.0 | chi2_red above this → WARNING |
| `chi2_fail_high` | 50.0 | chi2_red above this → ERROR |
| `chi2_warn_low` | 0.1 | chi2_red below this → WARNING (overfitting) |
| `max_relative_uncertainty` | 1.0 | Relative uncertainty above this → WARNING |
| `correlation_warn` | 0.95 | \|r\| above this → WARNING |
