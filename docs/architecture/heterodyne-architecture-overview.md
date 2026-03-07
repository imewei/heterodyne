# Heterodyne Package Architecture Overview

## Overview

`heterodyne` is a JAX-based Python package for two-component heterodyne X-ray
Photon Correlation Spectroscopy (XPCS) analysis. It fits the two-time
correlation function c2(t1, t2) to a 14-parameter model (PNAS Eq. S-95) using
a two-stage pipeline: GPU-accelerated non-linear least squares (NLSQ) warm-start
followed by Bayesian posterior sampling via NumPyro NUTS.

---

## Module Map

```
heterodyne/
├── core/               # Physics kernels and model abstractions
│   ├── jax_backend.py       # JIT-compiled c2 computation (stateless)
│   ├── models.py            # TwoComponentModel, ReducedModel, create_model()
│   ├── theory.py            # Transport integral helpers
│   └── physics*.py          # Supporting physics utilities
│
├── config/             # Parameter metadata and space management
│   ├── parameter_registry.py  # Immutable ParameterRegistry (MappingProxyType)
│   ├── parameter_names.py     # ALL_PARAM_NAMES canonical ordering
│   ├── parameter_manager.py   # ParameterManager: vary flags, bounds, defaults
│   └── parameter_space.py     # ParameterSpace for multi-angle scaling
│
├── data/               # I/O and preprocessing
│   ├── xpcs_loader.py        # HDF5/NPZ/MAT loading → XPCSData
│   ├── angle_filtering.py    # Phi-angle range selection
│   ├── preprocessing.py      # PreprocessingPipeline (normalize, outlier removal)
│   └── memory_manager.py     # MemoryManager: budget tracking, chunk sizing
│
├── optimization/
│   ├── nlsq/           # Non-linear least squares
│   │   ├── wrapper.py         # NLSQWrapper: retry with perturbation
│   │   ├── adapter.py         # ScipyNLSQAdapter / NLSQAdapter (JAX)
│   │   └── validation/        # InputValidator, ResultValidator, FitQualityValidator
│   │
│   └── cmc/            # Consensus/MCMC Bayesian sampling
│       ├── core.py            # fit_cmc_jax() entry point
│       ├── model.py           # NumPyro model definition
│       ├── reparameterization.py  # Z-space transforms for power-law pairs
│       ├── backends/          # CPUBackend, GPUBackend, WorkerPoolBackend
│       ├── diagnostics.py     # ArviZ R-hat, ESS convergence checks
│       └── sampler.py         # NUTS configuration wrappers
│
├── viz/                # Visualization
│   ├── diagnostics.py        # Posterior / residual diagnostic plots
│   ├── datashader_backend.py # Large-array rasterization
│   └── validation.py         # Parameter validation plots
│
└── cli/                # Command-line interface
    ├── commands.py           # Click command group
    └── config_generator.py   # YAML config scaffolding
```

---

## Analysis Pipeline

```
Raw file (HDF5/NPZ/MAT)
        │
        ▼
  XPCSDataLoader.load()
        │  validates shape, dtype, finiteness
        ▼
  AngleFiltering / PreprocessingPipeline
        │  phi-range selection, outlier removal, normalization
        ▼
  InputValidator.validate()
        │  data, bounds, initial-param checks
        ▼
  NLSQWrapper.fit()          ← ScipyNLSQAdapter (default) or NLSQAdapter (JAX)
        │  retry-with-perturbation up to max_retries
        ▼
  ResultValidator + FitQualityValidator
        │  chi2, uncertainty, bounds-proximity checks
        ▼
  fit_cmc_jax()              ← NumPyro NUTS via backend selection
        │  warm-started from NLSQ result
        ▼
  validate_convergence()     ← ArviZ R-hat, ESS
        ▼
  CMCResult (posterior samples, diagnostics)
```

---

## Design Principles

**JAX-first, stateless physics.**
`compute_c2_heterodyne` in `core/jax_backend.py` is a pure function decorated
with `@jax.jit`. It accepts a flat 14-element parameter array and returns the
c2 matrix. No mutable state. All JAX transformations (jit, vmap, grad,
jacobian) apply directly.

**Immutable parameter configuration.**
`ParameterRegistry._parameters` is a `MappingProxyType`. Bounds, defaults, and
prior statistics cannot be mutated at runtime. This prevents accidental
configuration drift between the NLSQ and CMC stages.

**Analysis mode factory.**
`create_model(mode)` in `core/models.py` returns a `TwoComponentModel` for
`"two_component"` (all 14 parameters free) or a `ReducedModel` that freezes
inactive parameters at canonical defaults. Registered modes: `"static_ref"`,
`"static_both"`, `"two_component"`.

**Backend selection at runtime.**
`select_backend(config)` in `optimization/cmc/backends/base.py` inspects
`jax.devices()` and returns a `CPUBackend` (sequential), `WorkerPoolBackend`
(multi-process, CPU with >= 3 chains), or `GPUBackend` (parallel) without
requiring caller changes.

**Units.**
All quantities use Angstroms: q in Å⁻¹, D0 in Å²/s^α, velocities in Å/s.
Wavelength follows λ = 12.398 / E[keV] Å.

---

## Key Entry Points

| Task | Function / Class |
|---|---|
| Load data | `XPCSDataLoader(path).load()` or `load_xpcs_data(path)` |
| Build model | `create_model("two_component")` |
| NLSQ fit | `NLSQWrapper(param_names).fit(residual_fn, p0, bounds, config)` |
| Bayesian fit | `fit_cmc_jax(model, c2_data, phi_angle, config, nlsq_result)` |
| Physics kernel | `compute_c2_heterodyne(params, t, q, dt, phi_angle)` |
| Parameter info | `DEFAULT_REGISTRY["D0_ref"]` |
