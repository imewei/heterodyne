# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Heterodyne: CPU-optimized JAX package for heterodyne X-ray Photon Correlation Spectroscopy (XPCS) analysis under nonequilibrium conditions. Two-component correlation model with 14 physics parameters + 2 scaling parameters per angle.

**Stack:** Python 3.12+, JAX >=0.8.2 (CPU-only), NumPy >=2.3, NLSQ >=0.6.10

## Commands

```bash
make dev          # Install with dev deps
make test         # All tests
make test-smoke   # Critical tests (~30s-2min)
make test-fast    # Exclude slow tests (~5-10min)
make test-unit    # Unit tests only
make test-nlsq    # NLSQ optimization tests
make test-cmc     # CMC optimization tests
make quality      # format + lint + type-check
make quick        # format + smoke tests (fast iteration)
make docs         # Build Sphinx documentation
make docs-serve   # Serve docs locally at :8000
```

### Single test patterns

```bash
pytest tests/unit/test_nlsq_core.py -v          # Single file
pytest tests/unit/test_physics.py::TestClass     # Single class
pytest -m "not slow" tests/                      # Skip slow markers
```

## Architecture

```
heterodyne/
├── core/               # Physics & JAX primitives (two-path integral)
│   ├── jax_backend.py       # NLSQ meshgrid path: cumsum → N×N matrix
│   ├── physics_cmc.py       # CMC element-wise path: ShardGrid + O(n_pairs) cumsum
│   ├── physics_utils.py     # Shared: trapezoid_cumsum, create_time_integral_matrix, smooth_abs
│   ├── physics_nlsq.py      # NLSQ residual/Jacobian adapters
│   ├── physics.py           # Constants, bounds, validation (PhysicsConstants)
│   ├── models.py            # ParameterInfo, ParameterSpace
│   ├── heterodyne_model.py  # HeterodyneModel with ParameterManager
│   ├── theory.py            # c1, c2, g2 correlation calculations
│   ├── scaling_utils.py     # Per-angle contrast/offset scaling
│   └── fitting.py           # Curve fitting interface
├── optimization/
│   ├── nlsq/           # Primary: Trust-region L-M optimizer
│   │   ├── core.py              # fit_nlsq_jax(), fit_nlsq_multi_phi()
│   │   ├── adapter.py           # NLSQAdapter (recommended entry)
│   │   ├── fallback_chain.py    # OptimizationStrategy enum, fallback logic
│   │   ├── recovery.py          # 3-attempt error recovery
│   │   ├── cmaes_wrapper.py     # CMA-ES global optimization
│   │   ├── fourier_reparam.py   # Fourier angular reparameterization
│   │   ├── strategies/
│   │   │   ├── stratified_ls.py     # Primary: stratified LS + anti-degeneracy
│   │   │   ├── hybrid_streaming.py  # Large datasets
│   │   │   └── out_of_core.py       # Out-of-core JTJ accumulation
│   │   └── validation/             # Input, bounds, convergence, fit quality
│   └── cmc/            # Secondary: Consensus Monte Carlo (NumPyro)
│       ├── core.py          # fit_cmc_jax(), fit_cmc_sharded()
│       ├── model.py         # NumPyro model definition
│       ├── sampler.py       # SamplingPlan, NUTS execution
│       ├── priors.py        # build_default_priors(), build_log_space_priors()
│       ├── reparameterization.py  # t_ref, log-space priors
│       └── backends/        # multiprocessing, PBS, pjit
├── config/             # YAML config (ConfigManager), parameter_registry.py
├── data/               # HDF5 loading (XPCSDataLoader)
├── cli/                # Command-line interface
│   ├── commands.py          # dispatch_command() orchestrator
│   ├── config_handling.py   # Device config, CLI overrides
│   ├── data_pipeline.py     # Data loading, angle filtering
│   ├── optimization_runner.py # NLSQ/CMC execution, warm-start
│   └── result_saving.py     # JSON/NPZ serialization
├── viz/                # MCMC diagnostics, comparison, dashboard, ArviZ
├── device/             # CPU/NUMA detection, XLA flag configuration
├── io/                 # File I/O utilities
├── utils/              # Logging, checkpoints, misc
└── runtime/            # Shell completion system
```

## Two-Path Integral Architecture

The physics model uses two distinct integral evaluation paths:

1. **Meshgrid path** (NLSQ): `jax_backend.py` builds a full N×N time-integral matrix via `create_time_integral_matrix()`. Fast for JIT-compiled least-squares but O(N²) memory.

2. **Element-wise path** (CMC): `physics_cmc.py` uses `ShardGrid` + `precompute_shard_grid()` for O(n_pairs) cumsum lookup. No N×N matrix — designed for per-shard NUTS evaluation.

Both paths share primitives from `physics_utils.py` (trapezoid_cumsum, rate functions).

## Analysis Modes — 14 Physics Parameters

All 14 physics parameters + 2 scaling (contrast, offset) per angle:

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| D0_ref, D0_sample | Diffusion coefficients | 1e4 | Å²/s^α |
| alpha_ref, alpha_sample | Anomalous exponents | 0.0 | — |
| D_offset_ref, D_offset_sample | Diffusion offsets | 0.0 | Å² |
| v0 | Velocity amplitude | 1e3 | Å/s |
| v_offset | Velocity offset | 0.0 | Å/s |
| t0_ref, t0_sample | Onset times | varies | s |
| sigma_ref, sigma_sample | Width parameters | varies | s |
| q_power_ref, q_power_sample | q-dependence exponents | 2.0 | — |

Units: All in Angstroms (Å). q in Å⁻¹, D₀ in Å²/s^α, velocities in Å/s.

## Data Flow

```
YAML → ConfigManager → XPCSDataLoader(HDF5) → HeterodyneModel → NLSQ/CMC → Result(JSON+NPZ)
```

## NLSQ Optimization

```python
from heterodyne.optimization.nlsq import fit_nlsq_jax

result = fit_nlsq_jax(data, config, use_adapter=True)
```

### Per-Angle Mode

```yaml
optimization:
  nlsq:
    anti_degeneracy:
      per_angle_mode: "auto"  # "auto", "constant", "individual", "fourier"
```

| Mode | Behavior |
|------|----------|
| auto | n_phi >= 3: averaged scaling, else individual |
| constant | Fixed scaling from quantile estimation |
| individual | Independent contrast/offset per angle |
| fourier | Truncated Fourier series for angular variation |

### CMA-ES for Multi-Scale Problems

```yaml
optimization:
  nlsq:
    cmaes:
      enable: true
      preset: "cmaes-global"
      refine_with_nlsq: true
      nlsq_warmstart: true
      warmstart_auto_skip: true
```

## CMC (Consensus Monte Carlo)

```python
from heterodyne.optimization.nlsq import fit_nlsq_jax
from heterodyne.optimization.cmc import fit_cmc_jax

nlsq_result = fit_nlsq_jax(data, config)
cmc_result = fit_cmc_jax(data, config, nlsq_result=nlsq_result)
```

### CMCConfig Attribute Names

Use the current names (not legacy):
- `target_accept_prob` (NOT `target_accept`)
- `max_r_hat` (NOT `r_hat_threshold`)
- `nlsq_prior_width_factor` (NOT `prior_width_factor`)

`from_dict()` handles legacy keys; internal code must use new names.

### Shard Size

```yaml
optimization:
  cmc:
    sharding:
      max_points_per_shard: "auto"  # ALWAYS use auto
```

**WARNING:** NUTS is O(n) per leapfrog step. Never use 100K+ shard size.

### JIT Cache in Workers

In JAX 0.8+, env var alone does NOT enable persistent cache. Workers must call:

```python
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
```

## Critical Rules

1. **Never subsample data** — full precision always
2. **CPU-only** — no GPU support
3. **JAX-first** — numerical core in JAX, NumPy only at I/O boundaries
4. **Explicit imports** — `from heterodyne.optimization import fit_nlsq_jax`
5. **Use NLSQ 0.6.10+** — heterodyne's `curve_fit()` with internal memory selection
6. **Narrow exceptions** — specific types (`OSError`, `ValueError`, `KeyError`) at function boundaries; broad `except Exception` only at top-level dispatchers with `log_exception()`
7. **Gradient-safe floors** — Use `jnp.where(x > eps, x, eps)` instead of `jnp.maximum(x, eps)`. `jnp.maximum` zeros the gradient below the floor, stalling NLSQ Jacobian and NUTS leapfrog.
8. **Float64 before JAX import** — `JAX_ENABLE_X64=1` must be set before first JAX import. `heterodyne/__init__.py` and `cli/main.py` both call `os.environ.setdefault("JAX_ENABLE_X64", "1")`. Workers re-set in multiprocessing since spawn-mode starts fresh.
9. **Dual prior system** — `parameter_registry.py` (prior_mean/prior_std) AND `parameter_space.py` (`_DEFAULT_PRIOR_SPECS`) must stay in sync. Registry consumed by `cmc/priors.py`; `_DEFAULT_PRIOR_SPECS` by `parameter_space.py:_default_prior()`.
10. **Never use analytical expressions of integrals** — always use numerical integration (e.g., `trapezoid_cumsum`). The transport coefficient integrals have no closed-form solutions for the general power-law parameterization; analytical shortcuts introduce silent approximation errors.

## Logging

```python
from heterodyne.utils.logging import get_logger, log_phase

logger = get_logger(__name__)

with log_phase("Optimization"):
    result = fit_nlsq_jax(data, config)
```

## CLI Entry Points

| Command | Purpose |
|---------|---------|
| `heterodyne` | Main analysis (NLSQ/CMC) |
| `heterodyne-config` | Config generation/validation |
| `heterodyne-config-xla` | XLA device configuration |
| `heterodyne-post-install` | Shell completion setup |
| `heterodyne-cleanup` | Remove shell completion files |
| `heterodyne-validate` | System validation |

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Add physics param | `config/parameter_registry.py`, `core/physics.py`, `core/models.py` |
| Modify fitting | `core/fitting.py`, `core/heterodyne_model.py` |
| NLSQ config | `optimization/nlsq/config.py` |
| NLSQ strategies | `optimization/nlsq/strategies/stratified_ls.py`, `hybrid_streaming.py`, `out_of_core.py` |
| CMC model/priors | `optimization/cmc/model.py`, `cmc/priors.py` |
| CMC config | `optimization/cmc/config.py` |
| CMC sampling | `optimization/cmc/sampler.py` (SamplingPlan, NUTS) |
| CLI dispatch | `cli/commands.py` (orchestrator) |
| MCMC viz | `viz/mcmc_diagnostics.py`, `viz/mcmc_dashboard.py`, `viz/mcmc_arviz.py` |
