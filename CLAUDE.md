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
├── core/               # Physics engine & JAX primitives
├── optimization/
│   ├── nlsq/           # Primary: trust-region L-M optimizer
│   │   ├── strategies/ #   Execution strategies (stratified, chunked, streaming, OOC)
│   │   └── validation/ #   Input, bounds, convergence, fit quality checks
│   └── cmc/            # Secondary: Consensus Monte Carlo (NumPyro NUTS)
│       └── backends/   #   Execution backends (CPU, multiprocessing, PBS, pjit)
├── config/             # YAML config (ConfigManager), parameter_registry
├── data/               # HDF5 loading (XPCSDataLoader)
├── cli/                # Command-line interface & orchestration
├── viz/                # MCMC diagnostics, comparison, dashboard, ArviZ
├── device/             # CPU/NUMA detection, XLA flag configuration
├── io/                 # File I/O utilities
├── utils/              # Logging, checkpoints, misc
└── runtime/            # Shell completion system
```

### Two-Path Integral Architecture

The physics model uses two distinct integral evaluation paths:

1. **Meshgrid path** (NLSQ): `core/jax_backend.py` builds a full N×N time-integral matrix via `create_time_integral_matrix()`. Fast for JIT-compiled least-squares but O(N²) memory.

2. **Element-wise path** (CMC): `core/physics_cmc.py` uses `ShardGrid` + `precompute_shard_grid()` for O(n_pairs) cumsum lookup. No N×N matrix — designed for per-shard NUTS evaluation.

Both paths share primitives from `core/physics_utils.py` (trapezoid_cumsum, rate functions).

## Key Files for Common Tasks

| Task | Primary files |
|------|---------------|
| Physics model | `core/physics_utils.py` (shared primitives), `core/theory.py` (c1/c2/g2), `core/physics.py` (constants/bounds) |
| Two-path integrals | `core/jax_backend.py` (meshgrid), `core/physics_cmc.py` (element-wise), `core/physics_nlsq.py` (NLSQ adapters) |
| Parameter system | `config/parameter_registry.py` (registry), `core/models.py` (ParameterInfo/ParameterSpace) |
| Model wrapper | `core/heterodyne_model.py` (HeterodyneModel + ParameterManager) |
| Scaling modes | `core/scaling_utils.py`, `optimization/nlsq/fourier_reparam.py` |
| NLSQ fitting | `optimization/nlsq/core.py` (fit_nlsq_jax), `optimization/nlsq/adapter.py` (NLSQAdapter entry) |
| NLSQ strategies | `optimization/nlsq/strategies/stratified_ls.py`, `hybrid_streaming.py`, `out_of_core.py` |
| NLSQ config | `optimization/nlsq/config.py`, `optimization/nlsq/fallback_chain.py`, `optimization/nlsq/recovery.py` |
| CMA-ES | `optimization/nlsq/cmaes_wrapper.py` |
| CMC fitting | `optimization/cmc/core.py` (fit_cmc_jax), `optimization/cmc/model.py`, `optimization/cmc/sampler.py` |
| CMC priors | `optimization/cmc/priors.py`, `optimization/cmc/reparameterization.py` |
| CMC config | `optimization/cmc/config.py` |
| CLI dispatch | `cli/commands.py` (orchestrator), `cli/optimization_runner.py` |
| MCMC viz | `viz/mcmc_diagnostics.py`, `viz/mcmc_dashboard.py`, `viz/mcmc_arviz.py` |
| Add physics param | `config/parameter_registry.py`, `core/physics.py`, `core/models.py` |

## Data Flow

```
YAML → ConfigManager → XPCSDataLoader(HDF5) → HeterodyneModel → NLSQ/CMC → Result(JSON+NPZ)
```

## 14 Physics Parameters

All 14 physics parameters + 2 scaling (contrast, offset) per angle, organized in five groups:

**Reference transport** — `J_r(t) = D0_ref * t^alpha_ref + D_offset_ref`

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| D0_ref | Reference diffusion prefactor | 1e4 | Å²/s^(α+1) |
| alpha_ref | Reference transport exponent | 0.0 | — |
| D_offset_ref | Reference transport rate offset | 0.0 | Å²/s |

**Sample transport** — `J_s(t) = D0_sample * t^alpha_sample + D_offset_sample`

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| D0_sample | Sample diffusion prefactor | 1e4 | Å²/s^(α+1) |
| alpha_sample | Sample transport exponent | 0.0 | — |
| D_offset_sample | Sample transport rate offset | 0.0 | Å²/s |

**Velocity** — `v(t) = v0 * t^beta + v_offset`

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| v0 | Velocity prefactor | 1e3 | Å/s^(β+1) |
| beta | Velocity exponent (0 = constant velocity) | 0.0 | — |
| v_offset | Velocity offset (negative for reversal) | 0.0 | Å/s |

**Sample fraction** — `f_s(t) = clip(f0 * exp(f1 * (t - f2)) + f3, 0, 1)`

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| f0 | Fraction amplitude | 0.5 | — |
| f1 | Exponential rate (0 = constant fraction) | 0.0 | s⁻¹ |
| f2 | Time shift | 0.0 | s |
| f3 | Baseline offset | 0.0 | — |

**Flow angle**

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| phi0 | Flow angle offset relative to q-vector | 0.0 | degrees |

Units: All in Angstroms (Å). q in Å⁻¹, D₀ in Å²/s^(α+1), velocities in Å/s.

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

1. **Never subsample data** — full precision always.
2. **CPU-only** — no GPU support.
3. **JAX-first** — numerical core in JAX, NumPy only at I/O boundaries.
4. **Explicit imports** — `from heterodyne.optimization import fit_nlsq_jax`.
5. **Use NLSQ 0.6.10+** — heterodyne's `curve_fit()` with internal memory selection.
6. **Narrow exceptions** — specific types (`OSError`, `ValueError`, `KeyError`) at function boundaries; broad `except Exception` only at top-level dispatchers with `log_exception()`.
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
