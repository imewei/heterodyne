# Heterodyne QA Report

## Performance Optimization Session — 2026-03-13

| Round | Debug | Review | Commits | Modules | Status |
|-------|-------|--------|---------|---------|--------|
| 1 | 0 new findings (perf changes verified) | In progress | Pending | core, optimization, config | Smoke tests PASS |

## Verification Summary

- **Ruff:** All checks passed
- **Mypy:** 103 pre-existing errors in 21 files (none introduced by perf changes)
- **Smoke tests:** 2,578 passed, 0 failures (122.83s)
- **Unit + regression:** 2,350 passed, 0 failures (208.44s)
- **Gradient safety:** 77 gradient-specific tests pass; jacfwd vs jacobian numerically identical (max diff: 4.4e-15)

## Changes Made

### Performance Optimizations (14 bottlenecks resolved)

| ID | Severity | Files | Description |
|----|----------|-------|-------------|
| B001 | CRITICAL | `core/jax_backend.py` | `jax.jacobian` -> `jax.jacfwd` (211x Jacobian speedup) |
| B002 | CRITICAL | `core/physics_nlsq.py` | Cached Jacobian closure + jacfwd |
| B003 | CRITICAL | `optimization/nlsq/core.py` | `joint_residual_fn` routed through vmap `compute_multi_angle_residuals` |
| B004 | CRITICAL | `optimization/nlsq/core.py` | `_make_numpy_residual_fn` pre-captured JAX arrays + vectorized scatter |
| B006 | HIGH | `config/parameter_manager.py` | Cached `varying_indices`/`varying_names`/`fixed_indices` |
| B007 | HIGH | `config/parameter_manager.py` | Cached `get_full_values()` with invalidation |
| B009 | HIGH | `optimization/nlsq/adapter.py` | Reuse optimizer's final residuals |
| B011 | MEDIUM | `core/jax_backend.py` | Hessian: `jacfwd(grad(...))` forward-over-reverse |
| B012 | MEDIUM | `core/physics_cmc.py` | Removed inner `@jax.jit` for XLA fusion |
| B013 | MEDIUM | `core/physics_utils.py` | Removed 5 inner `@jax.jit` on primitives |
| B014 | MEDIUM | `core/models.py` | Vectorized `_expand_to_full` scatter |
| B017 | MEDIUM | `core/theory.py` | `compute_batch_chi_squared` delegates to vmap |
| B019 | MEDIUM | `optimization/cmc/sampler.py` | Vectorized `jax.random.split(key, n)` |
| B022 | LOW | `config/parameter_manager.py` | `frozenset` cache key |
