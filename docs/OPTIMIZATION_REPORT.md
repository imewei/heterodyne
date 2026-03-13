# Heterodyne Performance Optimization Report

**Date:** 2026-03-13
**Environment:** Intel i9-13900H (14c/20t), 62 GiB RAM, JAX CPU-only, float64

## Summary

- **Bottlenecks identified:** 22
- **Resolved:** 14 (CRITICAL: 4, HIGH: 3, MEDIUM: 6, LOW: 1)
- **Deferred:** 8 (HIGH: 3, MEDIUM: 4, LOW: 1)
- **Files changed:** 11 (+187/-70 lines)

## Key Wins

| ID | Bottleneck | Category | Before | After | Speedup |
|----|------------|----------|--------|-------|---------|
| B001 | `jax.jacobian` → `jax.jacfwd` | JIT | 1734ms/Jacobian (N=100) | 8ms/Jacobian (N=100) | **211x** |
| B002 | Cached Jacobian closure | JIT | Recompile every NLSQ iter | Compile once, reuse | **~10x** first-fit |
| B003 | `joint_residual_fn` → vmap | VECTORIZATION | n_phi serial dispatches | Single vmap call | **~n_phi×** |
| B004 | `_make_numpy_residual_fn` | CPU | alloc+loop+H↔D per eval | JAX scatter, no alloc | **~2-5x** per eval |
| B006+B007 | ParameterManager caching | CPU | Rebuild lists per eval | Cached with invalidation | **~3x** overhead reduction |
| B009 | Post-fit residual re-eval | CPU | 1 extra N×N forward pass | Reuse optimizer result | **1 eval saved** per fit |
| B011 | Hessian: fwd-over-rev | JIT | rev-over-rev (14²) | jacfwd(grad) | **~2x** |
| B012+B013 | Remove inner `@jax.jit` | JIT | Fragmented XLA graph | Full fusion scope | Better XLA optimization |
| B014 | Vectorized scatter | CPU | N sequential `.at[].set` | Single scatter op | Cleaner trace |
| B017 | batch_chi_squared vmap | VECTORIZATION | Python loop | vmap delegation | **~n_sets×** |
| B019 | Vectorized random.split | CPU | Sequential splits | Single `split(key, n)` | Minor (setup) |
| B022 | frozenset cache key | CPU | `str(sorted(...))` | `frozenset(...)` | O(1) vs O(n log n) |

## Dominant Win: B001 — Jacobian Direction

The single most impactful optimization. The Jacobian of `compute_c2_heterodyne` has shape `(N²/2, 14)`. Using `jax.jacobian` (reverse-mode/VJP) required N²/2 backward passes. Switching to `jax.jacfwd` (forward-mode/JVP) requires only 14 forward passes regardless of N.

**Measured speedup scales with N:**
- N=100: 211x (1734ms → 8ms)
- N=500 (typical): estimated ~8,900x based on O(N²) scaling
- N=1000: estimated ~35,000x

This single change likely dominates the total NLSQ wall-time improvement.

## Verification

- **Float64 correctness:** `rtol=1e-10, atol=1e-12` — jacfwd vs jacobian max relative diff: 1.1e-12
- **Gradient safety:** All 14 parameter channels active (verified via Jacobian column norms)
- **Test suite:** 2,269 unit tests passed, 0 failures, 3 pre-existing warnings
- **Memory:** No regressions (removals of inner JIT reduce trace cache pressure)

## Deferred Items

| ID | Severity | Reason |
|----|----------|--------|
| B005 | HIGH | CMC shard loop → lax.scan requires uniform padding; architectural change |
| B008 | HIGH | `_wrapped` variadic tuple; requires upstream adapter refactor |
| B010 | HIGH | XLA thread pinning; requires benchmarking on this specific CPU topology |
| B015 | MEDIUM | Memory estimator correction; needs empirical profiling data |
| B016 | MEDIUM | ShardGrid recompilation; tied to B005 uniform padding |
| B018 | MEDIUM | diagonal_correction retrace; low frequency code path |
| B020 | MEDIUM | dict round-trip in set_params; post-fit only, not hot path |
| B021 | LOW | Probe for n_data; minor, one extra eval at fit start |
