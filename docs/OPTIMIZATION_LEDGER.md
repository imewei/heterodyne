# Optimization Ledger — Heterodyne XPCS

**Environment:** Intel i9-13900H (14c/20t hybrid), 62 GiB RAM, JAX CPU-only, float64
**Date:** 2026-03-12

## Findings

| ID | File:Line | Category | Severity | Owner | Status | Description |
|----|-----------|----------|----------|-------|--------|-------------|
| B001 | `jax_backend.py:355` | JIT | CRITICAL | jax | VERIFIED | `jax.jacobian` → `jax.jacfwd` — **211x speedup** (1734ms→8ms at N=100) |
| B002 | `physics_nlsq.py:206` | JIT | CRITICAL | jax | VERIFIED | Cached Jacobian closure + `jacfwd` — eliminates recompilation |
| B003 | `core.py:318-349` | VECTORIZATION | CRITICAL | python | VERIFIED | Routed through `compute_multi_angle_residuals` vmap — eliminates n_phi serial dispatches |
| B004 | `core.py:936-950` | CPU | CRITICAL | python | VERIFIED | Pre-captured JAX arrays + vectorized scatter; eliminates per-eval alloc + Python loop |
| B005 | `physics_cmc.py:562-581` | JIT | HIGH | jax | DEFERRED | Python loop over shards breaks XLA fusion in NUTS hot path — requires uniform shard padding + lax.scan |
| B006 | `parameter_manager.py:93-98` | CPU | HIGH | python | VERIFIED | Cached `varying_indices`/`varying_names`/`fixed_indices` with invalidation |
| B007 | `parameter_manager.py:150` | CPU | HIGH | python | VERIFIED | Cached `get_full_values()` with invalidation on mutation |
| B008 | `adapter.py:248,541` | CPU | HIGH | python | PARTIAL | `_wrapped` tuple→array: removed `list()` alloc in 6 strategy files; core adapter API constraint remains |
| B009 | `adapter.py:398-400` | CPU | HIGH | python | VERIFIED | Reuses optimizer's final residuals instead of re-evaluating |
| B010 | — | MULTIPROCESSING | HIGH | systems | DEFERRED | No XLA thread pinning; E-cores mixed with P-cores — requires hardware-specific benchmarking |
| B011 | `jax_backend.py:507` | JIT | MEDIUM | jax | VERIFIED | `jax.hessian` → `jacfwd(grad(...))` — forward-over-reverse |
| B012 | `physics_cmc.py:145,183` | JIT | MEDIUM | jax | VERIFIED | Removed inner `@jax.jit` — XLA sees full computation graph |
| B013 | `physics_utils.py:165-286` | JIT | MEDIUM | jax | VERIFIED | Removed 5 inner `@jax.jit` on primitives (kept `safe_sinc`) |
| B014 | `models.py:318-320` | CPU | MEDIUM | python | VERIFIED | Vectorized scatter via `_active_indices_array` |
| B015 | `memory.py:150` | MEMORY | MEDIUM | systems | DEFERRED | Memory estimator omits per-call N×N model cost — overhead factor absorbs it in practice |
| B016 | `physics_cmc.py:104` | JIT | MEDIUM | jax | CLOSED | `ShardGrid.n_pairs` never read during JIT trace — recompilation driven by idx array shapes, not n_pairs |
| B017 | `theory.py:377-381` | VECTORIZATION | MEDIUM | jax | VERIFIED | Delegated to `batch_chi_squared` vmap — eliminates Python loop |
| B018 | `diagonal_correction.py:588,593` | JIT | MEDIUM | debugger | VERIFIED | `jit(vmap(...))` closure → `lru_cache`-backed factory — eliminates per-call recompilation |
| B019 | `sampler.py:864-876` | CPU | MEDIUM | python | VERIFIED | Vectorized `jax.random.split(key, n)` — single call |
| B020 | `models.py:141-163` | CPU | MEDIUM | python | CLOSED | `params_to_dict`/`dict_to_params` — convenience API only, not on any hot path |
| B021 | `adapter.py:517` | CPU | LOW | python | DEFERRED | Probe residual for n_data — single call per fit, negligible impact |
| B022 | `parameter_manager.py:284` | CPU | LOW | python | VERIFIED | `frozenset` cache key — O(1) hash vs O(n log n) sort+str |

## Legend
- **Category:** MEMORY, CPU, JIT, VECTORIZATION, IO, MULTIPROCESSING, ALGORITHMIC
- **Severity:** CRITICAL > HIGH > MEDIUM > LOW
- **Status:** OPEN → IN_PROGRESS → VERIFIED → MERGED | DEFERRED
