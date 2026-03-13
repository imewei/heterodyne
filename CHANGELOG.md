# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Performance

- **Jacobian:** Switch to forward-mode AD (`jacfwd`) — 211x speedup over reverse-mode (`81e0455`).
- **Core:** Remove inner `@jax.jit` nesting, vectorize scatter operations, `vmap` batch chi-squared (`577a1d1`).
- **NLSQ:** `vmap` multi-angle residuals, JAX-native residual computation, reuse final residuals (`b2276f8`).
- **NLSQ:** Vectorize `jax.random.split` in parameter perturbation (`fe1de53`).
- **Config:** Cache `ParameterManager` properties with `frozenset` cache key (`c6054a5`).
- **Strategies:** Cache batch JIT compilation, remove `list()` allocation in `_wrapped` (`5efbefd`).

### Fixed

#### Gradient Safety
- Replace `jnp.maximum` with `jnp.where` on all gradient-critical paths — prevents zero-gradient stalling in NLSQ Jacobian and NUTS leapfrog (`2393297`).
- Gradient-safe `t=0` floor in g1 visualization helpers (`da7b8d6`).
- Fix velocity field floor inconsistency and `safe_divide` `sign(0)` NaN (`4c54cef`, `b628ac5`).

#### Data Pipeline
- Fix 3-D time-window indexing that collapsed to 1-D (`8cb3551`).
- Fix mask semantics, NaN outlier count, enforce float64 at load boundary (`45a4290`).

#### NLSQ Optimization
- Prevent config mutation in recovery retries (`d583c7b`).
- Mark `ChunkedStrategy` first-call failure as partial failure (`43683eb`).
- Harden caches and eliminate mutable state across core and optimization (`116ed49`).

#### CMC / Bayesian
- Pass fitted contrast/offset to NumPyro model in warm-start path (`c8115e6`).
- Use canonical `target_accept_prob` key in worker pool (`4a383ab`).
- Correct credible interval key lookup in MCMC summary I/O (`5cbf62f`).
- Warn on `ParameterSpace` fallback in CMC workers (`20fb120`).

#### Config
- Add `update_optimization_config()` for safe nested config updates; fix deep-copy mutation bug; rename deprecated CMC field (`704f84e`).

#### Code Quality
- Resolve all 82 mypy type errors across core, optimization, utils, viz (`51ff7cd`).
- Narrow exception handling to specific types across all modules — `cli`, `core`, `config`, `data`, `device`, `io`, `optimization`, `viz`, `utils`, `runtime` (QA rounds 1–2).
- Replace hardcoded BFMI threshold with `BFMI_THRESHOLD` constant (`f5a360b`).
- Clean up `json_safe` import path, fix matplotlib deprecations (`d222bae`).

### Added
- Regression tests for gradient safety and 3-D time windowing (`d0e32b3`).
- Sphinx documentation tree (`2596017`).
- Performance optimization ledger and QA report (`1b1bbbe`).

### Improved
- **Coverage:** Improved test coverage from 84% to 98% (`91d9ab8`).

## [2.0.0] - 2026-01-24
*Initial release (v2.0).*
