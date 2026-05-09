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
- **CMC diagnostics:** `check_convergence`, `create_diagnostics_dict`, `log_analysis_summary`, `get_convergence_recommendations`, `DEFAULT_MIN_ESS` — structured convergence checking with recommendations (`126c7b2`, `5aab5da`).
- **CMC I/O save pipeline:** `save_shard_results`, `save_samples_npz`, `save_fitted_data_npz`, `save_parameters_json`, `save_diagnostics_json`, `save_all_results`, `samples_to_arviz` in `optimization/cmc/io.py` — full homodyne parity (`6e5cde9`).
- **CMC combination dispatch:** `estimate_contrast_offset_from_data` and `combination_method` routing in consensus aggregation (`964ec65`).
- **CMC chain_method:** `chain_method` propagated from `CMCConfig` through `SamplingPlan` and `NUTSSampler`; `vectorized` added to `_VALID_CHAIN_METHOD` (`5aab5da`, `452cfd2`, `7d7bffc`).
- **CMC bimodality detection:** `BimodalResult`, `detect_bimodal`, `check_shard_bimodality`, `ModeCluster`, `BimodalConsensusResult`, `summarize_cross_shard_bimodality`, `cluster_shard_modes` (`15bc947`).
- **NLSQ residuals_normalized:** Exposed normalized residual array in `NLSQResult`; parity with homodyne (`3d9a458`).
- **CMA-ES BIPOP restart:** BIPOP (bi-population) restart strategy for CMA-ES global optimization (`3d9a458`).

### Improved
- **Coverage:** Improved test coverage from 84% to 98% (`91d9ab8`).
- **Architecture docs:** Five architecture docs (overview, NLSQ, CMC, data-handler, physical-model) updated with three-brain consensus review — corrected 24 documented gaps (`db9f2e6`, `3d9a458`).

## [2.0.0] - 2026-01-24
*Initial release (v2.0).*
