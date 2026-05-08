<!-- Package: heterodyne | Last verified: 2026-05-08 -->

# Data Handler Architecture

## Overview

This document covers the path from a YAML configuration and an XPCS data file
on disk to the result files written after fitting. The scope spans the
`heterodyne.config`, `heterodyne.data`, `heterodyne.io`, and `heterodyne.cli`
packages, plus the supporting parameter registry and quality-control
subsystem. The downstream NLSQ and CMC fitters are described in their own
architecture documents; this document treats them as black-box consumers of
preprocessed `XPCSData` and producers of `NLSQResult` / `CMCResult`.

---

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Configuration System](#configuration-system)
- [Component Map](#component-map)
- [Data Loading](#data-loading)
- [Phi Angle Filtering](#phi-angle-filtering)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Quality Control](#quality-control)
- [Dataset Optimization](#dataset-optimization)
- [Caching & Performance](#caching--performance)
- [Memory Manager](#memory-manager)
- [Result Writing](#result-writing)
- [CLI Orchestration](#cli-orchestration)
- [Data Flow Summary](#data-flow-summary)
- [Quick Reference Tables](#quick-reference-tables)
- [Key Files Reference](#key-files-reference)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER ENTRY POINTS                        │
│  heterodyne/ht CLI          XPCSDataLoader(...) Python API  │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  ConfigManager          │  config/manager.py
          │  XPCSDataLoader         │  data/xpcs_loader.py
          │  PreprocessingPipeline  │  data/preprocessing.py
          └────────────┬────────────┘
                       │  XPCSData (c2, t1, t2, q, phi_angles)
          ┌────────────▼────────────┐
          │  fit_nlsq_jax()         │  optimization/nlsq/core.py
          │  fit_cmc_jax()          │  optimization/cmc/core.py
          └────────────┬────────────┘
                       │  NLSQResult + CMCResult
          ┌────────────▼────────────┐
          │  Result Writers         │  heterodyne/io/
          │  JSON + NPZ output      │
          └─────────────────────────┘
```

---

## Configuration System

The configuration system is the entry point for every analysis run. A YAML
file is loaded by `ConfigManager`, parameters are resolved against the
immutable `ParameterRegistry`, and the resulting object is consumed by the
data loader, the fitters, and the result writers.

### YAML Schema

The canonical schema groups all data-acquisition metadata under
`analyzer_parameters` (homodyne parity). Legacy `temporal` and `scattering`
top-level sections are auto-migrated by `ConfigManager`.

```yaml
experimental_data:
  file_path: /path/to/run042.h5
  file_format: hdf5
  cache_file_path: ""              # optional; falls back to data_folder_path
  cache_filename_template: ""      # optional; supports ${var} substitution
  cache_compression: true

analyzer_parameters:
  dt: 0.001                        # time step [s]
  start_frame: 1000                # 1-indexed inclusive
  end_frame: 2000                  # 1-indexed inclusive
  scattering:
    wavevector_q: 0.0054           # [Å⁻¹]
    phi_angles: [0.0, 45.0, 90.0]  # optional
  geometry:
    stator_rotor_gap: 2000000      # [Å] — instrument metadata only

parameters:
  reference:
    D0_ref: {value: 1.0e4, vary: true}
    alpha_ref: {value: 0.0, vary: true}
    D_offset_ref: {value: 0.0, vary: true}
  sample:
    D0_sample: {value: 1.0e4, vary: true}
    alpha_sample: {value: 0.0, vary: true}
    D_offset_sample: {value: 0.0, vary: true}
  velocity:
    v0: {value: 1.0e3, vary: true}
    beta: {value: 0.0, vary: true}
    v_offset: {value: 0.0, vary: true}
  fraction:
    f0: {value: 0.5, vary: true}
    f1: {value: 0.0, vary: true}
    f2: {value: 0.0, vary: true}
    f3: {value: 0.0, vary: true}
  angle:
    phi0: {value: 0.0, vary: true}

optimization:
  method: nlsq                     # "nlsq" or "cmc"
  nlsq: {...}                      # see NLSQ architecture doc
  cmc: {...}                       # see CMC architecture doc

output:
  output_dir: ./output
```

### ConfigManager

`ConfigManager` (`config/manager.py`) is the single source of truth for
configuration access. It deep-copies the input dict on construction, runs
`_normalize_schema()` to migrate deprecated keys, and validates required
sections.

| Constructor / Method | Purpose |
|---|---|
| `ConfigManager(config: dict)` | Direct dict construction (validates) |
| `ConfigManager.from_yaml(path)` | Load from YAML |
| `ConfigManager.from_json(path)` | Load from JSON |
| `ConfigManager.from_dict(config)` | Explicit dict path |
| `load_xpcs_config(path)` | Module-level YAML convenience wrapper |
| `to_yaml(path)` | Serialize back to YAML |

Required sections enforced by `_validate()`: `experimental_data`,
`analyzer_parameters`, `parameters`. Missing sections raise
`ConfigurationError`. The `optimization.method` field, if present, must be
one of `{"nlsq", "cmc"}`.

Public properties (selection):

| Property | Returns | Notes |
|---|---|---|
| `data_file_path` | `Path` | From `experimental_data.file_path` |
| `data_folder_path` | `Path \| None` | Fallback for cache path |
| `file_format` | `str` | Defaults to `"hdf5"` |
| `cache_file_path` | `Path \| None` | Falls back to `data_folder_path` |
| `cache_filename_template` | `str \| None` | `${var}` template syntax |
| `cache_compression` | `bool` | Default `True` |
| `dt`, `start_frame`, `end_frame` | scalars | 1-indexed frame range |
| `time_length`, `t_start` | scalars | Derived (0-indexed for `t_start`) |
| `wavevector_q` | `float` | [Å⁻¹] |
| `phi_angles` | `list[float] \| None` | Optional explicit angles |
| `stator_rotor_gap` | `float \| None` | Geometry metadata only |
| `optimization_method` | `str` | Default `"nlsq"` |
| `nlsq_config`, `cmc_config` | `dict` | Deep-copied on access |
| `output_dir` | `Path` | Default `./output` |
| `get_parameter_value(group, name)` | `float` | Looks up nested config |
| `get_parameter_vary(group, name)` | `bool` | Default `True` |
| `update_optimization_config(section, key, value)` | mutation helper |
| `get_cmc_config()` | merged with sensible defaults (NUTS knobs) |

CMC defaults applied by `_merge_cmc_config()`:
`{num_warmup: 500, num_samples: 1000, num_chains: 4, target_accept_prob: 0.8, max_tree_depth: 10}`.

### Parameter Registry

`ParameterRegistry` (`config/parameter_registry.py`) is an immutable
dataclass that wraps a `MappingProxyType` view of 16 `ParameterInfo`
entries (14 physics + 2 scaling). Both the wrapper and the underlying
mapping are frozen — the `__post_init__` guard re-wraps any mutable
`Mapping` passed in to enforce immutability.

| Method | Returns |
|---|---|
| `__getitem__(name)` | `ParameterInfo` (raises `KeyError` if unknown) |
| `__iter__()` | Names in canonical 14+2 order |
| `__len__()` | `16` |
| `get_defaults()` | `dict[str, float]` |
| `get_bounds()` | `(lower_list, upper_list)` |
| `get_group(group_name)` | `list[ParameterInfo]` for a group |
| `get_varying_indices(vary_flags)` | `list[int]` |
| `get_log_space_names()` | Names that should be sampled in log-space |
| `get_scaling_names()` | `["contrast", "offset"]` |

`ParameterInfo` fields: `name`, `default`, `min_bound`, `max_bound`,
`description`, `unit`, `group`, `vary_default`, `log_space`, `prior_mean`,
`prior_std`, `is_scaling`, `is_physical`, `is_flow`. Helper methods
`validate_value(v)` and `clip_value(v)` are used at config-validation
boundaries.

The module also exports `DEFAULT_REGISTRY` (singleton) and
`SCALING_PARAMS` (a 2-entry view containing only `contrast` and `offset`).

### Parameter Space and Dual Prior System

The project carries a **dual-prior** invariant: two independent default
specifications must stay in sync.

1. `parameter_registry.py` — `ParameterInfo.prior_mean` and
   `ParameterInfo.prior_std` are consumed by
   `cmc/priors.py:build_default_priors()` and
   `build_log_space_priors()`.
2. `parameter_space.py` — `_DEFAULT_PRIOR_SPECS` is consumed by
   `parameter_space.py:_default_prior()` to initialise `ParameterSpace`
   for the NLSQ/optimization side.

When updating any prior, both sources MUST be edited together. Drift
between them produces silent inconsistency between NLSQ warm-start
priors and CMC NUTS priors.

### 14 Physics Parameters + 2 Scaling

**Reference transport** — `J_r(t) = D0_ref * t^alpha_ref + D_offset_ref`

| Parameter | Description | Default | Units |
|---|---|---|---|
| D0_ref | Reference diffusion prefactor | 1e4 | Å²/s^α |
| alpha_ref | Reference transport exponent | 0.0 | — |
| D_offset_ref | Reference transport rate offset | 0.0 | Å²/s |

**Sample transport** — `J_s(t) = D0_sample * t^alpha_sample + D_offset_sample`

| Parameter | Description | Default | Units |
|---|---|---|---|
| D0_sample | Sample diffusion prefactor | 1e4 | Å²/s^α |
| alpha_sample | Sample transport exponent | 0.0 | — |
| D_offset_sample | Sample transport rate offset | 0.0 | Å²/s |

**Velocity** — `v(t) = v0 * t^beta + v_offset`

| Parameter | Description | Default | Units |
|---|---|---|---|
| v0 | Velocity prefactor | 1e3 | Å/s^β |
| beta | Velocity exponent (0 = constant) | 0.0 | — |
| v_offset | Velocity offset (negative = reversal) | 0.0 | Å/s |

**Sample fraction** — `f_s(t) = clip(f0 * exp(f1 * (t - f2)) + f3, 0, 1)`

| Parameter | Description | Default | Units |
|---|---|---|---|
| f0 | Fraction amplitude | 0.5 | — |
| f1 | Exponential rate (0 = constant) | 0.0 | 1/s |
| f2 | Time shift | 0.0 | s |
| f3 | Baseline offset | 0.0 | — |

**Flow angle**

| Parameter | Description | Default | Units |
|---|---|---|---|
| phi0 | Flow angle offset relative to q-vector | 0.0 | degrees |

**Per-angle scaling** (not part of the 14-element physics array)

| Parameter | Description | Default | Units |
|---|---|---|---|
| contrast | Optical contrast (per-angle) | 0.5 | — |
| offset | Baseline offset (per-angle) | 1.0 | — |

> See `physical-model-architecture.md §Parameters` for the authoritative contrast bounds.

---

## Component Map

```
data/
├── xpcs_loader.py        # XPCSDataLoader, XPCSData, load_xpcs_data(),
│                         #   probe_hdf5_structure(), load_xpcs_batch(),
│                         #   select_optimal_wavevector()
├── angle_filtering.py    # filter_by_angle_range(), select_single_angle()
├── phi_filtering.py      # PhiAngleFilter, PhiFilterResult
├── preprocessing.py      # PreprocessingPipeline, PreprocessingResult,
│                         #   PreprocessingProvenance, NoiseReductionMethod
├── memory_manager.py     # MemoryManager, MemoryBudget
├── types.py              # AngleRange dataclass
├── config.py             # DataConfig dataclass
├── validation.py         # Data validation rules and checks
├── validators.py         # Shape, dtype, finiteness checks
├── quality_controller.py # QualityController + 4-stage QC pipeline
├── filtering_utils.py    # NaN masking helpers
├── optimization.py       # DatasetSizeCategory, create_loading_plan(),
│                         #   estimate_optimal_time_range(),
│                         #   compute_dataset_statistics(), recommend_strategy()
└── performance_engine.py # PerformanceEngine LRU cache

config/
├── manager.py            # ConfigManager
├── parameter_registry.py # DEFAULT_REGISTRY (16 entries)
├── parameter_names.py    # ALL_PARAM_NAMES_WITH_SCALING ordering
├── parameter_manager.py  # ParameterManager: vary flags, bounds
├── parameter_space.py    # ParameterSpace + _DEFAULT_PRIOR_SPECS
├── physics_validators.py # Declarative parameter constraints
└── types.py              # PARAMETER_NAME_MAPPING (legacy → canonical)

io/
├── json_utils.py         # json_safe(), json_serializer(), save_json()
├── nlsq_writers.py       # save_nlsq_json_files(), save_nlsq_npz_file()
└── mcmc_writers.py       # save_mcmc_results(), save_mcmc_diagnostics()

cli/
├── commands.py           # dispatch_command() — main orchestrator
├── optimization_runner.py# run_nlsq() / run_cmc() per-method runners
└── main.py               # main(), main_hexp(), main_hsim() entry points
```

---

## Data Loading

### XPCSData Container

All loaded data is returned as an `XPCSData` dataclass:

| Field | Type | Description |
|---|---|---|
| `c2` | `np.ndarray` | Two-time correlation matrix; `(N, N)` or `(n_phi, N, N)` |
| `t1`, `t2` | `np.ndarray` | Time arrays (identical on load; `t1 == t2 == t`) |
| `q` | `float \| None` | Single scattering wavevector in Å⁻¹ |
| `phi_angles` | `np.ndarray \| None` | Per-bin phi angles in degrees |
| `uncertainties` | `np.ndarray \| None` | Per-element uncertainty (if available) |
| `q_values` | `np.ndarray \| None` | Per-bin q values for multi-q data |
| `metadata` | `dict[str, Any]` | File-level attributes |

Properties: `shape`, `n_times`, `has_multi_phi` (true when `c2.ndim == 3`),
`has_multi_q` (true when `q_values is not None`).

### Supported Formats

`XPCSDataLoader` detects the format from the file extension or accepts an
explicit `format` keyword:

| Extension | Format | Backend |
|---|---|---|
| `.h5`, `.hdf5`, `.hdf` | `"hdf5"` | `h5py` |
| `.npz` | `"npz"` | `numpy.load` (object deserialization disabled) |
| `.npy` | `"npy"` | `numpy.load` (object deserialization disabled) |
| `.mat` | `"mat"` | `scipy.io.loadmat` |

**Security.** NPZ and NPY files are loaded with object-deserialization
disabled to prevent execution of arbitrary objects from untrusted files.
A `ValueError` is raised before any data is read if the file requires
object deserialization.

**Key inference.** When the time key is absent, the loader falls back to
integer indices `arange(n_t)` with a `WARNING` log. Missing `q` or `phi`
keys are silently set to `None`.

### HDF5 Format Detection

`XPCSDataLoader._detect_hdf5_format()` inspects the open file object and
selects one of four layout conventions:

| Format | Detection | Loader |
|---|---|---|
| `"aps_u"` | `xpcs/twotime/correlation_map` AND `xpcs/qmap/dynamic_v_list_dim0` | `_load_hdf5_aps_u` |
| `"aps_old"` | `xpcs/dqlist` AND `exchange/C2T_all` | `_load_hdf5_aps_old` |
| `"exchange"` | `/exchange/` group present | `_load_hdf5_exchange` |
| `"flat"` | Default; datasets at root | `_load_hdf5` (flat path) |

**APS-U layout.** `xpcs/qmap/dynamic_v_list_dim0` contains q-values,
`dim1` contains phi-values. `xpcs/twotime/processed_bins` (1-based) maps
correlation-matrix keys to `(q_idx, phi_idx)` via
`bin_idx = bin - 1; q_idx = bin_idx // n_phi; phi_idx = bin_idx % n_phi`.
Each correlation entry is stored as a triangular half-matrix and is
reconstructed via `_reconstruct_from_half_matrix()` (`M + M.T - diag(M)`).

**APS old layout.** `xpcs/dqlist` and `xpcs/dphilist` are squeezed to 1-D.
Half-matrices in `exchange/C2T_all` are sorted by key and stacked.

**Exchange layout.** Looks up `twotime_corr` / `twotime` / `c2` / `corr`
for the correlation, `tau` / `t` / `times` / `delay_time` for time, and
`q_val` / `q_values` / `q` / `qval` for q.

**Flat layout.** Uses the user-specified `c2_key` and `time_key` at the
root. Issues a `WARNING` log when the time key is missing and synthesises
indices.

The half-triangle reconstruction inspects the L1 norm of the strict
upper and lower triangles to log which side is populated; the formula
`M + M.T - diag(M)` works for both.

### Multi-Q Support

When the underlying file contains multiple q-bins, `c2` is loaded as
`(n_q, n_t, n_t)` with `q_values` set to the per-bin wavevectors. The
loader exposes two parameters that drive q-bin selection:

| Argument | Purpose |
|---|---|
| `select_q: float` | Target wavevector in Å⁻¹ |
| `q_tolerance: float \| None` | Maximum absolute deviation; `None` keeps only the single closest bin |

The selection is delegated to `select_optimal_wavevector(q_values,
target_q, tolerance)`, which returns `(indices, selected_q_values)`.
`_apply_q_selection()` then either reduces the array to a single 2-D
slice (single-bin path: clears `q_values`, sets scalar `q`) or keeps it
3-D with updated `q_values` (multi-bin path).

When q-selection is paired with `use_cache=True`, the q-filter is applied
**before** the cache is written so q-specific cache files store the
selected subset (homodyne parity).

### Convenience Function

```python
from heterodyne.data.xpcs_loader import load_xpcs_data

data = load_xpcs_data("run042.h5", c2_key="c2", time_key="t")
# data.c2: (N, N) or (n_phi, N, N)
# data.q: float or None
```

### Batch Loading

`load_xpcs_batch(file_paths, ...)` loads multiple files independently;
failed files are logged and skipped. The same `frame_range`, `select_q`,
`q_tolerance`, and `apply_diag_correction` knobs apply uniformly to
every file.

### Structure Probe

`probe_hdf5_structure(file_path)` walks an HDF5 file and returns a dict
with `datasets`, `groups`, `root_attrs`, `n_datasets`, `n_groups`. Useful
for discovering the right `c2_key` before configuring a run.

---

## Phi Angle Filtering

Multi-phi datasets contain a 3-D array of shape `(n_phi, N, N)`. Two
modules provide complementary phi-selection paths.

### `angle_filtering.py` — range-based selection

```python
from heterodyne.data.angle_filtering import filter_by_angle_range
from heterodyne.data.types import AngleRange

c2_filtered, phi_selected = filter_by_angle_range(
    data.c2,           # shape (n_phi, N, N)
    data.phi_angles,   # shape (n_phi,)
    AngleRange(phi_min=80.0, phi_max=100.0),
)
# c2_filtered: shape (n_selected, N, N)
# phi_selected: shape (n_selected,)
```

Validation errors raised when:
- `c2_3d` is not 3-D.
- `phi_angles` length does not match the first axis of `c2_3d`.
- `phi_min > phi_max`.
- No angles fall within the requested range.

### `phi_filtering.py` — class-based with averaging

`PhiAngleFilter` is the higher-level interface, returning a
`PhiFilterResult`:

| Method | Description |
|---|---|
| `select_range(phi_min, phi_max)` | Select angles within a range |
| `average_symmetric(target_phi)` | Average c2 at phi and 180°–phi |
| `filter_by_phi(data, phi_ranges)` | Convenience function; multiple ranges |

`PhiFilterResult` carries the filtered c2 array, selected phi values,
and `selected_indices` (integer positions into the original phi axis).

---

## Preprocessing Pipeline

`PreprocessingPipeline` is a composable, chainable sequence of array
transformations applied to the c2 array (2-D or 3-D). Each step is a named
callable stored in an ordered list.

### Core Pipeline Steps

| Method | Description |
|---|---|
| `normalize_diagonal()` | Normalize so diagonal values equal 1 |
| `subtract_baseline(baseline)` | Subtract a scalar baseline |
| `clip_values(min_val, max_val)` | Clip array values to a range |
| `remove_outliers(n_sigma, replace_with)` | Replace elements beyond n_sigma from off-diagonal mean |
| `symmetrize()` | Enforce c2(t1, t2) = c2(t2, t1) via `nanmean` |
| `crop_time(t_start, t_end)` | Restrict time axis to `[t_start, t_end)` |
| `add_step(name, func)` | Add a custom transformation |

### Outlier Removal Detail

For square matrices, statistics (mean, std) are computed from off-diagonal
elements only. This prevents the typically larger diagonal values from
biasing the outlier threshold. Three replacement strategies are
supported: `"median"`, `"nan"`, and `"clip"`. For 3-D inputs each
phi-slice is processed independently.

### Diagonal Handling

Diagonal artefacts in `c2(t1, t2)` are handled separately from
`normalize_diagonal()`. Whereas `normalize_diagonal()` rescales the c2
matrix so the diagonal equals 1, the diagonal-correction utilities in
`heterodyne.core.diagonal_correction` *replace* the diagonal band with
estimates computed from off-diagonal neighbours.

| Function | Input | Behaviour |
|---|---|---|
| `apply_diagonal_correction(c2, width, method, backend)` | `(N, N)` | Single matrix; auto-detects JAX vs NumPy backend |
| `apply_diagonal_correction_batch(c2_batch, width, method, backend)` | `(N, N)` or `(n_batch, N, N)` | Vectorised; uses cached `jit(vmap)` for JAX path |

Supported `method` values:

| Method | Strategy |
|---|---|
| `"basic"` | Side-band averaging: average first upper/lower off-diagonals, then average adjacent side-band values onto the diagonal (homodyne parity) |
| `"interpolate"` | Heterodyne alias for `"basic"` |
| `"interpolation"` | Linear interpolation from neighbouring off-diagonal values (NumPy backend only) |
| `"mask"` | Set diagonal-band elements to NaN |
| `"mirror"` | Replace via `c2[i,j] = c2[j,i]` symmetry |
| `"statistical"` | Per-row statistic from a window at distances `[width, width+3)` (NumPy backend only) |

For 3-D inputs the correction is applied independently to each q-slice.
The `width` argument controls the half-width of the corrected band
(`width=1` corrects only the main diagonal). `XPCSDataLoader` exposes a
NumPy mirror of this routine via the `apply_diag_correction` keyword on
`load_xpcs_batch()`.

`estimate_diagonal_excess(c2, width)` returns summary statistics
(`mean_diagonal`, `mean_off_diagonal`, `mean_excess`, `std_ratio`,
`excess_sigma`) used by the quality controller's diagonal check.
`compute_weights_excluding_diagonal(shape, width)` builds a 0/1 weight
mask that excludes the diagonal band — used by NLSQ when fitting only
off-diagonal residuals.

### Noise Reduction (`apply_noise_reduction()`)

Four noise filters are available via `NoiseReductionMethod`:

| Method | Description |
|---|---|
| `WIENER` | Adaptive Wiener filter (default) |
| `SAVGOL` | Savitzky-Golay polynomial smoothing |
| `MEDIAN` | Median filter with configurable kernel size |
| `GAUSSIAN` | Gaussian convolution smoothing |

### Normalization Utilities

Standalone normalization functions (not part of the pipeline builder):

| Function | Description |
|---|---|
| `normalize_zscore(c2)` | Zero-mean, unit-variance normalization |
| `normalize_minmax(c2)` | Min-max rescaling to [0, 1] |
| `normalize_robust(c2)` | IQR-based robust normalization |
| `apply_baseline_correction(c2)` | Polynomial baseline estimation and subtraction |

### PreprocessingProvenance

`PreprocessingProvenance` is an audit-trail dataclass that records every
transformation applied to a c2 array, including array hashes (SHA-256
prefix), step timings, and a pipeline UUID
(`_generate_pipeline_id()`). It is attached to `PreprocessingResult` when
`track_provenance=True` is passed to `pipeline.run()`.
`TransformationRecord` captures per-step metadata: step name,
parameters, input/output shapes, and wall time.

### Usage Example

```python
from heterodyne.data.preprocessing import PreprocessingPipeline

pipeline = (
    PreprocessingPipeline()
    .symmetrize()
    .remove_outliers(n_sigma=5.0, replace_with="median")
    .normalize_diagonal()
)
result = pipeline.run(data.c2)
# result.c2: preprocessed array
# result.applied_steps: ["symmetrize", "remove_outliers(5.0s)", "normalize_diagonal"]
# result.statistics: logged metrics (outlier count, etc.)
# result.provenance: PreprocessingProvenance (if track_provenance=True)
```

---

## Quality Control

`heterodyne.data.quality_controller` provides single-shot quality
assessment and a structured 4-stage pipeline.

### QualityController

`QualityController(config: DataConfig | None = None)` runs a battery of
checks on a loaded c2 array and produces a `QualityReport`. Individual
checks:

| Check | Threshold | Critical condition |
|---|---|---|
| `_check_nan_fraction` | 5 % | NaN fraction > 5 % |
| `_check_snr` | SNR ≥ 5 | SNR < 2 |
| `_check_symmetry` | rel asymmetry ≤ 1 % | rel asymmetry > 5 % |
| `_check_diagonal_excess` | ratio ≤ 2 | ratio > 5 |
| `_check_value_range` | abs max ≤ 100 | (warning above 100) |
| `_check_time_coverage` | ≥ 10 points | < 5 points or non-monotonic |

Each check returns a `QualityMetric(name, value, threshold, level,
message)`; `QualityLevel` is one of `GOOD`, `ACCEPTABLE`, `WARNING`,
`CRITICAL`. The aggregate `QualityReport` carries the worst-case
`overall_level` and a `recommendations` list.

### 4-Stage Pipeline

`run_4_stage_pipeline(controller, c2, t, config)` runs the quality
assessment at four progressive stages:

| Stage | Enum value | Triggers |
|---|---|---|
| RAW | `QualityControlStage.RAW` | Run on freshly loaded data; warns when score < 0.5 |
| FILTERED | `QualityControlStage.FILTERED` | After phi-angle filtering; warns when score < 0.7 |
| PREPROCESSED | `QualityControlStage.PREPROCESSED` | After preprocessing pipeline; warns when score < 0.8 |
| FINAL | `QualityControlStage.FINAL` | Before fitting; warns when score < 0.9 |

Between stages, `apply_auto_corrections()` may apply optional repairs
when enabled in `QualityControlConfig`:

| Flag | Action |
|---|---|
| `auto_repair_nans` | Replace NaNs with finite-mean fill |
| `auto_repair_outliers` | Clip values beyond `outlier_sigma` × σ |

`_compute_adaptive_thresholds(c2, config)` returns a relaxed
`QualityControlConfig` for very noisy data (loosens SNR threshold by 2×
when `|std/mean| > 1`, loosens NaN threshold by 1.5× when initial NaN
fraction > 1 %).

### Result Types

| Class | Purpose |
|---|---|
| `QualityMetric` | Single metric with name, value, threshold, level, message |
| `QualityReport` | Aggregated metrics, worst-case overall level, recommendations |
| `QualityControlConfig` | Thresholds and auto-repair toggles for the pipeline |
| `QualityControlResult` | Per-stage report, auto-corrections, score in [0, 1] |
| `QualityControlStage` | Enum: RAW / FILTERED / PREPROCESSED / FINAL |
| `QualityLevel` | Enum: GOOD / ACCEPTABLE / WARNING / CRITICAL |

Helpers: `assess_stage()`, `suggest_fixes(report)` (returns prioritised
action dicts), `apply_auto_corrections()`, `track_quality_history()`,
`export_report(result, format="text" | "json")`.

### QC Downstream Policy

| QC Level | NLSQ Entry | CMC Entry | CLI Exit Code | Log Level |
|---|---|---|---|---|
| `GOOD` (χ² < 1.5, no bounds) | Proceeds | Proceeds with MAP warm-start | 0 | INFO |
| `ACCEPTABLE` (χ² 1.5–3.0, or bounds hit) | Proceeds with warning | Proceeds with wider priors | 0 | WARNING |
| `WARNING` (χ² ≥ 3.0 or None) | Skipped (no CMC warm-start) | Proceeds with default priors | 0 | WARNING |
| `CRITICAL` (exception/timeout) | N/A | Skipped entirely | 1 | ERROR |

QC level does not block pipeline progression by default — it gates warm-start quality, not execution. Set `CMCConfig(require_good_nlsq=True)` to abort CMC on poor NLSQ quality.

### Relationship to validation.py and validators.py

- `data/validation.py` and `data/validators.py` enforce **hard**
  constraints at I/O boundaries (shape, dtype, finiteness,
  monotonicity). They raise `DataValidationError` on violation.
- `xpcs_loader.py:validate_loaded_data(data)` is the loader-level
  hard-validation entry point: enforces NaN/Inf finiteness, shape
  consistency, positive diagonal, and time monotonicity. Soft
  symmetry violations are returned as warning strings.
- `quality_controller.py` produces **soft** assessments: scores,
  levels, and recommendations. It never raises — it surfaces issues
  for downstream decisions.

---

## Dataset Optimization

`data/optimization.py` classifies datasets and recommends loading
strategies before fitting begins.

### DatasetSizeCategory

`categorize_dataset(shape, dtype, available_memory)` returns a
`DatasetSizeCategory` with the following thresholds (using `_MB =
100_000_000` and `_GB = 1_000_000_000` byte heuristics from the source):

| Category | Size range | `chunk_size` | `use_memory_mapping` | `use_progressive_loading` | `use_compression` |
|---|---|---|---|---|---|
| small | `< 100 MB` | `n` | False | False | False |
| medium | `100 MB – 1 GB` | `n // 4` | False | False | False |
| large | `1 GB – 10 GB` | `n // 16` | True | False | True |
| very_large | `> 10 GB` | `n // 64` | True | True | True |

Where `n` is the first axis length. `available_memory` is auto-detected
via `psutil.virtual_memory().available`, falling back to 8 GB.

### create_loading_plan

`create_loading_plan(file_path, dataset_shape, dtype)` combines
categorisation with throughput estimates:

| Category | Throughput estimate | Strategy |
|---|---|---|
| small | 500 MB/s | `full_load` |
| medium | 300 MB/s | `chunked` |
| large | 150 MB/s | `mmap_chunked` |
| very_large | 80 MB/s | `progressive_mmap` |

The returned dict carries `category`, `strategy`, `chunk_size`,
`use_cache`, `use_mmap`, and `estimated_load_time_seconds`. Caching is
enabled (`use_cache=True`) for `medium`, `large`, and `very_large`
datasets.

### Other Utilities

| Function | Purpose |
|---|---|
| `compute_dataset_statistics(c2, t)` | mean, std, SNR, n_nan, dynamic_range, effective_rank (SVD-based) |
| `recommend_strategy(c2, t)` | `use_upper_triangle`, `exclude_diagonal`, weight method, chunk size |
| `estimate_optimal_time_range(c2, t, snr_threshold)` | SNR-thresholded time window from diagonal lag profile |
| `process_chunks_parallel(c2, process_fn, chunk_size, max_workers)` | Thread-pool chunked processing of 3-D batches |
| `subsample_correlation(c2, config)` | Explicit opt-in subsampling — emits WARNING on every call |

Subsampling is **off by default** and every invocation logs a warning,
per project rules. Subsampling methods: `"uniform"` (every Nth index),
`"random"` (uniform random without replacement), `"adaptive"` (60 %
near diagonal, 40 % far).

---

## Caching & Performance

### NPZ Disk Cache

`XPCSDataLoader.load(use_cache=True)` enables NPZ caching for HDF5 and
MAT formats. The cache lives next to the source file (or in
`cache_file_path` when configured) and is invalidated automatically
when the source file's mtime changes.

| Field | Source |
|---|---|
| Cache directory | `cache_dir` argument, falls back to source's parent |
| Cache filename | `cache_filename_template` with `${var}` substitution, fallback `<source>.heterodyne_cache.npz` |
| Compression | `cache_compression` (default `True`) |

The cache stores `source_mtime`, `c2`, `t`, optional `q`, `q_values`,
and `phi`. `_cache_is_valid()` compares the stored mtime with the
current source mtime — no content hashing is performed. Cache writes
are non-fatal: filesystem failures are logged as warnings and the
loader continues from source.

The legacy `{var}` format in `cache_filename_template` is auto-migrated
to `${var}` on first use; a deprecation warning is logged once. Generated
filenames are validated against path traversal (rejecting `os.sep` or
`..`).

When `frame_range` or `select_q` is specified with caching, the slice/
selection is applied **before** the cache is written so the on-disk
cache stores the resolved subset.

### In-Memory LRU Cache (`PerformanceEngine`)

`PerformanceEngine` provides an LRU-eviction cache for expensive XPCS
arrays in process memory:

```python
from heterodyne.data.performance_engine import PerformanceEngine

engine = PerformanceEngine(max_cache_bytes=2 * 1024**3)  # 2 GB
engine.put("run042_c2", c2_array)
cached = engine.get("run042_c2")   # None on miss
engine.evict("run042_c2")
engine.clear()
stats = engine.stats()  # total_size, hit_count, miss_count, eviction_count
```

`CacheEntry` stores the array, access time, and byte size. Cache
operations are thread-safe. The engine is distinct from `MemoryManager`
— it tracks actual cached bytes, not user-registered budgets.

### Memory-Mapped Access

When `DatasetSizeCategory` returns `use_memory_mapping=True` (large /
very_large categories), downstream loaders may use memory-mapped HDF5
access to avoid materialising the full array in RAM. The loading plan
exposes this hint via the `use_mmap` field.

---

## Memory Manager

`MemoryManager` tracks allocations against a configurable byte budget,
enabling downstream code to choose chunk sizes that fit in memory
without trial and error.

```python
from heterodyne.data.memory_manager import MemoryManager

mm = MemoryManager(budget_bytes=None)   # auto-detect via psutil; fallback 8 GB
mm.allocate("c2_matrix", n_times**2 * 8)  # 8 bytes/float64
budget = mm.get_budget()
# budget.total_bytes, budget.allocated_bytes, budget.peak_bytes
mm.release("c2_matrix")
```

Key properties:
- When `budget_bytes=None`, auto-detects available system memory via
  `psutil.virtual_memory()`. Falls back to 8 GB if psutil is not
  installed.
- All public methods are thread-safe (protected by `threading.Lock`).
- `get_budget()` returns a snapshot `MemoryBudget` dataclass. It tracks
  explicitly registered allocations only, not OS-level memory
  consumption.

For a holistic view of dataset size handling, the `MemoryManager`
budget is consumed by NLSQ strategies (`optimization/nlsq/memory.py`)
in tandem with the categorical guidance from
`DatasetSizeCategory`. Categories with bytes ranges from "small"
(< 100 MB) up to "very_large" (> 10 GB) drive both chunk-size choices
and the on-disk caching policy.

---

## Result Writing

After fitting, results are written to `output_dir` by the writers in
`heterodyne/io/`. All output paths are absolute or relative to
`output_dir`.

### NLSQ Results

`save_nlsq_json_files(result, output_dir, prefix="nlsq")` writes:

| Filename | Contents |
|---|---|
| `{prefix}_parameters.json` | `parameters`, `uncertainties`, `parameter_names`, `timestamp` |
| `{prefix}_metadata.json` | `success`, `message`, `n_iterations`, `n_function_evals`, `final_cost`, `reduced_chi_squared`, `convergence_reason`, `wall_time_seconds`, `metadata` |

`save_nlsq_npz_file(result, output_path, include_residuals, include_jacobian)` writes a single compressed `.npz` containing:

| Key | Description |
|---|---|
| `parameters` | Fitted parameter array |
| `parameter_names` | Per-element name strings (`U64`) |
| `success`, `message` | Convergence status |
| `final_cost`, `reduced_chi_squared` | Fit quality (NaN sentinels) |
| `n_iterations`, `n_function_evals`, `convergence_reason` | Convergence telemetry |
| `wall_time_seconds` | Wall-clock duration |
| `metadata_json` | JSON-encoded metadata dict |
| `uncertainties`, `covariance` | Parameter uncertainty (when present) |
| `residuals` | Optional, gated by `include_residuals=True` |
| `jacobian` | Optional, gated by `include_jacobian=False` (large) |
| `fitted_correlation` | Reconstructed c2 from final parameters |

`load_nlsq_npz_file(path)` round-trips the NPZ back into an
`NLSQResult`. `format_nlsq_summary(result)` returns a human-readable
text summary.

### CMC Results

`save_mcmc_results(result, output_dir, prefix="mcmc")` writes (using a
staging tmpdir + atomic `os.replace` / `shutil.move`):

| Filename | Contents |
|---|---|
| `{prefix}_summary.json` | `parameter_names`, `posterior_mean`, `posterior_std`, `credible_intervals`, `map_estimate`, `timestamp`, `num_samples`, `num_chains` |
| `{prefix}_diagnostics.json` | Per-parameter `r_hat`, `ess_bulk`, `ess_tail`; aggregate `max_r_hat`, `min_ess_bulk`, `bfmi`, plus `sampling_info` |
| `{prefix}_samples.npz` | `parameter_names`, `samples_<name>` per parameter, `r_hat`, `ess_bulk`, `ess_tail` |

`save_mcmc_diagnostics()` checks individual R-hat values against
`r_hat_threshold` (default 1.1) and BFMI against `min_bfmi` (default
0.3). `format_mcmc_summary(result)` returns a tabular text summary
with mean ± std and 95 % credible intervals.

### JSON Serialization

`heterodyne/io/json_utils.py` provides JAX-aware JSON helpers:

| Function | Purpose |
|---|---|
| `json_safe(obj)` | Recursively convert JAX arrays, NumPy arrays, complex numbers, `Path`, `datetime`, etc. to JSON-serialisable forms |
| `json_serializer(obj)` | Pretty-print JSON string; `allow_nan=False` |
| `save_json(data, path)` | Atomic write (write-to-temp + `os.replace`) to prevent partial writes |
| `load_json(path)` | Plain JSON load returning `dict[str, Any]` |

NaN/Inf floats are rejected at serialisation time
(`_sanitize_float()`) — finite-number-only output is enforced. Complex
arrays are encoded as `{"__complex_array__": True, "shape": [...],
"data": [{"real": ..., "imag": ...}, ...]}` for shape-preserving
round-trip.

---

## CLI Orchestration

### CLI Entry Points

Defined in `pyproject.toml` `[project.scripts]`:

| Command | Short alias | Module entry | Purpose |
|---|---|---|---|
| `heterodyne` | `ht` | `heterodyne.cli.main:main` | Main analysis (NLSQ/CMC) |
| `heterodyne-config` | `ht-config` | `heterodyne.cli.config_generator:main` | Config generation/validation |
| `heterodyne-config-xla` | `ht-config-xla` | `heterodyne.cli.xla_config:main` | XLA device configuration |
| `heterodyne-post-install` | `ht-post-install` | `heterodyne.post_install:main` | Shell completion setup |
| `heterodyne-cleanup` | `ht-cleanup` | `heterodyne.uninstall_scripts:main` | Remove shell completion |
| `heterodyne-validate` | `ht-validate` | `heterodyne.runtime.utils.system_validator:main` | System validation |
| `hexp` | — | `heterodyne.cli.main:main_hexp` | Plot experimental data (skip optimisation) |
| `hsim` | — | `heterodyne.cli.main:main_hsim` | Plot simulated C2 heatmaps from config |

### Dispatch Flow

`cli/commands.py:dispatch_command(args)` is the top-level orchestrator.
Phase order:

1. **config_loading** — `load_and_merge_config(args.config, args)` →
   `ConfigManager`.
2. **data_loading** — `XPCSDataLoader(...).load()` → `XPCSData`.
3. **plotting** (optional, `--plot-only` / `--simulate-only`) — skip
   optimisation entirely.
4. **nlsq_optimization** — only when `method in ("nlsq", "both")`.
5. **cmc_optimization** — only when `method in ("cmc", "both")`. In
   `cmc`-only mode the runner calls `resolve_nlsq_warmstart()` to load
   a previously saved NLSQ result from disk.
6. **cmc_diagnostics** — when CMC results are present.
7. **plotting** + **save_plots** — final figures.

### --method Behaviour

The `--method` argument (resolved via `getattr(args, "method", "nlsq")`)
selects the dispatch path:

| Value | Behaviour |
|---|---|
| `nlsq` | Run NLSQ only. No CMC stage. |
| `cmc` | Run CMC only. The orchestrator does **not** auto-run NLSQ; instead it loads a previously saved NLSQ warm-start from disk via `resolve_nlsq_warmstart()`. If no warm-start is found, a `WARNING` is logged ("Run NLSQ first ... or use optimizer: both") and CMC starts from the prior. |
| `both` | Run NLSQ first, then CMC with the in-memory NLSQ result list passed as `nlsq_results` (no disk round-trip). |

### t=0 Exclusion for NLSQ

`optimization_runner.py:_exclude_first_time_point_for_nlsq(model,
c2_data)` drops the leading time point before NLSQ runs and re-syncs
the model's time axis via `model.sync_time_axis(np.arange(...))`. The
diagonal at `t=0` is dominated by photon shot-noise; excluding it
sharpens the early-time dynamics fit. A guard prevents trimming when
either time axis has length ≤ 1 (single-frame data) and when `c2_data`
is not 2-D or 3-D.

### Phase-Level Logging

Both `commands.py` and `optimization_runner.py` use:

- `log_phase(name, logger, track_memory=True)` — context manager that
  logs phase start/end with wall time and (optionally) peak memory in
  GB. The yielded object's `memory_peak_gb` attribute is forwarded to
  the analysis summary.
- `AnalysisSummaryLogger(run_id, analysis_mode)` — accumulates
  per-phase timing and memory peaks; `start_phase()` / `end_phase()`
  bracket each major operation and `set_config_summary()` records the
  optimiser choice.

Per-angle CMC loops also wrap each `cmc_phi_<i>` iteration in a
`log_phase` so phi-by-phi timing is captured separately.

---

## Data Flow Summary

```
YAML config on disk
      │
      ▼ ConfigManager.from_yaml(path)
ConfigManager
      │
      ▼ XPCSDataLoader(...).load(...)
XPCSData (c2, t, q, phi_angles, q_values, metadata)
      │
      ▼ filter_by_angle_range() / PhiAngleFilter   [optional, multi-phi only]
XPCSData slice (n_selected, N, N)
      │
      ▼ PreprocessingPipeline.run()
PreprocessingResult (c2_clean, applied_steps, statistics, provenance?)
      │
      ▼ apply_diagonal_correction()                [optional]
c2 with corrected diagonal band
      │
      ▼ run_4_stage_pipeline()                     [optional]
QualityControlResult ×4 (RAW → FILTERED → PREPROCESSED → FINAL)
      │
      ▼ _exclude_first_time_point_for_nlsq()       [NLSQ path only]
c2 trimmed at t=0
      │
      ▼ fit_nlsq_jax() / fit_cmc_jax()
NLSQResult / CMCResult
      │
      ▼ save_nlsq_json_files() / save_nlsq_npz_file() / save_mcmc_results()
output_dir/{nlsq_parameters.json, nlsq_metadata.json,
            mcmc_summary.json, mcmc_diagnostics.json, mcmc_samples.npz}
```

---

## Quick Reference Tables

### Data Shapes at Each Stage

| Stage | c2 shape | Notes |
|---|---|---|
| Raw HDF5 (flat) | `(N, N)` or `(n_q, N, N)` | depends on file |
| Raw HDF5 (APS-U / APS old) | `(n_bins, N, N)` | half-matrices reconstructed |
| After `_apply_q_selection` (1 bin) | `(N, N)` | `q_values` cleared, `q` set |
| After `_apply_q_selection` (k bins) | `(k, N, N)` | `q_values` updated |
| After `filter_by_angle_range` | `(n_selected, N, N)` | multi-phi only |
| After `_apply_frame_slicing` | `(N', N')` or `(n_phi, N', N')` | 1-based inclusive |
| After preprocessing pipeline | unchanged | values updated, shape preserved |
| After `_exclude_first_time_point_for_nlsq` | `(N-1, N-1)` or `(n_phi, N-1, N-1)` | leading time point dropped |

### Error Types

| Exception | Module | Meaning |
|---|---|---|
| `ConfigurationError` | `config/manager.py` | Missing required section, invalid `optimization.method`, missing `value` in parameter dict |
| `DataValidationError` | `data/xpcs_loader.py` | NaN/Inf, shape mismatch, non-positive diagonal, non-monotonic time |
| `ValueError` (loader) | `data/xpcs_loader.py` | Unknown file format; object-deserialization required; empty correlation group; `c2.ndim ∉ {2, 3}`; `select_q` matched no bin |
| `KeyError` (loader) | `data/xpcs_loader.py` | Missing `c2_key` (or no recognised c2 key in `/exchange/`) |
| `PathValidationError` | `utils/path_validation` | File does not exist or is not readable |
| `OptimizationError` (base) | `optimization/exceptions.py` | Generic optimisation failure |
| `ConvergenceError` / `NumericalError` | `optimization/exceptions.py` | NLSQ/CMC convergence or NaN/Inf |
| `BoundsError` | `optimization/exceptions.py` | Parameter hit optimisation bound |
| `DegeneracyError` | `optimization/exceptions.py` | Correlated/degenerate parameters |
| `ValidationError` | `optimization/exceptions.py` | Result validation failure |
| `StreamingError` / `ShardingError` / `BackendError` | `optimization/exceptions.py` | Strategy/backend-specific failures |

### Key Config Defaults

| Field | Default | Source |
|---|---|---|
| `experimental_data.file_format` | `"hdf5"` | `ConfigManager.file_format` |
| `experimental_data.cache_compression` | `True` | `ConfigManager.cache_compression` |
| `analyzer_parameters.dt` | `1.0` | `_normalize_analyzer_parameters` fallback |
| `analyzer_parameters.start_frame` | `1` | same |
| `analyzer_parameters.end_frame` | `1000` | same |
| `analyzer_parameters.scattering.wavevector_q` | `0.01` | same |
| `optimization.method` | `"nlsq"` | `ConfigManager.optimization_method` |
| `optimization.cmc.num_warmup` | `500` | `_merge_cmc_config` |
| `optimization.cmc.num_samples` | `1000` | same |
| `optimization.cmc.num_chains` | `4` | same |
| `optimization.cmc.target_accept_prob` | `0.8` | same |
| `optimization.cmc.max_tree_depth` | `10` | same |
| `output.output_dir` | `./output` | `ConfigManager.output_dir` |

---

## Key Files Reference

| File | One-line purpose |
|---|---|
| `config/manager.py` | `ConfigManager` — YAML/JSON loading, schema migration, validated public properties |
| `config/parameter_registry.py` | Immutable 16-entry `DEFAULT_REGISTRY` with bounds, defaults, priors, log-space flags |
| `config/parameter_names.py` | `ALL_PARAM_NAMES_WITH_SCALING` canonical ordering |
| `config/parameter_manager.py` | Mutable `ParameterManager` with vary flags, bounds, override hooks |
| `config/parameter_space.py` | `ParameterSpace` for priors + `_DEFAULT_PRIOR_SPECS` (must stay in sync with registry) |
| `config/physics_validators.py` | Declarative parameter constraint rules |
| `config/types.py` | `PARAMETER_NAME_MAPPING` (legacy → canonical) and TypedDict structures |
| `data/xpcs_loader.py` | `XPCSDataLoader`, `XPCSData`, `load_xpcs_data`, `load_xpcs_batch`, `select_optimal_wavevector`, `probe_hdf5_structure` |
| `data/angle_filtering.py` | `filter_by_angle_range`, `select_single_angle` |
| `data/phi_filtering.py` | `PhiAngleFilter`, `PhiFilterResult` (range + symmetric averaging) |
| `data/preprocessing.py` | `PreprocessingPipeline`, `PreprocessingResult`, `PreprocessingProvenance`, `NoiseReductionMethod` |
| `data/memory_manager.py` | `MemoryManager`, `MemoryBudget` (thread-safe budget tracker) |
| `data/types.py` | `AngleRange` dataclass |
| `data/config.py` | `DataConfig` dataclass |
| `data/validation.py` | Hard-validation rules and check definitions |
| `data/validators.py` | Shape, dtype, and finiteness validators |
| `data/quality_controller.py` | `QualityController`, 4-stage QC pipeline, auto-corrections |
| `data/filtering_utils.py` | NaN-masking helpers |
| `data/optimization.py` | `DatasetSizeCategory`, `categorize_dataset`, `create_loading_plan`, `estimate_optimal_time_range`, `subsample_correlation` |
| `data/performance_engine.py` | `PerformanceEngine` LRU cache for in-memory arrays |
| `core/diagonal_correction.py` | `apply_diagonal_correction` + batch variant; JAX/NumPy backends |
| `io/json_utils.py` | `json_safe`, `json_serializer`, `save_json`, `load_json` (atomic writes) |
| `io/nlsq_writers.py` | `save_nlsq_json_files`, `save_nlsq_npz_file`, `load_nlsq_npz_file`, `format_nlsq_summary` |
| `io/mcmc_writers.py` | `save_mcmc_results`, `save_mcmc_diagnostics`, `format_mcmc_summary` |
| `cli/main.py` | Argparse setup; `main`, `main_hexp`, `main_hsim` entry points |
| `cli/commands.py` | `dispatch_command` orchestrator + per-phase `log_phase` instrumentation |
| `cli/optimization_runner.py` | `run_nlsq`, `run_cmc`, `_exclude_first_time_point_for_nlsq`, result aggregation |

---

> **Note.** CMC fitting pipeline parity vs homodyne is tracked as a
> separate review item. See `docs/architecture/cmc-fitting-architecture.md`
> for the CMC-specific data flow, NUTS configuration, sharding, and
> warm-start contract.
