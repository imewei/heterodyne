# Data Handler Architecture

## Overview

The `heterodyne.data` package handles everything between raw XPCS files and the
arrays passed to the optimization stages. It covers file loading (HDF5, NPZ,
MAT, NPY), phi-angle selection for multi-angle datasets, a composable
preprocessing pipeline, and a memory manager that tracks allocation budgets for
large datasets.

---

## Component Map

```
data/
├── xpcs_loader.py        # XPCSDataLoader, XPCSData, load_xpcs_data()
├── angle_filtering.py    # filter_by_angle_range(), select_single_angle()
├── phi_filtering.py      # Phi-angle filtering utilities
├── preprocessing.py      # PreprocessingPipeline, PreprocessingResult
├── memory_manager.py     # MemoryManager, MemoryBudget
├── types.py              # AngleRange dataclass
├── config.py             # DataConfig dataclass
├── validation.py         # Data validation rules and checks
├── validators.py         # Shape, dtype, finiteness checks
├── quality_controller.py # Data quality scoring
├── filtering_utils.py    # NaN masking helpers
├── optimization.py       # Chunk-size optimization utilities
└── performance_engine.py # Profiling and throughput instrumentation
```

---

## Data Loading

### XPCSData Container

All loaded data is returned as an `XPCSData` dataclass:

| Field | Type | Description |
|---|---|---|
| `c2` | `np.ndarray` | Two-time correlation matrix; shape `(N, N)` or `(n_phi, N, N)` |
| `t1`, `t2` | `np.ndarray` | Time arrays (identical on load; t1 == t2 == t) |
| `q` | `float \| None` | Scattering wavevector in Å⁻¹ |
| `phi_angles` | `np.ndarray \| None` | Detector phi angles in degrees |
| `uncertainties` | `np.ndarray \| None` | Per-element uncertainty (if available) |
| `metadata` | `dict` | File-level attributes (HDF5 attrs, MATLAB struct fields) |

`has_multi_phi` returns `True` when `c2.ndim == 3`. `n_times` returns the
time-axis size from either the 2D or 3D shape.

### Supported Formats

`XPCSDataLoader` detects the format from the file extension or accepts an
explicit `format` keyword:

| Extension | Format | Backend |
|---|---|---|
| `.h5`, `.hdf5`, `.hdf` | `"hdf5"` | `h5py` |
| `.npz` | `"npz"` | `numpy.load(allow_pickle=False)` |
| `.npy` | `"npy"` | `numpy.load(allow_pickle=False)` |
| `.mat` | `"mat"` | `scipy.io.loadmat` |

**Security.** NPZ and NPY files are loaded with `allow_pickle=False` to
prevent deserialization of arbitrary objects from untrusted files. If the file
requires object deserialization, a `ValueError` is raised before any data is
read.

**Key inference.** When the time key is absent from the file, the loader falls
back to integer indices `arange(n_t)` with a WARNING log. When the `q` or `phi`
key is absent it is silently set to `None`.

### Convenience Function

```python
from heterodyne.data.xpcs_loader import load_xpcs_data

data = load_xpcs_data("run042.h5", c2_key="c2", time_key="t")
# data.c2: (N, N) or (n_phi, N, N)
# data.q: float or None
```

---

## Phi Angle Filtering

Multi-phi datasets contain a 3D array of shape `(n_phi, N, N)`. The
`angle_filtering` module selects a subset of phi slices before fitting.

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
- `c2_3d` is not 3D.
- `phi_angles` length does not match the first axis of `c2_3d`.
- `phi_min > phi_max`.
- No angles fall within the requested range.

---

## Preprocessing Pipeline

`PreprocessingPipeline` is a composable, chainable sequence of array
transformations applied to the c2 array (2D or 3D). Each step is a named
callable stored in an ordered list.

### Available Steps

| Method | Description |
|---|---|
| `normalize_diagonal()` | Normalize so diagonal values equal 1 |
| `subtract_baseline(baseline)` | Subtract a scalar baseline |
| `clip_values(min_val, max_val)` | Clip array values to a range |
| `remove_outliers(n_sigma, replace_with)` | Replace elements beyond n_sigma from off-diagonal mean |
| `symmetrize()` | Enforce c2(t1, t2) = c2(t2, t1) via nanmean |
| `crop_time(t_start, t_end)` | Restrict time axis to [t_start, t_end) |
| `add_step(name, func)` | Add a custom transformation |

### Outlier Removal Detail

For square matrices, statistics (mean, std) are computed from off-diagonal
elements only. This prevents the typically larger diagonal values from biasing
the outlier threshold. Three replacement strategies are supported: `"median"`,
`"nan"`, and `"clip"`. For 3D inputs each phi-slice is processed independently.

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
```

---

## Memory Manager

`MemoryManager` tracks allocations against a configurable byte budget, enabling
downstream code to choose chunk sizes that fit in memory without trial and error.

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
  `psutil.virtual_memory()`. Falls back to 8 GB if psutil is not installed.
- All public methods are thread-safe (protected by `threading.Lock`).
- `get_budget()` returns a snapshot `MemoryBudget` dataclass. It tracks
  explicitly registered allocations only, not OS-level memory consumption.

---

## Data Flow Summary

```
File on disk
      │
      ▼ XPCSDataLoader.load()
XPCSData (c2, t, q, phi_angles, metadata)
      │
      ▼ filter_by_angle_range()        [optional, multi-phi only]
XPCSData slice (n_selected, N, N)
      │
      ▼ PreprocessingPipeline.run()
PreprocessingResult (c2_clean, applied_steps, statistics)
      │
      ▼ InputValidator.validate()
        (passes arrays to NLSQ subsystem)
```
