.. _data-loading:

============
Data Loading
============

The :class:`~heterodyne.data.xpcs_loader.XPCSDataLoader` class provides a
unified interface for loading two-time correlation matrices from various
file formats produced by synchrotron beamline pipelines.


Supported Formats
=================

The loader auto-detects the file format from the internal structure:

**HDF5** (``.h5``, ``.hdf5``)
   The most common output from beamline reduction pipelines.
   Auto-detection distinguishes:

   * **APS-U format** -- Current APS upgrade pipeline layout.
   * **APS legacy format** -- Older 8-ID-I / XPCS analysis pipeline.
   * **Generic HDF5** -- Any HDF5 file containing a ``C2`` or
     ``two_time`` dataset at a known path.

**NumPy** (``.npz``)
   Compressed NumPy archives containing ``c2`` and ``timestamps``
   arrays.  Useful for sharing preprocessed data or synthetic test
   cases.

**MATLAB** (``.mat``)
   Version 5 MAT files with variables ``C2`` and ``t``.


Basic Usage
===========

.. code-block:: python

   from heterodyne.data.xpcs_loader import XPCSDataLoader

   # Load a single-angle dataset
   loader = XPCSDataLoader("run42_q3.h5")
   data = loader.load()

   print(data.c2.shape)        # (N_frames, N_frames)
   print(data.timestamps[:5])  # First 5 frame timestamps in seconds

The returned data object carries the :math:`C_2` matrix, timestamps,
and any metadata present in the source file (q-value, temperature,
exposure time, etc.).


Multi-Angle Data
================

For heterodyne analysis each azimuthal angle :math:`\phi` typically
corresponds to a separate file or dataset group.  Load them
individually and pass the angle values to the fitting functions:

.. code-block:: python

   import numpy as np

   phi_angles = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
   datasets = []

   for phi in phi_angles:
       loader = XPCSDataLoader(f"run42_phi{phi:.1f}.h5")
       datasets.append(loader.load())

   # Stack C2 matrices for multi-angle fitting
   c2_stack = np.stack([d.c2 for d in datasets], axis=0)
   print(c2_stack.shape)  # (8, N_frames, N_frames)


Inspecting Loaded Data
=======================

Before fitting, verify that the data is well-formed:

.. code-block:: python

   data = loader.load()

   # Check for NaN or negative values on the diagonal
   diag = np.diag(data.c2)
   assert not np.any(np.isnan(diag)), "NaN on C2 diagonal"
   assert np.all(diag > 0), "Non-positive diagonal values"

   # Verify timestamps are monotonically increasing
   dt = np.diff(data.timestamps)
   assert np.all(dt > 0), "Non-monotonic timestamps"

   # Print summary
   print(f"Frames:     {data.c2.shape[0]}")
   print(f"Duration:   {data.timestamps[-1] - data.timestamps[0]:.1f} s")
   print(f"Frame rate: {1.0 / np.median(dt):.1f} Hz")


NPZ Caching
============

For large HDF5 files, the loader supports transparent NPZ caching.
On the first load, a ``.npz`` companion file is written next to the
source.  Subsequent loads read the NPZ directly, which is significantly
faster.

Cache validity is checked via **mtime comparison**: if the source file
is newer than the cache, the cache is regenerated automatically.

.. code-block:: python

   loader = XPCSDataLoader("run42_q3.h5", use_cache=True)
   data = loader.load()  # First call: reads HDF5, writes .npz cache
   data = loader.load()  # Second call: reads .npz (faster)


Memory Management
=================

For datasets with thousands of frames the :math:`C_2` matrix can
consume gigabytes of memory.  The loader uses **adaptive chunking**
when reading HDF5 files to avoid peak-memory spikes:

* Files smaller than 2 GB are read in a single pass.
* Larger files are read in row-chunks and assembled incrementally.

If memory is still a concern, consider trimming the frame range before
fitting:

.. code-block:: python

   # Load only frames 100--500
   data = loader.load(frame_start=100, frame_end=500)
