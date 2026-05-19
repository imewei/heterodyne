.. _batch_processing:

Batch Processing
================

This guide describes how to run heterodyne on multiple datasets or configuration
files in sequence or in parallel. Batch processing is useful for:

- **Parameter sweeps**: systematically varying one or more YAML parameters (e.g., ``dt``,
  ``start_frame``, ``end_frame``) across a range of values.
- **Dataset surveys**: running the same configuration over many HDF5 data files.
- **Reproducibility checks**: re-running analyses with different random seeds to verify
  posterior stability.
- **Comparative analysis**: comparing NLSQ and CMC results across experimental conditions.


Recommended Pattern: Shell Loop
---------------------------------

The simplest batch pattern is a shell loop over YAML config files. Each invocation of
``heterodyne`` (or ``ht``) is independent: it writes output to the directory configured
in its YAML file, and exits with a status code (see :ref:`batch_exit_codes`).

.. code-block:: bash

   # Run all configs in a directory, one at a time
   for cfg in configs/*.yaml; do
       echo "Processing: $cfg"
       heterodyne --config "$cfg" --data data/experiment.h5
   done

If each config file specifies its own ``output_dir``, outputs accumulate in separate
directories without collision.


Parallel Execution
------------------

Heterodyne is CPU-only and uses multiple cores internally (CMC spawns worker processes
that each use several cores). For batch parallelism across configs, limit per-job
parallelism to avoid oversubscribing the CPU.

**Using GNU parallel**:

.. code-block:: bash

   # Run 4 configs in parallel, each using 8 cores
   export HETERODYNE_NUM_WORKERS=2   # Limit CMC workers per job
   parallel -j 4 heterodyne --config {} --data data/experiment.h5 ::: configs/*.yaml

**Using xargs**:

.. code-block:: bash

   # Run up to 4 configs in parallel
   ls configs/*.yaml | xargs -P 4 -I{} heterodyne --config {} --data data/experiment.h5

**Core budget rule of thumb**: If the workstation has ``N`` cores, run at most
``N / (num_cmc_workers + 1)`` jobs in parallel, where ``num_cmc_workers`` is the
CMC worker count configured in YAML (default: ``⌊N_cores/2⌋ − 1``). Oversubscription
causes thread contention that slows all jobs without improving total throughput.

.. tip::

   For NLSQ-only batch runs (``--method nlsq``), there are no CMC workers. NLSQ uses
   a single process with ``jax.vmap`` over angles. You can run more jobs in parallel:
   ``N / 2`` jobs is a reasonable starting point.


Collecting Outputs
------------------

Each ``heterodyne`` run writes results to the configured ``output_dir``. By default,
outputs are named by run ID (e.g., ``het_<8-char-hash>/``). To collect results:

.. code-block:: bash

   # List all result JSON files
   find results/ -name "*.json" | sort

   # Extract a specific field from all results (requires jq)
   find results/ -name "*.json" | xargs -I{} jq '.nlsq.D0_ref' {}

   # Tabulate D0_ref and alpha_ref across all runs
   find results/ -name "*.json" \
     | xargs -I{} jq -r '[input_filename, .nlsq.D0_ref, .nlsq.alpha_ref] | @csv' {} \
     | sort > summary.csv

The NPZ files (``*.npz``) alongside each JSON contain the full posterior samples for
CMC runs. NPZ files store plain NumPy arrays (no pickle); load them with
``numpy.load()``:

.. code-block:: python

   import numpy as np
   import pathlib

   results = {}
   for npz_path in pathlib.Path("results/").rglob("*.npz"):
       data = np.load(npz_path)   # NPZ arrays are plain float64; no pickle needed
       results[npz_path.stem] = {key: data[key] for key in data.files}


.. _batch_exit_codes:

Exit Codes
----------

Heterodyne returns standard exit codes that batch scripts can check:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - Success: analysis completed, outputs written.
   * - ``1``
     - Failure: unrecoverable error (I/O error, invalid config, unexpected exception).
   * - ``2``
     - Convergence failure: NLSQ did not converge, or CMC R-hat exceeds threshold.
   * - ``130``
     - Interrupted: Ctrl-C / SIGINT received.

Capture exit codes in a loop to detect and log failures:

.. code-block:: bash

   failed=()
   for cfg in configs/*.yaml; do
       heterodyne --config "$cfg" --data data/experiment.h5
       code=$?
       if [[ $code -ne 0 ]]; then
           echo "FAILED: $cfg (exit code $code)"
           failed+=("$cfg")
       fi
   done

   echo "Failed configs: ${failed[*]}"

Convergence failures (exit code 2) are not hard errors: the output JSON is written with
a ``convergence_warning`` field, and the NPZ contains whatever samples were collected.
You can inspect these runs and decide whether to re-run with adjusted settings.


Reproducibility
---------------

To reproduce a batch run exactly:

1. **Pin the heterodyne version**: record ``pip show heterodyne`` or ``uv pip show
   heterodyne`` output. The run ID in each output directory encodes the heterodyne
   commit hash.

2. **Set the random seed**: add ``seed`` to the CMC config in each YAML file:

   .. code-block:: yaml

      optimization:
        cmc:
          seed: 42   # Integer seed for NumPyro NUTS

3. **Record the data file hash**: the output JSON includes ``data_sha256``, which is
   the SHA-256 checksum of the input HDF5 file. Verify this matches across re-runs.

4. **Use a lockfile**: ``uv.lock`` pins all Python dependencies. Commit ``uv.lock``
   alongside your config files.


Parameter Sweep Example
-----------------------

The following script runs a sweep over ``start_frame`` values and collects results:

.. code-block:: bash

   #!/usr/bin/env bash
   set -euo pipefail

   DATA="data/experiment_run42.h5"
   BASE_CONFIG="configs/base.yaml"
   OUTDIR="sweep_start_frame"
   mkdir -p "$OUTDIR"

   for start in 100 500 1000 2000 5000; do
       cfg="$OUTDIR/config_sf${start}.yaml"
       # Patch start_frame in a copy of the base config
       sed "s/start_frame:.*/start_frame: ${start}/" "$BASE_CONFIG" > "$cfg"

       echo "Running start_frame=${start} ..."
       heterodyne --config "$cfg" --data "$DATA" --output-dir "$OUTDIR/sf${start}/"
       echo "  Done (exit $?)"
   done

   # Summarize D0_ref across the sweep
   echo "start_frame,D0_ref"
   for start in 100 500 1000 2000 5000; do
       D0=$(jq '.nlsq.D0_ref' "$OUTDIR/sf${start}/"*/result.json 2>/dev/null || echo "N/A")
       echo "$start,$D0"
   done


CMC-Specific Batch Concerns
----------------------------

CMC (Consensus Monte Carlo) runs take significantly longer than NLSQ-only runs (minutes
to hours vs. seconds to minutes). For CMC batch jobs:

- **Always NLSQ-warm-start CMC**: pass the NLSQ result as ``nlsq_result`` in the Python
  API, or ensure the YAML includes ``warmstart: true`` (the default). CMC without warm-start
  has high divergence rates for the 14-parameter heterodyne model.

- **Monitor R-hat per shard**: the CMC output JSON includes per-shard diagnostics.
  A shard with R-hat > ``max_r_hat`` (default: 1.05) is flagged. Review these before
  treating the consensus posterior as converged.

- **Use ``max_points_per_shard: "auto"``**: the automatic shard size selection is tuned
  for the heterodyne model. Manual shard sizes above 50K points will cause very slow NUTS
  steps. See :ref:`adr_cmc_consensus` for the underlying constraints.

- **Disk space**: each CMC run writes NPZ files containing all posterior samples. For
  a sweep of 20 configurations with 1000 samples each and 14 parameters, expect
  approximately 20 × 1000 × 14 × 8 bytes ≈ 2.2 MB of NPZ data per run.


Cross-References
----------------

.. seealso::

   - :ref:`adr_cmc_consensus` — CMC architectural decision and shard size guidance
   - :ref:`theory_anti_degeneracy` — anti-degeneracy system description
