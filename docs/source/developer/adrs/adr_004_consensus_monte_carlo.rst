.. _adr_cmc_consensus:

ADR-004: Consensus Monte Carlo for Bayesian Inference
======================================================

:Status: Accepted
:Date: 2026-05-19
:Deciders: Core team

Context
-------

The heterodyne 14-parameter model posterior

.. math::

   p(\theta\,|\,\mathcal{D}) \propto p(\mathcal{D}\,|\,\theta)\, p(\theta)

is evaluated over thousands of :math:`(t_1, t_2)` data points per shard. Standard MCMC
(single-chain NUTS) requires O(N) likelihood evaluations per leapfrog step, where N is
the total number of time-pair observations. For a 1000-frame dataset, N ≈ 500K pairs;
a single NUTS step costs ~50 ms on a 32-core workstation. Convergence requires ~2000
warmup + 1000 production steps: total ~2.5 hours for a single chain.

Users need posterior samples for all 14 physical parameters plus per-angle scaling
within 30–90 minutes on typical beamline hardware (32–64 cores).


Decision
--------

Heterodyne implements **Consensus Monte Carlo** (CMC, Scott et al. 2016) as its Bayesian
backend. The algorithm is:

1. **Shard**: Partition the :math:`n` data points into :math:`M` shards of size
   :math:`n_s \approx n/M`.
2. **Parallel NUTS**: Run independent NUTS chains on each shard (in separate worker
   processes), obtaining :math:`K` posterior samples :math:`\{\theta^{(k)}_s\}` from the
   shard-specific posterior :math:`p_s(\theta\,|\,\mathcal{D}_s)`.
3. **Consensus**: Combine the :math:`M` sets of shard samples into a single approximation
   of the global posterior :math:`p(\theta\,|\,\mathcal{D})`.

The multiprocessing backend spawns :math:`N_\mathrm{workers} = \lfloor N_\mathrm{cores}/2\rfloor - 1`
worker processes, each with 4 virtual JAX devices (via
``--xla_force_host_platform_device_count=4``). Each worker runs NUTS in ``parallel`` mode
(pmap over 4 devices), achieving near-full CPU utilization.


Shard Size Selection
~~~~~~~~~~~~~~~~~~~~

NUTS is O(n) per leapfrog step in the shard size :math:`n_s`. A shard that is too large
produces slow NUTS; one that is too small produces a shard posterior that is a poor
approximation of the global posterior (the CMC consensus is theoretically justified only
when each shard contains enough data for the Bernstein–von Mises theorem to apply).

The default ``max_points_per_shard: "auto"`` selects a shard size that balances these
constraints. Manual override is possible via:

.. code-block:: yaml

   optimization:
     cmc:
       sharding:
         max_points_per_shard: 5000  # Override auto-selection

.. warning::

   NUTS is O(n) per leapfrog step. Never use 100K+ shard size. The default ``"auto"``
   is the recommended setting.


NUTS Warmup Floor
~~~~~~~~~~~~~~~~~

``num_warmup`` defaults to 1500 because ``dense_mass=True`` on the 14-parameter
heterodyne model requires ≥ 100 mass-matrix adaptation steps per dimension. Lowering
this without simultaneously reducing parameter count or disabling ``dense_mass`` will
produce high R-hat and divergence storms. This floor is enforced by tests.


JIT Cache in Workers
~~~~~~~~~~~~~~~~~~~~

In JAX 0.8+, the ``JAX_COMPILATION_CACHE_DIR`` environment variable alone does **not**
enable the persistent cache. Workers must call:

.. code-block:: python

   jax.config.update("jax_compilation_cache_dir", cache_dir)
   jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

This is done in the worker initialization function in
``optimization/cmc/backends/multiprocessing.py``.


Environment Isolation
~~~~~~~~~~~~~~~~~~~~~

Before spawning workers, the backend:

1. Saves the current environment (``OMP_PROC_BIND``, ``OMP_PLACES``).
2. Clears ``OMP_PROC_BIND`` and ``OMP_PLACES`` to prevent OpenMP thread binding
   conflicts between workers.
3. Sets ``OMP_NUM_THREADS=1`` or 2 per worker to prevent thread oversubscription
   (each worker manages its own JAX device count via XLA_FLAGS).
4. Restores the parent environment after all workers are spawned.


Rationale
---------

**Why CMC over standard MCMC?**

Standard single-chain NUTS on all N data points takes O(hours) per chain, without
parallelism. CMC decomposes the problem: each shard runs NUTS independently (trivially
parallel), and the consensus step is a simple weighted average of the shard posteriors.
The total wall time is approximately ``T_shard / N_workers``, where ``T_shard`` is the
time to run NUTS on a single shard.

**Why NUTS over Metropolis-Hastings?**

The 14-parameter heterodyne model has strong posterior correlations (e.g.,
:math:`D_{0,\mathrm{ref}}` and :math:`D_{0,\mathrm{sample}}`). Metropolis-Hastings
requires hand-tuned proposal distributions for each parameter pair, which is
impractical for production use. NUTS adapts its step size and trajectory length
automatically via dual averaging.

**Why dense mass matrix?**

The 14-parameter space has off-diagonal correlations that would cause NUTS to take very
small steps with a diagonal mass matrix. The dense mass matrix (full Hessian
approximation) enables NUTS to take larger, correlated steps, reducing the number of
leapfrog steps needed for convergence.

**Why multiprocessing over threading?**

JAX uses the Global Interpreter Lock (GIL) at the Python level; threading would
serialize the NUTS leapfrog steps. Multiprocessing (``spawn`` mode) gives each worker
a fresh Python interpreter with its own JAX device context and XLA flags.


Consequences
------------

**Positive**:

- Wall time scales as ``T_shard / N_workers`` — near-linear speedup with core count.
- Each shard posterior is independently diagnosed (R-hat, ESS, divergences per shard).
- Failed shards can be retried or skipped without re-running all shards.
- The consensus posterior approximates the full posterior to the same degree regardless
  of the number of shards (Scott et al. 2016 guarantee).

**Negative / Accepted trade-offs**:

- CMC is an approximation: the consensus posterior is not exactly equal to the full
  posterior. The approximation is better as shard size increases.
- Worker spawning overhead (~2–5 s per worker) adds latency for small datasets.
- The multiprocessing backend requires ``spawn`` mode (not ``fork``), which re-imports
  all modules in each worker. Heavy imports (JAX, NumPyro) add ~3 s per worker startup.
- ArviZ diagnostics must aggregate across shards; the ``summarize_diagnostics()``
  function handles this but adds code complexity.


Alternatives Considered
-----------------------

**A. Standard NUTS on full data**

Correct by construction. Rejected because: too slow for practical use (> 8 hours on
typical hardware for the 14-parameter heterodyne model with a 1000-frame dataset).

**B. Variational inference (ADVI)**

ADVI in NumPyro produces approximate posteriors much faster than NUTS. Rejected because:
mean-field ADVI produces overconfident posteriors for correlated parameter spaces; the
full-rank ADVI covariance requires O(d²) parameters (196 for 14 dims), which is slow
to optimize and numerically unstable.

**C. Parallel tempering (Pigeons-style)**

Non-reversible parallel tempering (NRPT) is superior to CMC for multimodal posteriors.
Rejected for the default backend because: the heterodyne posterior is expected to be
unimodal after NLSQ warm-start; the computational cost of the swap moves between
temperature levels adds overhead. May be reconsidered if multimodal posteriors are
observed in practice.

**D. Ensemble sampler (emcee-style)**

Affine-invariant ensemble samplers handle correlated posteriors without tuning.
Rejected because: ensemble samplers are not trivially parallelizable across data shards,
and the CMC consensus framework is not directly applicable to ensemble states.


.. seealso::

   - :ref:`adr_jax_cpu_only` — CPU-only JAX decision
   - :ref:`adr_nlsq_primary` — NLSQ / CMC architectural split
   - :ref:`theory_computational_methods` — CMC mathematical details
