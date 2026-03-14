.. _troubleshooting:

===============
Troubleshooting
===============

This page lists common problems encountered when using the heterodyne
package and their recommended solutions.


NLSQ Convergence Failure
-------------------------

**Symptom:** ``result.success`` is ``False``; the solver reports
"Maximum number of function evaluations reached" or "Cost function
not decreasing."

**Possible causes and remedies:**

1. **Poor initial guess** -- Use multi-start optimisation
   (``n_starts=20`` or more) to explore the parameter space from
   diverse starting points.

2. **Multi-modal landscape** -- Switch to CMA-ES for global search,
   then refine with NLSQ.  See :doc:`../03_advanced_topics/cmaes_optimization`.

3. **Tight bounds** -- Widen parameter bounds in the configuration.
   Check whether any parameter is hitting its bound at the solution
   (``result.validate()`` will flag this).

4. **Ill-conditioned Jacobian** -- Check the condition number.  If
   extremely large, consider fixing one or more weakly constrained
   parameters (e.g., ``D_offset_ref``, ``f2``).

5. **Insufficient function evaluations** -- Increase ``max_nfev`` in
   ``NLSQConfig``.


CMC Divergent Transitions
-------------------------

**Symptom:** NumPyro reports divergent transitions during NUTS sampling;
``result.convergence_passed`` is ``False``.

**Remedies:**

1. **Increase target acceptance probability** -- Set
   ``target_accept_prob=0.95`` in ``CMCConfig``.  This reduces the
   step size, improving sampling in regions of high curvature.

2. **Check priors** -- Overly wide priors can send the sampler into
   unphysical regions.  Reduce ``nlsq_prior_width_factor`` from 5.0
   to 3.0.

3. **Tighten bounds** -- Ensure parameter bounds exclude regions
   where the model is undefined or numerically unstable.

4. **Increase warmup** -- More warmup iterations allow the sampler to
   adapt its step size and mass matrix more thoroughly.


Memory Errors
-------------

**Symptom:** ``MemoryError`` or the process is killed by the OOM killer.

**Remedies:**

1. **Switch to chunked or sequential strategy** for NLSQ:

   .. code-block:: python

      config = NLSQConfig(strategy="chunked", chunk_size=128)

2. **Trim frame range** -- Load only the frames you need:

   .. code-block:: python

      data = loader.load(frame_start=0, frame_end=500)

3. **Increase CMC shards** -- More shards means less memory per shard:

   .. code-block:: python

      cmc_config = CMCConfig(num_shards=16)

4. **Check for memory leaks** -- If memory grows across multiple fits,
   ensure you are not accumulating JAX arrays in a loop without
   releasing references.


JAX Compilation Slow
--------------------

**Symptom:** The first NLSQ call takes minutes before any fitting
begins.

**Causes:**

1. **Large array shapes** -- JIT compilation time scales with the
   complexity of the computation graph.  For very large
   :math:`C_2` matrices, use the chunked or sequential strategy
   to avoid compiling a single monolithic kernel.

2. **Inconsistent shapes** -- JAX recompiles whenever input shapes
   change.  Ensure all angles use the same number of frames, or
   pad to a common size.

3. **Thread contention** -- Verify ``OMP_NUM_THREADS`` is set
   appropriately.  Over-subscription can slow compilation.

4. **XLA flags not set** -- Run ``heterodyne-config-xla`` to
   configure optimal compiler flags.


Parameter at Bounds
-------------------

**Symptom:** A fitted parameter is exactly at its lower or upper
bound; ``result.validate()`` may not flag this directly, but
uncertainties for that parameter will be unreliable.

**Remedies:**

1. **Widen bounds** -- If the physical range permits, increase the
   bound.

2. **Check initial values** -- A poor starting point near a bound
   can trap the optimiser.

3. **Fix the parameter** -- If the data cannot constrain a parameter,
   fix it to a physically motivated value and re-fit.

4. **Inspect the residuals** -- Parameter-at-bound may indicate a
   model mismatch rather than a bound problem.


Poor R-hat After CMC
--------------------

**Symptom:** :math:`\hat{R} > 1.1` for one or more parameters.

**Remedies:**

1. **Run longer** -- Increase ``num_warmup`` and ``num_samples``.

2. **Increase chains** -- More chains provide better :math:`\hat{R}`
   estimates: ``num_chains=6`` or ``num_chains=8``.

3. **Check for bimodality** -- Use
   ``plot_shard_comparison(shard_results)`` to see if shards converge
   to different modes.

4. **Improve warm-start** -- A better NLSQ solution as the CMC
   warm-start helps chains explore the correct region faster.


NaN or Inf in Results
---------------------

**Symptom:** Fitted parameters contain ``NaN`` or ``Inf``.

**Causes:**

1. **NaN in input data** -- Check ``np.any(np.isnan(c2_data))``.
   The loader's validation should catch this, but preprocessed data
   may slip through.

2. **Numerical overflow** -- Very large ``D0`` or ``v0`` values
   combined with long time spans can cause overflow in the
   exponential.  Tighten bounds.

3. **Division by zero** -- If the fraction function reaches exactly
   0 or 1, some terms may become degenerate.  Check ``f0`` and
   ``f3`` bounds.
