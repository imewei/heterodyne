.. _adr_nlsq_primary:

ADR-002: NLSQ / CMC Architectural Split
=========================================

:Status: Accepted
:Date: 2026-05-19
:Deciders: Core team

Context
-------

Heterodyne must provide both:

1. **Fast point estimates** at the beamline (seconds to minutes) to guide experiment
   decisions in real time.
2. **Full posterior distributions** for publication-quality uncertainty quantification
   (minutes to hours, run after data collection).

Both requirements must be served by the same 14-parameter physics model:
``(D0_ref, alpha_ref, D_offset_ref, D0_sample, alpha_sample, D_offset_sample,
v0, beta, v_offset, f0, f1, f2, f3, phi0)`` plus 2 per-angle scaling parameters.

A single algorithm cannot satisfy both requirements: MCMC (NUTS) produces exact
posteriors but is too slow for real-time use; gradient-based NLSQ converges in seconds
but returns only a point estimate with no uncertainty.


Decision
--------

Heterodyne implements a **two-stage pipeline** where NLSQ is the **primary** optimizer
and CMC is the **secondary** Bayesian sampler:

.. code-block:: python

   # Stage 1: Fast point estimate (seconds to minutes)
   nlsq_result = fit_nlsq_jax(data, config)

   # Stage 2: Full posterior (minutes to hours), initialized from Stage 1
   cmc_result = fit_cmc_jax(data, config, nlsq_result=nlsq_result)

The two stages share the same physics model (via ``HeterodyneModel``) but use
**separate physics backends**:

- NLSQ uses ``jax_backend.py`` (**meshgrid mode**): evaluates :math:`c_2(t_i, t_j)` for
  all pairs simultaneously using 2D JAX broadcasting.
- CMC uses ``physics_cmc.py`` (**element-wise mode**): evaluates :math:`c_2` at specific
  :math:`(t_1, t_2)` pairs given as flat shard vectors.

This split is intentional and maintained as a first-class architectural boundary.


Rationale
---------

**Two physics backends: why not one?**

The two evaluation patterns have fundamentally different JAX execution graphs:

- Meshgrid mode: ``create_time_integral_matrix()`` builds an N×N matrix via
  ``jax.vmap`` over the outer time axis, then uses 2D broadcasting for the correlation
  formula. This is O(N²) memory but trivially JIT-compiled and differentiable.
- Element-wise mode: ``precompute_shard_grid()`` builds a cumulative-sum lookup
  (``trapezoid_cumsum``), then indexes into it with flat ``(t1, t2)`` vectors. This is
  O(n_pairs) memory and designed for per-shard NUTS evaluation where n_pairs ≪ N².

Unifying the two into a single function would require runtime branching inside a JIT
context (``jax.lax.cond``) that adds dead computation on both paths, or a non-JIT-safe
Python-level branch that recompiles on mode change. The separation avoids both.

**NLSQ as the primary optimizer**

NLSQ (Levenberg-Marquardt via ``nlsq.curve_fit``) converges to a point estimate in
10–60 seconds for typical datasets. This is fast enough for beamline use. The trust-region
algorithm handles the non-convex 14-parameter landscape via automatic damping and Jacobian
scaling.

Jacfwd (forward-mode automatic differentiation) is used for the Jacobian, giving a 211×
speedup over finite differences.

**CMC warm-start**

Without an NLSQ warm-start, CMC NUTS divergence rates exceed 20% for the heterodyne
model. The 14-parameter space has strong degeneracies (e.g., ``D0_ref``/``D0_sample``
trading) that trap NUTS in unphysical regions during warmup. The NLSQ point estimate
provides a physically meaningful initialization that reduces warmup divergences to
< 2%.

**Stage independence**

NLSQ can be run without CMC (for fast beamline analysis). CMC can be run without NLSQ
(using registry defaults as initialization), though this is not recommended for the
14-parameter model. The CLI supports both patterns:

.. code-block:: bash

   # NLSQ only
   heterodyne --config config.yaml --data data.h5 --method nlsq

   # NLSQ + CMC
   heterodyne --config config.yaml --data data.h5 --method cmc


Consequences
------------

**Positive**:

- Real-time beamline feedback via NLSQ (< 60 s).
- Publication-quality posteriors via CMC (warm-started, low divergence).
- Both stages use the same ``HeterodyneModel`` class: parameter bounds, priors, and
  physics model are defined once.
- The meshgrid/element-wise split allows each backend to be independently optimized
  and tested.

**Negative / Accepted trade-offs**:

- Two physics backends must be kept in numerical parity. Divergence between them is
  a correctness bug, not a design choice. Regression tests enforce parity.
- The two-stage API is more complex than a single ``fit()`` function. Users must
  understand that CMC output depends on NLSQ results.
- ``HeterodyneModel`` carries both backends, increasing module surface area.


Alternatives Considered
-----------------------

**A. MCMC only (no NLSQ)**

Simpler API. Rejected because: no fast feedback path; CMC without warm-start has 20%+
divergence rates for the heterodyne 14-parameter model; users cannot run analysis at
the beamline in real time.

**B. Single physics backend for both**

Would simplify maintenance. Rejected because: the meshgrid and element-wise calling
conventions cannot be efficiently unified without JAX control flow that is not JIT-safe
for one of the two use cases.

**C. Variational inference (VI) instead of MCMC**

VI (e.g., ADVI in NumPyro) produces approximate posteriors much faster than NUTS.
Rejected because: VI produces overconfident posteriors for the correlated parameters
in the heterodyne model (known pathology of mean-field VI for high-dimensional
correlated spaces); full NUTS is required for accurate uncertainty quantification.

**D. Ensemble sampler (emcee-style)**

Affine-invariant ensemble samplers handle correlated posteriors without tuning.
Rejected because: ensemble samplers are not trivially parallelizable across data shards,
and the CMC consensus framework is not directly applicable to ensemble states.


.. seealso::

   - :ref:`developer_architecture` — full data flow diagram
   - :ref:`adr_cmc_consensus` — why Consensus Monte Carlo
   - :ref:`theory_computational_methods` — mathematical details of both methods
