.. _cmaes-optimization:

========================
CMA-ES Global Optimisation
========================

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a
derivative-free global optimisation algorithm.  The heterodyne package
provides a CMA-ES wrapper built on `evosax <https://github.com/RobertTLange/evosax>`_
for cases where gradient-based NLSQ fails to find the global minimum.


When to Use CMA-ES
===================

CMA-ES is most useful when:

* **NLSQ converges to a local minimum** -- Multi-start NLSQ with
  different initial points yields inconsistent solutions with similar
  :math:`\chi^2` values.
* **The landscape is multi-modal** -- Bayesian inference reveals
  bimodal posteriors or shard disagreement.
* **Parameters are poorly constrained** -- The Jacobian is
  near-singular, and the covariance matrix is unreliable.
* **Initial guesses are poor** -- No good warm-start is available from
  prior measurements or physical intuition.

CMA-ES is **not** recommended as the default optimiser because:

* It requires many more function evaluations than NLSQ (thousands vs.
  tens).
* It does not provide a covariance matrix or Jacobian.
* For well-behaved problems, NLSQ is faster and more precise.


CMAESConfig
===========

The :class:`~heterodyne.optimization.nlsq.cmaes_wrapper.CMAESConfig`
dataclass controls the algorithm:

.. code-block:: python

   from heterodyne.optimization.nlsq.cmaes_wrapper import CMAESConfig

   config = CMAESConfig(
       sigma0=0.25,            # Initial step-size (fraction of parameter range)
       population_size=64,     # Number of candidate solutions per generation
       max_generations=500,    # Maximum number of generations
       seed=42,                # JAX PRNG seed for reproducibility
   )

Key parameters:

``sigma0`` (``float``, default 0.25)
   Initial standard deviation of the search distribution, expressed as
   a fraction of the parameter range.  A good default is ~1/4 of the
   expected parameter range.  Too small: slow exploration.  Too large:
   the search wastes evaluations on infeasible regions.

``population_size`` (``int``)
   Number of candidate solutions evaluated per generation.  Larger
   populations improve exploration of multi-modal landscapes but
   increase computation per generation.  The default heuristic is
   :math:`4 + \lfloor 3 \ln(n) \rfloor` where :math:`n` is the
   number of parameters.

``max_generations`` (``int``, default 500)
   Upper bound on the number of generations.  The algorithm may
   terminate earlier if convergence criteria are met.

``seed`` (``int``)
   JAX PRNG key for deterministic initialisation and sampling.


Running CMA-ES
==============

The :func:`~heterodyne.optimization.nlsq.cmaes_wrapper.fit_with_cmaes`
function accepts an objective function, initial parameters, and bounds:

.. code-block:: python

   from heterodyne.optimization.nlsq.cmaes_wrapper import fit_with_cmaes
   import numpy as np

   # Define objective (e.g., sum of squared residuals)
   def objective(params: np.ndarray) -> float:
       residuals = model.residuals(params, c2_data, phi_angle=45.0)
       return float(np.sum(residuals ** 2))

   result = fit_with_cmaes(
       objective_fn=objective,
       initial_params=nlsq_result.parameters,
       bounds=(lower_bounds, upper_bounds),
       config=config,
   )

The result can then be used to warm-start a final NLSQ refinement or
directly as input to the Bayesian pipeline.


Sigma Scheduling
================

CMA-ES adapts the covariance matrix of its search distribution
automatically.  The step-size :math:`\sigma` evolves according to the
*cumulative step-size adaptation* (CSA) rule, which increases
:math:`\sigma` when consecutive steps are correlated (moving in a
consistent direction) and decreases it when steps are uncorrelated
(oscillating around a minimum).

No manual sigma schedule is needed.  However, monitoring the
:math:`\sigma` trajectory (available in the result metadata) can help
diagnose convergence problems:

* **Sigma decays steadily** -- Normal convergence toward a minimum.
* **Sigma oscillates without decaying** -- The landscape is flat or
  the population size is too small.
* **Sigma grows** -- The algorithm is escaping a local basin, which
  may be desirable in a multi-modal landscape.


Combining CMA-ES with NLSQ
===========================

A common pattern is to use CMA-ES for coarse global search, then
refine with NLSQ:

.. code-block:: python

   # Step 1: Global search with CMA-ES
   cmaes_result = fit_with_cmaes(
       objective_fn=objective,
       initial_params=initial_guess,
       bounds=(lb, ub),
       config=CMAESConfig(population_size=128, max_generations=300),
   )

   # Step 2: Local refinement with NLSQ
   nlsq_result = fit_nlsq_jax(
       model=model,
       c2_data=c2_data,
       phi_angle=45.0,
       config=NLSQConfig(initial_params=cmaes_result),
   )

This two-stage approach combines the global exploration of CMA-ES with
the fast local convergence and uncertainty estimation of NLSQ.
