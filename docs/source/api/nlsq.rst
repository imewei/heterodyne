======================
NLSQ Optimization
======================

Non-linear least squares fitting with JAX JIT-compiled residuals,
multiple solver strategies, Fourier reparameterization for multi-angle
fits, CMA-ES global optimization, and multi-start support.

Core Fitting
============

.. automodule:: heterodyne.optimization.nlsq.core
   :members: fit_nlsq_jax
   :undoc-members:
   :show-inheritance:

Configuration
=============

.. automodule:: heterodyne.optimization.nlsq.config
   :members: NLSQConfig
   :undoc-members:
   :show-inheritance:

Results
=======

.. automodule:: heterodyne.optimization.nlsq.results
   :members: NLSQResult
   :undoc-members:
   :show-inheritance:

Adapter
=======

.. automodule:: heterodyne.optimization.nlsq.adapter
   :members:
   :undoc-members:
   :show-inheritance:

JIT Strategies
==============

.. automodule:: heterodyne.optimization.nlsq.strategies
   :members:
   :undoc-members:
   :show-inheritance:

Fourier Reparameterization
==========================

Joint multi-angle fitting via Fourier coefficient reparameterization of
contrast and offset parameters.

.. automodule:: heterodyne.optimization.nlsq.fourier_reparam
   :members: FourierReparameterizer
   :undoc-members:
   :show-inheritance:

CMA-ES Wrapper
==============

Covariance Matrix Adaptation Evolution Strategy for global optimization.

.. automodule:: heterodyne.optimization.nlsq.cmaes_wrapper
   :members: fit_with_cmaes
   :undoc-members:
   :show-inheritance:

Multi-Start Optimizer
=====================

.. automodule:: heterodyne.optimization.nlsq.multistart
   :members: MultiStartOptimizer
   :undoc-members:
   :show-inheritance:
