.. _computational-methods:

=======================
Computational Methods
=======================

This section describes the numerical methods used to evaluate the
heterodyne correlation model efficiently and accurately. The implementation
targets JAX with JIT compilation on CPU, leveraging vectorized operations
and careful numerical safeguards.

Trapezoidal Cumulative Integration
-----------------------------------

The transport and velocity integrals
:math:`\int_{t_i}^{t_j} f(t')\, dt'` are evaluated via the
**trapezoidal cumulative sum** (``trapezoid_cumsum``), which computes:

.. math::

   S_0 = 0, \qquad
   S_k = \sum_{i=0}^{k-1}
   \frac{f(t_i) + f(t_{i+1})}{2}\, \Delta t

for :math:`k = 1, \ldots, N-1`, where :math:`\Delta t` is the uniform
time step. This achieves :math:`O(\Delta t^2)` accuracy (second-order)
per step, with total error :math:`O(N \Delta t^2) = O(\Delta t)` for the
full integral.

The cumulative sum is computed in a single vectorized pass:

1. Compute midpoint averages: :math:`m_i = (f_i + f_{i+1}) / 2`
2. Cumulative sum: :math:`S_k = \Delta t \sum_{i=0}^{k-1} m_i`
3. Prepend zero: :math:`S_0 = 0`

This yields the **numerical** antiderivative sampled at the time grid,
from which any interval integral is a simple subtraction. No analytical
closed-form expressions are ever used — the cumulative sum is the sole
mechanism for integral evaluation throughout the package.

Time Integral Matrix (NLSQ Path)
---------------------------------

For the NLSQ fitting path, the full :math:`N \times N` matrix of pairwise
integrals is needed. The function ``create_time_integral_matrix`` builds
this from the cumulative sum vector:

.. math::

   M_{ij} = S_j - S_i = \int_{t_i}^{t_j} f(t')\, dt'

This is computed as an outer subtraction:

.. code-block:: python

   M = cumsum[None, :] - cumsum[:, None]

The matrix is antisymmetric (:math:`M_{ij} = -M_{ji}`) with zero
diagonal. For transport integrals, the absolute value is taken via
``smooth_abs`` to ensure direction-independent decay. For velocity
integrals, the signed matrix is used directly since it enters the cosine
phase factor.

**Complexity**: :math:`O(N)` for the cumulative sum, :math:`O(N^2)` for
the outer subtraction and the subsequent element-wise operations. The
total memory footprint is dominated by the :math:`N \times N` matrices
(approximately 12 float64 matrices per correlation evaluation).

Upper-Triangle Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two-time correlation :math:`c_2(t_1, t_2)` is symmetric:
:math:`c_2(t_1, t_2) = c_2(t_2, t_1)`. In principle, only the upper
triangle (:math:`i \leq j`) needs evaluation. However, JAX's JIT
compilation and vectorized execution model make masking overhead
non-trivial on CPU. The current implementation evaluates the full matrix
and relies on JAX to optimize memory access patterns.

Element-Wise Evaluation (CMC Path)
-----------------------------------

For MCMC sampling (CMC path), evaluating the full :math:`N \times N`
matrix at every likelihood call is prohibitively expensive. Instead, the
CMC path uses an **element-wise evaluation** strategy:

1. **ShardGrid**: a precomputed structure holding the specific
   :math:`(i, j)` index pairs to evaluate, typically selected by data
   masking or thinning.

2. **precompute_shard_grid**: selects :math:`n_\mathrm{pairs}` index pairs
   (from the upper triangle, respecting any data mask), enabling
   :math:`O(n_\mathrm{pairs})` likelihood evaluation instead of
   :math:`O(N^2)`.

3. For each pair :math:`(i, j)`, the integral
   :math:`\int_{t_i}^{t_j} f(t')\, dt'` is obtained by looking up the
   precomputed cumulative sum: :math:`S_j - S_i`. No :math:`N \times N`
   matrix is ever allocated.

This reduces per-sample cost from :math:`O(N^2)` to
:math:`O(n_\mathrm{pairs})`, which is critical for NUTS/HMC where thousands
of likelihood evaluations are needed.

Numerically Safe Primitives
----------------------------

The physics computations involve exponentials, powers, and divisions that
can overflow or produce NaN at domain boundaries. The following safe
wrappers are used throughout:

``safe_exp(x)``
   Clips the argument to :math:`[-500, 500]` before computing
   :math:`\exp(x)`, preventing overflow to infinity. The upper bound
   gives :math:`\exp(500) \approx 1.4 \times 10^{217}`, within float64
   range.

``safe_log(x)``
   Floors the argument at :math:`10^{-30}` before computing
   :math:`\log(x)`, preventing :math:`-\infty` from zero or negative
   inputs.

``safe_power(base, exponent)``
   Returns :math:`0` for non-positive bases (the physical limit for
   :math:`t^\alpha` at :math:`t \leq 0`), and :math:`\text{base}^\text{exponent}`
   otherwise. The base is clamped at :math:`10^{-30}` internally.

``safe_divide(a, b)``
   Returns a fill value (default 0) where :math:`|b| < 10^{-30}`,
   preventing division by zero.

``smooth_abs(x)``
   Computes :math:`\sqrt{x^2 + \varepsilon}` with
   :math:`\varepsilon = 10^{-12}` as a differentiable approximation to
   :math:`|x|`. The standard ``jnp.abs(x)`` has undefined gradient at
   :math:`x = 0`, which causes NaN propagation during NUTS
   backpropagation on the matrix diagonal (where transport integrals are
   identically zero). The smooth version matches :math:`|x|` to
   :math:`O(\sqrt{\varepsilon}) \approx 10^{-6}` and has well-defined
   gradients everywhere.

Log-Space Clipping
^^^^^^^^^^^^^^^^^^

The half-transport factors :math:`\exp\!\left(-\frac{q^2}{2} \int J\, dt'\right)`
can underflow for large diffusion coefficients or long lag times. The
implementation computes these in log space:

.. math::

   \log h_{ij} \;=\; \mathrm{clip}\!\left(
   -\frac{q^2}{2} \cdot M_{ij},\; -700,\; 0\right)

then exponentiates. The lower bound of :math:`-700` corresponds to
:math:`\exp(-700) \approx 10^{-304}`, close to the float64 minimum
subnormal. This prevents silent underflow to exactly zero, which would
eliminate gradient information for the optimizer.
