# Physical Model Architecture

## Overview

The heterodyne model computes the two-time correlation function c2(t1, t2) for
a two-component scattering system where a reference (static or slowly varying)
field mixes with a sample field. The formulation follows PNAS Eq. S-95 and uses
a time-integral representation to correctly capture non-stationary dynamics.

---

## Top-Level Equation (PNAS Eq. S-95)

```
c2(t1, t2) = offset + contrast * [ref_term + sample_term + cross_term] / f_norm
```

where:

- `ref_term   = f_ref(t1)² * f_ref(t2)² * half_tr_ref(t1,t2)²`
- `sample_term = f_s(t1)² * f_s(t2)² * half_tr_sample(t1,t2)²`
- `cross_term  = 2 * f_cross(t1,t2) * half_tr_ref * half_tr_sample * cos(phase)`
- `f_norm      = [f_s(t1)² + f_ref(t1)²] * [f_s(t2)² + f_ref(t2)²]`
- `f_cross(t1,t2) = f_ref(t1)*f_s(t1) * f_ref(t2)*f_s(t2)`

`offset` and `contrast` are per-angle scaling parameters tracked separately
from the 14 physics parameters.

---

## Parameter Table

| Index | Name | Group | Description | Units | Default |
|---|---|---|---|---|---|
| 0 | D0_ref | reference | Diffusion prefactor (reference) | Å²/s^α | 1e4 |
| 1 | alpha_ref | reference | Transport exponent (reference) | — | 0.0 |
| 2 | D_offset_ref | reference | Transport rate offset (reference) | Å²/s | 0.0 |
| 3 | D0_sample | sample | Diffusion prefactor (sample) | Å²/s^α | 1e4 |
| 4 | alpha_sample | sample | Transport exponent (sample) | — | 0.0 |
| 5 | D_offset_sample | sample | Transport rate offset (sample) | Å²/s | 0.0 |
| 6 | v0 | velocity | Velocity prefactor | Å/s^β | 1e3 |
| 7 | beta | velocity | Velocity exponent | — | 0.0 |
| 8 | v_offset | velocity | Velocity offset | Å/s | 0.0 |
| 9 | f0 | fraction | Fraction amplitude | — | 0.5 |
| 10 | f1 | fraction | Fraction exponential rate | 1/s | 0.0 |
| 11 | f2 | fraction | Fraction time shift | s | 0.0 |
| 12 | f3 | fraction | Fraction baseline | — | 0.0 |
| 13 | phi0 | angle | Flow angle relative to q-vector | degrees | 0.0 |
| — | contrast | scaling | Speckle contrast (per-angle) | — | 0.5 |
| — | offset | scaling | Baseline offset (per-angle) | — | 1.0 |

The 14 physics parameters are passed as a flat array to `compute_c2_heterodyne`.
`contrast` and `offset` are passed as separate keyword arguments and are not
part of the physics array.

---

## Transport Rate and Integral

The transport rate for each component is a power law:

```
J_rate(t) = D0 * t^alpha + D_offset     (clipped to >= 0)
```

The half-transport matrix uses the cumulative integral of J_rate:

```
J_integral[i, j] = | ∫_{t_i}^{t_j} J_rate(t') dt' |

half_tr[i, j] = exp(-0.5 * q² * J_integral[i, j])
```

Self-terms recover the full decay via `half_tr²`, and the cross-term multiplies
`half_tr_ref * half_tr_sample` for geometric mean coupling.

**Implementation: O(N) via cumsum.**
`compute_transport_integral_matrix` uses a trapezoidal-rule cumsum:

```python
midpoints = (J_rate[:-1] + J_rate[1:]) / 2.0
cumsum = concatenate([0, cumsum(midpoints) * dt])
J_integral = |cumsum[None, :] - cumsum[:, None]|
```

This is O(N) instead of O(N²), and the absolute value ensures symmetric decay
regardless of time ordering. Accuracy is O(dt²) from the trapezoidal rule.

---

## Sample Fraction

The time-varying sample fraction follows a sigmoidal exponential form:

```
f_s(t) = clip(f0 * exp(f1 * (t - f2)) + f3, 0, 1)
f_ref(t) = 1 - f_s(t)
```

The exponent is pre-clipped to [-100, 100] to prevent overflow:

```python
exponent = clip(f1 * (t - f2), -100, 100)
f_s = clip(f0 * exp(exponent) + f3, 0, 1)
```

Special cases:
- `f1 = 0, f0 = 0, f3 = 0.5`: constant 50/50 mixture
- `f3 = 1.0, f0 = 0`: pure sample (homodyne limit)
- `f3 = 0.0, f0 = 0`: pure reference

---

## Velocity and Phase

The instantaneous velocity follows the same power-law form:

```
v(t) = v0 * t^beta + v_offset
```

The velocity integral matrix is computed identically via cumsum:

```
v_integral[i, j] = ∫_{t_i}^{t_j} v(t') dt'
                 = cumsum[j] - cumsum[i]
```

The cross-term oscillation (Doppler phase) combines the detector phi angle with
the fitted angle offset:

```
total_phi = phi_angle + phi0
phase[i, j] = q * cos(deg2rad(total_phi)) * v_integral[i, j]
```

---

## Model Class Hierarchy

```
HeterodyneModelBase (ABC)
├── TwoComponentModel       -- all 14 parameters free
└── ReducedModel            -- subset of parameters free, rest fixed at defaults
```

`create_model(mode)` is the factory entry point:

| Mode | Class | Free Parameters |
|---|---|---|
| `"static_ref"` | ReducedModel | D0_ref, alpha_ref, D_offset_ref |
| `"static_both"` | ReducedModel | ref (3) + sample (3) = 6 |
| `"two_component"` | TwoComponentModel | All 14 |

`ReducedModel._expand_to_full` maps the reduced parameter vector back to the
full 14-element array before calling `compute_c2_heterodyne`, using JAX
`.at[].set()` for JIT compatibility.

---

## Two-Path Integral Architecture

The physics model is evaluated via two distinct computational paths, both
producing identical c2 values but optimized for different use cases:

### Meshgrid Path (NLSQ)

**Module:** `core/jax_backend.py` — `compute_c2_heterodyne()`

Builds full N×N time-integral matrices via `trapezoid_cumsum()` →
`create_time_integral_matrix()`. Returns the complete c2 correlation matrix.

- **Memory:** O(N²) — allocates N×N arrays for transport integrals, velocity
  integrals, fraction products, and the final c2 matrix.
- **Use case:** NLSQ fitting where the full matrix is needed for residual
  computation and Jacobian evaluation via `jax.jacobian`.
- **Upper-triangle optimization:** `physics_nlsq.py` provides
  `compute_flat_residuals()` which extracts only the upper triangle of the
  symmetric c2 matrix, halving the residual vector size.

### Element-Wise Path (CMC)

**Module:** `core/physics_cmc.py` — `compute_c2_elementwise()`

Uses a pre-computed `ShardGrid` (NamedTuple with `time_grid`, `idx1`, `idx2`,
`n_pairs`) to evaluate c2 at specific (t1, t2) pairs without building the full
N×N matrix.

- **Memory:** O(n_pairs) — only stores values at the requested pair indices.
- **Use case:** NUTS sampling where the model is evaluated thousands of times
  per chain. The ShardGrid is built once (outside the NUTS loop) via
  `precompute_shard_grid()`, and only the O(G) cumsum computation (where G is
  the time grid size) is repeated on each leapfrog step.
- **Speedup:** 2–5× CMC wall time reduction by avoiding repeated searchsorted
  and singularity-floor computation inside the NUTS hot loop.

### Shared Primitives

Both paths share core functions from `core/physics_utils.py`:

| Function | Description |
|----------|-------------|
| `trapezoid_cumsum(f, dt)` | Trapezoidal cumulative integral; O(dt²) accuracy |
| `create_time_integral_matrix(cumsum)` | Signed N×N difference matrix from cumsum |
| `smooth_abs(x, eps=1e-12)` | Gradient-safe `√(x² + ε)` for transport integrals |
| `compute_transport_rate(t, D0, alpha, offset)` | Power-law rate J(t) = D₀·t^α + offset |
| `compute_velocity_rate(t, v0, beta, v_offset)` | Power-law velocity v(t) = v₀·t^β + v_offset |

---

## Numerical Safety

### Log-Space Transport Clipping

The half-transport computation uses log-space clipping to prevent underflow
without introducing artificial plateaus:

```python
log_half_tr = -0.5 * q**2 * J_integral
log_half_tr_clipped = clip(log_half_tr, -700.0, 0.0)
half_tr = exp(log_half_tr_clipped)
```

Value-space clipping (`max(exp(...), floor)`) creates flat regions where the
gradient is zero, stalling NLSQ Jacobian computation and NUTS leapfrog steps.

### Gradient-Safe Floors

All floors use `jnp.where(x > eps, x, eps)` instead of `jnp.maximum(x, eps)`.
`jnp.maximum` zeros the gradient below the floor, while `jnp.where` preserves
gradient flow through both branches.

### Velocity Sign Preservation

Unlike transport integrals (which take absolute value via `smooth_abs`),
velocity integrals are **signed**: `v_integral[i,j] = cumsum[j] - cumsum[i]`.
The sign enters the cross-term phase factor `cos(q · cos(φ) · v_integral)`,
where negative velocity represents reversed flow direction.

### Fraction Normalization Floor

The normalization denominator `f² = (f_s² + f_ref²)₁ × (f_s² + f_ref²)₂`
is floored at 1e-10 via `jnp.where` to prevent division by zero when one
component dominates completely.

---

## JAX Compatibility

All production physics functions (`compute_c2_heterodyne`,
`compute_c2_elementwise`, `compute_transport_integral_matrix`,
`compute_velocity_integral_matrix`, `compute_fraction_jit`) are JIT-compatible
and contain no Python control flow that depends on array values. They are safe
for `jax.grad` and `jax.jacobian`, which the NLSQ stage uses to compute the
analytic Jacobian. The element-wise path functions are additionally safe for
use inside NumPyro's NUTS sampler, which requires full differentiability of the
log-likelihood with respect to all sampled parameters.
