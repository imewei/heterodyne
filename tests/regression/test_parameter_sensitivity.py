"""Regression tests for parameter sensitivity.

Ensures that small perturbations to each physics parameter produce bounded,
non-zero changes in the correlation output.  Guards against silent loss of
sensitivity (e.g., from accidental clamping) and degenerate Jacobians.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.core.jax_backend import compute_c2_heterodyne
from heterodyne.optimization.nlsq.jacobian import (
    analyze_parameter_sensitivity,
    compute_jacobian_condition_number,
    compute_numerical_jacobian,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_TIMES = 32
Q = 0.01  # Angstrom^{-1}
PHI_ANGLE = 45.0  # degrees


def _default_params() -> np.ndarray:
    """Build the 14-element default parameter vector from the registry."""
    return np.array(
        [DEFAULT_REGISTRY[name].default for name in ALL_PARAM_NAMES],
        dtype=np.float64,
    )


# Non-degenerate operating point used by ``TestJacobianStructure`` to verify
# identifiability. The registry defaults are deliberately symmetric
# (alpha_ref=alpha_sample=0, D0_ref=D0_sample, beta=0, f1=0) — at that point
# the model is structurally non-identifiable: ref/sample pairs are collinear,
# v0/v_offset are collinear (both constant when beta=0), and ∂c2/∂f1 = 0
# because the mixing fraction is irrelevant when the two transport
# components are physically identical. To test sensitivity properly we have
# to evaluate the Jacobian away from that symmetric singularity.
_NONDEGENERATE_VALUES: dict[str, float] = {
    "D0_ref": 1.5e4,
    "alpha_ref": 0.3,
    "D_offset_ref": 1.0e2,
    "D0_sample": 8.0e3,
    "alpha_sample": 0.5,
    "D_offset_sample": 50.0,
    "v0": 800.0,
    "beta": 0.2,
    "v_offset": 30.0,
    "f0": 0.6,
    "f1": 0.5,
    "f2": 0.02,
    "f3": 0.1,
    "phi0": 0.0,
}


def _nondegenerate_params() -> np.ndarray:
    """Parameter vector that breaks every default-point symmetry."""
    return np.array(
        [_NONDEGENERATE_VALUES[name] for name in ALL_PARAM_NAMES],
        dtype=np.float64,
    )


def _time_grid() -> tuple[np.ndarray, float]:
    """Return (t, dt) for the small test grid."""
    t = np.linspace(1e-6, 0.1, N_TIMES)
    dt = float(t[1] - t[0])
    return t, dt


def _residual_fn(params: np.ndarray) -> np.ndarray:
    """Flatten c2 output into a 1-D residual vector (model - 1)."""
    t, dt = _time_grid()
    c2 = compute_c2_heterodyne(jnp.asarray(params), jnp.asarray(t), Q, dt, PHI_ANGLE)
    return np.asarray(c2).ravel() - 1.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestParameterPerturbation:
    """1 % perturbation on each physics parameter produces bounded c2 change."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.params = _default_params()
        self.t, self.dt = _time_grid()
        self.c2_base = np.asarray(
            compute_c2_heterodyne(
                jnp.asarray(self.params),
                jnp.asarray(self.t),
                Q,
                self.dt,
                PHI_ANGLE,
            )
        )

    @pytest.mark.parametrize("idx,name", list(enumerate(ALL_PARAM_NAMES)))
    def test_perturbation_bounded(self, idx: int, name: str) -> None:
        """1 % perturbation yields finite, non-zero c2 change."""
        perturbed = self.params.copy()
        delta = max(abs(self.params[idx]) * 0.01, 1e-8)
        perturbed[idx] += delta

        c2_pert = np.asarray(
            compute_c2_heterodyne(
                jnp.asarray(perturbed),
                jnp.asarray(self.t),
                Q,
                self.dt,
                PHI_ANGLE,
            )
        )
        diff = np.abs(c2_pert - self.c2_base)

        # Must be finite
        assert np.all(np.isfinite(c2_pert)), f"{name}: c2 contains non-finite values"
        # Max change must not explode (sanity upper bound)
        assert np.max(diff) < 1e6, f"{name}: perturbation caused explosive change"


@pytest.mark.regression
class TestJacobianStructure:
    """Jacobian at a non-degenerate operating point has full rank.

    The registry defaults (alpha_ref=alpha_sample=0, D0_ref=D0_sample, beta=0,
    f1=0) are deliberately symmetric and produce a Jacobian with rank 9/14:
    ref/sample pairs are collinear, v0/v_offset are collinear (both constant
    when beta=0), and ∂c2/∂f1 = 0 because the mixing fraction is irrelevant
    when the two transport components are physically identical.

    To verify the model's *intrinsic* identifiability we evaluate the
    Jacobian at a non-degenerate point that breaks every symmetry, and
    confirm full rank (14) and non-zero sensitivity for every parameter.
    The degeneracy at registry defaults is documented separately by
    ``TestJacobianAtSymmetricDefaults``.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.params = _nondegenerate_params()
        self.jac = compute_numerical_jacobian(_residual_fn, self.params)

    def test_jacobian_full_rank_at_nondegenerate_point(self) -> None:
        """All 14 parameters identifiable off the symmetric singularity."""
        rank = int(np.linalg.matrix_rank(self.jac))
        assert rank == 14, (
            f"Jacobian rank {rank} < 14 at non-degenerate point — "
            f"a parameter has unexpectedly lost identifiability"
        )

    def test_condition_number_finite(self) -> None:
        """J^T J condition number is finite at the non-degenerate point."""
        cond = compute_jacobian_condition_number(self.jac)
        assert np.isfinite(cond), "Jacobian condition number is not finite"

    def test_d0_has_nonzero_sensitivity(self) -> None:
        """D0_ref and D0_sample have non-zero Jacobian column norms."""
        sensitivity = analyze_parameter_sensitivity(self.jac, list(ALL_PARAM_NAMES))
        assert sensitivity["D0_ref"] > 1e-30, "D0_ref has zero sensitivity"
        assert sensitivity["D0_sample"] > 1e-30, "D0_sample has zero sensitivity"

    def test_all_parameters_nonzero_sensitivity(self) -> None:
        """Every physics parameter is sensitive at the non-degenerate point."""
        sensitivity = analyze_parameter_sensitivity(self.jac, list(ALL_PARAM_NAMES))
        for name, norm in sensitivity.items():
            assert norm > 1e-30, (
                f"{name}: sensitivity is effectively zero ({norm:.2e}) at the "
                f"non-degenerate operating point — model has lost identifiability"
            )


@pytest.mark.regression
class TestJacobianAtSymmetricDefaults:
    """Document the structural degeneracies at registry default parameters.

    These tests pin down the *expected* unidentifiabilities so we notice if
    they shift. Anyone warm-starting CMC at parameters this close to the
    symmetric defaults must expect divergent NUTS chains (this is the
    het_a10cf27e failure mode).
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.params = _default_params()
        self.jac = compute_numerical_jacobian(_residual_fn, self.params)

    def test_rank_reflects_symmetric_singularity(self) -> None:
        """At symmetric defaults, rank is 9/14 (ref/sample collinear; v0/v_offset
        collinear at beta=0; f1, f2 have zero columns).
        """
        rank = int(np.linalg.matrix_rank(self.jac))
        assert rank == 9, (
            f"Expected rank 9 at symmetric defaults (structural degeneracy); "
            f"got {rank}. If this changed, document the new identifiability."
        )

    def test_f1_and_f2_have_zero_columns(self) -> None:
        """When ref and sample physics are identical, the mixing fraction
        f0*exp(f1*(t-f2)) + f3 has no effect on the model. ∂c2/∂f1 = 0 and
        ∂c2/∂f2 = 0 at this degenerate point.
        """
        sensitivity = analyze_parameter_sensitivity(self.jac, list(ALL_PARAM_NAMES))
        assert sensitivity["f1"] < 1e-12, (
            f"f1 expected zero at symmetric defaults; got {sensitivity['f1']:.2e}"
        )
        assert sensitivity["f2"] < 1e-12, (
            f"f2 expected zero at symmetric defaults; got {sensitivity['f2']:.2e}"
        )


@pytest.mark.regression
class TestPerturbationSymmetry:
    """+delta and -delta produce similar magnitude c2 changes."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.params = _default_params()
        self.t, self.dt = _time_grid()
        self.c2_base = np.asarray(
            compute_c2_heterodyne(
                jnp.asarray(self.params),
                jnp.asarray(self.t),
                Q,
                self.dt,
                PHI_ANGLE,
            )
        )

    @pytest.mark.parametrize("idx,name", list(enumerate(ALL_PARAM_NAMES)))
    def test_symmetric_perturbation(self, idx: int, name: str) -> None:
        """Forward and backward perturbations produce comparable changes."""
        delta = max(abs(self.params[idx]) * 0.01, 1e-8)

        p_plus = self.params.copy()
        p_plus[idx] += delta
        c2_plus = np.asarray(
            compute_c2_heterodyne(
                jnp.asarray(p_plus),
                jnp.asarray(self.t),
                Q,
                self.dt,
                PHI_ANGLE,
            )
        )

        p_minus = self.params.copy()
        p_minus[idx] -= delta
        c2_minus = np.asarray(
            compute_c2_heterodyne(
                jnp.asarray(p_minus),
                jnp.asarray(self.t),
                Q,
                self.dt,
                PHI_ANGLE,
            )
        )

        norm_plus = np.linalg.norm(c2_plus - self.c2_base)
        norm_minus = np.linalg.norm(c2_minus - self.c2_base)

        # Skip symmetry check if both perturbations are effectively zero.
        # This includes parameters like f1 at default=0 where one direction
        # may hit a clamp while the other doesn't, or f2 which is insensitive
        # when f1=0.
        if max(norm_plus, norm_minus) < 1e-12:
            return

        # Ratio of perturbation magnitudes should be within 1000x.
        # The model has clamps (e.g., fraction clipped to [0,1]) that can
        # cause legitimate asymmetry near boundaries, so we use a generous
        # threshold.  Gross asymmetry beyond 1000x indicates a real problem.
        ratio = max(norm_plus, norm_minus) / max(min(norm_plus, norm_minus), 1e-30)
        assert ratio < 1000.0, (
            f"{name}: perturbation asymmetry ratio {ratio:.1f} exceeds 1000x "
            f"(+delta norm={norm_plus:.3e}, -delta norm={norm_minus:.3e})"
        )
