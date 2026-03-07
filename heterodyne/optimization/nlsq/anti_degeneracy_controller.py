"""Anti-degeneracy controller for NLSQ optimization.

Detects parameter configurations that are degenerate or near-degenerate,
such as strongly correlated parameter pairs, parameters stuck at bounds,
or cost-function plateaus.  Particularly relevant for the 14-parameter
heterodyne model where D0_ref/D0_sample correlation and f0/f3 trading
are known failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.config.parameter_manager import ParameterManager
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)

# Known problematic parameter pairs in the heterodyne model
_KNOWN_DEGENERATE_PAIRS: list[tuple[str, str, str]] = [
    ("D0_ref", "D0_sample", "diffusion coefficient degeneracy"),
    ("f0", "f3", "fraction amplitude/baseline trading"),
    ("v0", "v_offset", "velocity magnitude/offset trading"),
    ("alpha_ref", "D0_ref", "exponent-prefactor compensation"),
    ("alpha_sample", "D0_sample", "exponent-prefactor compensation"),
]


@dataclass
class DegeneracyCheck:
    """Result of a degeneracy check.

    Attributes:
        is_degenerate: Whether a degeneracy was detected.
        affected_params: Names of parameters involved.
        message: Human-readable description of the issue.
        suggested_action: Recommended remediation step.
    """

    is_degenerate: bool
    affected_params: list[str] = field(default_factory=list)
    message: str = ""
    suggested_action: str = ""


class AntiDegeneracyController:
    """Detect and report parameter degeneracies.

    Performs three classes of checks on a completed NLSQ result:

    1. **Correlation degeneracy** -- parameter pairs with |r| > threshold
       in the covariance matrix.
    2. **Bound saturation** -- parameters whose fitted values sit at or
       very near their optimization bounds.
    3. **Cost plateau** -- the optimization made many iterations but
       achieved negligible cost reduction.

    Parameters
    ----------
    correlation_threshold : float
        Absolute correlation coefficient above which a pair is flagged.
    bound_tolerance : float
        Relative tolerance for declaring a parameter "at bounds".
        A parameter is flagged if it is within
        ``bound_tolerance * (upper - lower)`` of either bound.
    plateau_min_iterations : int
        Minimum iteration count before plateau detection activates.
    plateau_cost_rtol : float
        Relative cost change below which a plateau is declared.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.99,
        bound_tolerance: float = 1e-4,
        plateau_min_iterations: int = 10,
        plateau_cost_rtol: float = 1e-10,
    ) -> None:
        if not 0 < correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in (0, 1]")
        if bound_tolerance < 0:
            raise ValueError("bound_tolerance must be non-negative")

        self._corr_thresh = correlation_threshold
        self._bound_tol = bound_tolerance
        self._plateau_min_iter = plateau_min_iterations
        self._plateau_rtol = plateau_cost_rtol

    def check(
        self,
        result: NLSQResult,
        param_manager: ParameterManager | None = None,
    ) -> DegeneracyCheck:
        """Run all degeneracy checks on a fit result.

        Args:
            result: Completed NLSQ result with parameters and (optionally)
                covariance.
            param_manager: Optional parameter manager providing bounds.
                When ``None``, the bound-saturation check is skipped.

        Returns:
            A :class:`DegeneracyCheck` summarizing the worst issue found.
        """
        issues: list[str] = []
        affected: list[str] = []
        actions: list[str] = []

        # --- 1. Correlation degeneracy ---
        corr_result = self._check_correlation(result)
        if corr_result.is_degenerate:
            issues.append(corr_result.message)
            affected.extend(corr_result.affected_params)
            actions.append(corr_result.suggested_action)

        # --- 2. Bound saturation ---
        if param_manager is not None:
            bound_result = self._check_bounds(result, param_manager)
            if bound_result.is_degenerate:
                issues.append(bound_result.message)
                affected.extend(bound_result.affected_params)
                actions.append(bound_result.suggested_action)

        # --- 3. Cost plateau ---
        plateau_result = self._check_plateau(result)
        if plateau_result.is_degenerate:
            issues.append(plateau_result.message)
            actions.append(plateau_result.suggested_action)

        if not issues:
            return DegeneracyCheck(is_degenerate=False, message="No degeneracy detected")

        # De-duplicate affected parameters
        seen: set[str] = set()
        unique_affected: list[str] = []
        for p in affected:
            if p not in seen:
                seen.add(p)
                unique_affected.append(p)

        combined_message = "; ".join(issues)
        combined_action = " ".join(dict.fromkeys(actions))  # de-dup, preserve order

        logger.warning("Degeneracy detected: %s", combined_message)

        return DegeneracyCheck(
            is_degenerate=True,
            affected_params=unique_affected,
            message=combined_message,
            suggested_action=combined_action,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_correlation(self, result: NLSQResult) -> DegeneracyCheck:
        """Check for highly correlated parameter pairs."""
        corr = result.get_correlation_matrix()
        if corr is None:
            return DegeneracyCheck(is_degenerate=False)

        names = result.parameter_names
        n = len(names)
        flagged_pairs: list[tuple[str, str, float]] = []

        for i in range(n):
            for j in range(i + 1, n):
                r = abs(corr[i, j])
                if r > self._corr_thresh:
                    flagged_pairs.append((names[i], names[j], float(corr[i, j])))

        if not flagged_pairs:
            return DegeneracyCheck(is_degenerate=False)

        affected: list[str] = []
        descriptions: list[str] = []
        for name_a, name_b, r_val in flagged_pairs:
            affected.extend([name_a, name_b])
            # Annotate known problematic pairs
            context = ""
            for pa, pb, desc in _KNOWN_DEGENERATE_PAIRS:
                if {pa, pb} == {name_a, name_b}:
                    context = f" ({desc})"
                    break
            descriptions.append(f"|r({name_a}, {name_b})| = {abs(r_val):.4f}{context}")

        message = "Correlated parameters: " + ", ".join(descriptions)
        action = "Consider fixing one parameter in each correlated pair."

        return DegeneracyCheck(
            is_degenerate=True,
            affected_params=affected,
            message=message,
            suggested_action=action,
        )

    def _check_bounds(
        self,
        result: NLSQResult,
        param_manager: ParameterManager,
    ) -> DegeneracyCheck:
        """Check for parameters sitting at their bounds."""
        lower, upper = param_manager.get_bounds()
        varying_names = param_manager.varying_names

        at_bound: list[str] = []
        for name, val, lo, hi in zip(
            varying_names, result.parameters, lower, upper, strict=True
        ):
            span = hi - lo
            if span <= 0:
                continue
            tol = self._bound_tol * span
            if val <= lo + tol:
                at_bound.append(f"{name} (at lower bound {lo:.3e})")
            elif val >= hi - tol:
                at_bound.append(f"{name} (at upper bound {hi:.3e})")

        if not at_bound:
            return DegeneracyCheck(is_degenerate=False)

        message = "Parameters at bounds: " + ", ".join(at_bound)
        action = "Widen bounds or fix the saturated parameter(s)."

        return DegeneracyCheck(
            is_degenerate=True,
            affected_params=[s.split(" ")[0] for s in at_bound],
            message=message,
            suggested_action=action,
        )

    def _check_plateau(self, result: NLSQResult) -> DegeneracyCheck:
        """Check for cost plateau (many iterations, negligible improvement)."""
        if result.n_iterations < self._plateau_min_iter:
            return DegeneracyCheck(is_degenerate=False)

        if result.final_cost is None:
            return DegeneracyCheck(is_degenerate=False)

        # Use metadata if available for initial cost
        initial_cost = result.metadata.get("initial_cost")
        if initial_cost is not None:
            denom = max(abs(float(initial_cost)), 1e-30)
            rel_change = abs(float(result.final_cost) - float(initial_cost)) / denom
            if rel_change < self._plateau_rtol:
                return DegeneracyCheck(
                    is_degenerate=True,
                    message=(
                        f"Cost plateau: relative change {rel_change:.2e} over "
                        f"{result.n_iterations} iterations"
                    ),
                    suggested_action="Try different initial parameters or a global optimizer.",
                )

        return DegeneracyCheck(is_degenerate=False)
