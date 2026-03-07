"""Gradient quality monitoring for NLSQ optimization.

Tracks gradient statistics across iterations to detect pathological
behaviour such as vanishing or exploding gradients that would prevent
convergence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GradientSnapshot:
    """Single-iteration gradient statistics.

    Attributes:
        iteration: Optimizer iteration number.
        gradient_norm: L2 norm of the full gradient vector.
        max_gradient: Maximum absolute gradient component.
        parameter_gradients: Per-parameter gradient values (copy).
    """

    iteration: int
    gradient_norm: float
    max_gradient: float
    parameter_gradients: np.ndarray


class GradientMonitor:
    """Track gradient history and detect pathological behaviour.

    Usage::

        monitor = GradientMonitor(parameter_names=["D0_ref", "alpha_ref", ...])
        for it in range(max_iter):
            grad = compute_gradient(...)
            monitor.record(it, grad)
        if monitor.check_vanishing(threshold=1e-12):
            logger.warning("Vanishing gradients detected")
        print(monitor.get_summary())

    Parameters
    ----------
    parameter_names : list[str] | None
        Optional names for each gradient component, used in diagnostics.
    """

    def __init__(self, parameter_names: list[str] | None = None) -> None:
        self._parameter_names = parameter_names
        self._history: list[GradientSnapshot] = []

    @property
    def history(self) -> list[GradientSnapshot]:
        """Full gradient history."""
        return list(self._history)

    @property
    def n_records(self) -> int:
        """Number of recorded snapshots."""
        return len(self._history)

    def record(self, iteration: int, gradients: np.ndarray) -> None:
        """Record gradient vector for one iteration.

        Args:
            iteration: Current iteration number.
            gradients: Gradient vector, shape ``(n_params,)``.
        """
        gradients = np.asarray(gradients, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(gradients))
        max_abs = float(np.max(np.abs(gradients))) if gradients.size > 0 else 0.0

        snapshot = GradientSnapshot(
            iteration=iteration,
            gradient_norm=norm,
            max_gradient=max_abs,
            parameter_gradients=gradients.copy(),
        )
        self._history.append(snapshot)

        if np.isnan(norm) or np.isinf(norm):
            logger.warning(
                "Gradient contains NaN/Inf at iteration %d (norm=%.3e)",
                iteration,
                norm,
            )

    def check_vanishing(self, threshold: float = 1e-12) -> bool:
        """Check whether recent gradients have vanished.

        A vanishing gradient indicates the optimizer is stuck in a flat
        region or at a saddle point.

        Args:
            threshold: Gradient norm below which is considered vanishing.

        Returns:
            ``True`` if the last recorded gradient norm is below *threshold*.
        """
        if not self._history:
            return False
        return self._history[-1].gradient_norm < threshold

    def check_exploding(self, threshold: float = 1e10) -> bool:
        """Check whether recent gradients have exploded.

        Exploding gradients can cause divergence and numerical overflow.

        Args:
            threshold: Gradient norm above which is considered exploding.

        Returns:
            ``True`` if the last recorded gradient norm exceeds *threshold*.
        """
        if not self._history:
            return False
        last_norm = self._history[-1].gradient_norm
        return last_norm > threshold or np.isnan(last_norm) or np.isinf(last_norm)

    def get_summary(self) -> dict[str, object]:
        """Summarize gradient history.

        Returns:
            Dictionary with keys:
            - ``n_iterations``: Number of recorded iterations.
            - ``final_norm``: Last gradient norm.
            - ``min_norm``: Minimum gradient norm across history.
            - ``max_norm``: Maximum gradient norm across history.
            - ``mean_norm``: Mean gradient norm.
            - ``is_vanishing``: Whether the final gradient is vanishing.
            - ``is_exploding``: Whether the final gradient is exploding.
            - ``worst_parameters``: Names of parameters with largest final
              gradient components (if names were provided).
        """
        if not self._history:
            return {
                "n_iterations": 0,
                "final_norm": None,
                "min_norm": None,
                "max_norm": None,
                "mean_norm": None,
                "is_vanishing": False,
                "is_exploding": False,
                "worst_parameters": [],
            }

        norms = np.array([s.gradient_norm for s in self._history])
        final = self._history[-1]

        # Identify parameters with largest gradient components
        worst_params: list[str] = []
        if self._parameter_names is not None and final.parameter_gradients.size > 0:
            abs_grad = np.abs(final.parameter_gradients)
            n_show = min(3, len(self._parameter_names))
            top_indices = np.argsort(abs_grad)[::-1][:n_show]
            worst_params = [
                f"{self._parameter_names[i]} ({abs_grad[i]:.3e})"
                for i in top_indices
                if i < len(self._parameter_names)
            ]

        return {
            "n_iterations": len(self._history),
            "final_norm": float(final.gradient_norm),
            "min_norm": float(np.nanmin(norms)),
            "max_norm": float(np.nanmax(norms)),
            "mean_norm": float(np.nanmean(norms)),
            "is_vanishing": self.check_vanishing(),
            "is_exploding": self.check_exploding(),
            "worst_parameters": worst_params,
        }
