"""Result container for CMC analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class CMCResult:
    """Result of CMC (Consensus Monte Carlo) analysis.
    
    Contains posterior samples, summaries, and convergence diagnostics.
    """

    # Core results
    parameter_names: list[str]
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    credible_intervals: dict[str, dict[str, float]]

    # Convergence diagnostics
    convergence_passed: bool
    r_hat: np.ndarray | None = None
    ess_bulk: np.ndarray | None = None
    ess_tail: np.ndarray | None = None
    bfmi: list[float] | None = None

    # Full posterior samples
    samples: dict[str, np.ndarray] | None = None

    # MAP estimate (maximum a posteriori)
    map_estimate: np.ndarray | None = None

    # Sampling info
    num_warmup: int = 0
    num_samples: int = 0
    num_chains: int = 0
    wall_time_seconds: float | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return len(self.parameter_names)

    def get_param_summary(self, name: str) -> dict[str, float]:
        """Get summary statistics for a parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Dict with mean, std, and credible interval bounds
        """
        try:
            idx = self.parameter_names.index(name)
        except ValueError:
            raise KeyError(f"Parameter '{name}' not found") from None

        summary = {
            "mean": float(self.posterior_mean[idx]),
            "std": float(self.posterior_std[idx]),
        }

        if name in self.credible_intervals:
            summary.update(self.credible_intervals[name])

        if self.r_hat is not None:
            summary["r_hat"] = float(self.r_hat[idx])

        if self.ess_bulk is not None:
            summary["ess_bulk"] = float(self.ess_bulk[idx])

        return summary

    def get_samples(self, name: str) -> np.ndarray | None:
        """Get posterior samples for a parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Array of samples or None if not stored
        """
        if self.samples is None:
            return None
        return self.samples.get(name)

    def params_dict(self) -> dict[str, float]:
        """Get posterior means as dictionary."""
        return {
            name: float(self.posterior_mean[i])
            for i, name in enumerate(self.parameter_names)
        }

    def validate_convergence(
        self,
        r_hat_threshold: float = 1.1,
        min_ess: int = 100,
        min_bfmi: float = 0.3,
    ) -> list[str]:
        """Validate convergence diagnostics.
        
        Args:
            r_hat_threshold: Maximum acceptable R-hat
            min_ess: Minimum effective sample size
            min_bfmi: Minimum BFMI value
            
        Returns:
            List of warning messages
        """
        warnings = []

        if self.r_hat is not None:
            bad_rhat = np.where(self.r_hat > r_hat_threshold)[0]
            for idx in bad_rhat:
                warnings.append(
                    f"R-hat for {self.parameter_names[idx]}: "
                    f"{self.r_hat[idx]:.3f} > {r_hat_threshold}"
                )

        if self.ess_bulk is not None:
            low_ess = np.where(self.ess_bulk < min_ess)[0]
            for idx in low_ess:
                warnings.append(
                    f"Low ESS for {self.parameter_names[idx]}: "
                    f"{self.ess_bulk[idx]:.0f} < {min_ess}"
                )

        if self.bfmi is not None:
            low_bfmi = [b for b in self.bfmi if b < min_bfmi]
            if low_bfmi:
                warnings.append(
                    f"Low BFMI: {min(low_bfmi):.3f} < {min_bfmi}"
                )

        return warnings

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "CMC Analysis Result",
            "=" * 60,
            f"Convergence: {'PASSED' if self.convergence_passed else 'FAILED'}",
            f"Chains: {self.num_chains} | Samples: {self.num_samples} | Warmup: {self.num_warmup}",
            "",
            "Posterior Summary:",
            "-" * 60,
            f"{'Parameter':18s} {'Mean':>12s} {'Std':>10s} {'R-hat':>8s} {'ESS':>8s}",
            "-" * 60,
        ]

        for i, name in enumerate(self.parameter_names):
            mean = self.posterior_mean[i]
            std = self.posterior_std[i]
            r_hat = self.r_hat[i] if self.r_hat is not None else np.nan
            ess = self.ess_bulk[i] if self.ess_bulk is not None else np.nan

            r_hat_str = f"{r_hat:.3f}" if not np.isnan(r_hat) else "N/A"
            ess_str = f"{ess:.0f}" if not np.isnan(ess) else "N/A"

            lines.append(f"{name:18s} {mean:12.4e} {std:10.2e} {r_hat_str:>8s} {ess_str:>8s}")

        lines.append("-" * 60)

        if self.wall_time_seconds is not None:
            lines.append(f"Wall time: {self.wall_time_seconds:.1f} s")

        return "\n".join(lines)
