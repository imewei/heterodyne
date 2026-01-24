"""Configuration for NLSQ optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class NLSQConfig:
    """Configuration for NLSQ (Non-Linear Least Squares) fitting.
    
    Attributes:
        max_iterations: Maximum number of optimization iterations
        tolerance: Convergence tolerance for cost function
        method: Optimization method ('trf' or 'lm')
        multistart: Whether to use multi-start optimization
        multistart_n: Number of starting points for multi-start
        verbose: Verbosity level (0=silent, 1=summary, 2=detailed)
        use_jac: Whether to use analytic Jacobian
        x_scale: Parameter scaling ('jac' for Jacobian-based, or array)
        ftol: Function tolerance
        xtol: Parameter tolerance
        gtol: Gradient tolerance
        loss: Loss function ('linear', 'soft_l1', 'huber', 'cauchy')
    """
    
    max_iterations: int = 100
    tolerance: float = 1e-8
    method: Literal["trf", "lm", "dogbox"] = "trf"
    multistart: bool = False
    multistart_n: int = 10
    verbose: int = 1
    use_jac: bool = True
    x_scale: str | list[float] = "jac"
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    loss: Literal["linear", "soft_l1", "huber", "cauchy"] = "linear"
    
    # Advanced options
    diff_step: float | None = None
    max_nfev: int | None = None
    
    # Memory management
    chunk_size: int | None = None  # None for auto
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.multistart_n < 1:
            raise ValueError("multistart_n must be >= 1")
    
    @classmethod
    def from_dict(cls, config: dict) -> NLSQConfig:
        """Create from dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            NLSQConfig instance
        """
        # Filter to known fields
        known_fields = {
            "max_iterations", "tolerance", "method", "multistart",
            "multistart_n", "verbose", "use_jac", "x_scale",
            "ftol", "xtol", "gtol", "loss", "diff_step", "max_nfev",
            "chunk_size",
        }
        filtered = {k: v for k, v in config.items() if k in known_fields}
        return cls(**filtered)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "method": self.method,
            "multistart": self.multistart,
            "multistart_n": self.multistart_n,
            "verbose": self.verbose,
            "use_jac": self.use_jac,
            "x_scale": self.x_scale,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "gtol": self.gtol,
            "loss": self.loss,
            "diff_step": self.diff_step,
            "max_nfev": self.max_nfev,
            "chunk_size": self.chunk_size,
        }


@dataclass
class NLSQValidationConfig:
    """Configuration for NLSQ result validation."""
    
    # Reduced chi-squared thresholds
    chi2_warn_low: float = 0.5
    chi2_warn_high: float = 2.0
    chi2_fail_high: float = 10.0
    
    # Uncertainty validation
    max_relative_uncertainty: float = 1.0  # 100%
    
    # Correlation threshold for parameters
    correlation_warn: float = 0.95
