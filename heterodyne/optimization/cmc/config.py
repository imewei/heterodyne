"""Configuration for CMC (Consensus Monte Carlo) analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class CMCConfig:
    """Configuration for CMC Bayesian analysis.
    
    Attributes:
        enable: When to run CMC ('auto', 'always', 'never')
        num_warmup: Number of warmup/burn-in samples
        num_samples: Number of posterior samples
        num_chains: Number of MCMC chains
        target_accept: Target acceptance rate for NUTS
        max_tree_depth: Maximum tree depth for NUTS
        seed: Random seed for reproducibility
        dense_mass: Whether to use dense mass matrix
        init_strategy: Initialization strategy
    """
    
    enable: Literal["auto", "always", "never"] = "auto"
    num_warmup: int = 500
    num_samples: int = 1000
    num_chains: int = 4
    target_accept: float = 0.8
    max_tree_depth: int = 10
    seed: int | None = None
    dense_mass: bool = False
    init_strategy: Literal["init_to_median", "init_to_sample", "init_to_value"] = "init_to_median"
    
    # Convergence thresholds
    r_hat_threshold: float = 1.1
    min_ess: int = 100
    min_bfmi: float = 0.3
    
    # NLSQ warm-start
    use_nlsq_warmstart: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_warmup < 100:
            raise ValueError("num_warmup should be >= 100")
        if self.num_samples < 100:
            raise ValueError("num_samples should be >= 100")
        if self.num_chains < 1:
            raise ValueError("num_chains must be >= 1")
        if not (0.5 <= self.target_accept <= 0.99):
            raise ValueError("target_accept should be in [0.5, 0.99]")
    
    @classmethod
    def from_dict(cls, config: dict) -> CMCConfig:
        """Create from dictionary."""
        known_fields = {
            "enable", "num_warmup", "num_samples", "num_chains",
            "target_accept", "max_tree_depth", "seed", "dense_mass",
            "init_strategy", "r_hat_threshold", "min_ess", "min_bfmi",
            "use_nlsq_warmstart",
        }
        filtered = {k: v for k, v in config.items() if k in known_fields}
        return cls(**filtered)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enable": self.enable,
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "target_accept": self.target_accept,
            "max_tree_depth": self.max_tree_depth,
            "seed": self.seed,
            "dense_mass": self.dense_mass,
            "init_strategy": self.init_strategy,
            "r_hat_threshold": self.r_hat_threshold,
            "min_ess": self.min_ess,
            "min_bfmi": self.min_bfmi,
            "use_nlsq_warmstart": self.use_nlsq_warmstart,
        }
