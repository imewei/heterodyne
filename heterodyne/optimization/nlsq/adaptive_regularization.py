"""Adaptive Tikhonov regularization for ill-conditioned NLSQ problems.

The 14-parameter heterodyne model often produces near-singular Jacobians,
especially when velocity and fraction parameters are simultaneously varied.
This module provides Levenberg-Marquardt-style adaptive regularization
that stabilizes the Gauss-Newton step and covariance estimates.

Usage::

    reg = AdaptiveRegularizer(RegularizationConfig(lambda_init=1e-6))
    step = reg.compute_regularized_step(J, r, lambda_)
    lambda_ = reg.adapt_lambda(cost_new, cost_old, lambda_)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RegularizationConfig:
    """Configuration for adaptive Tikhonov regularization.

    Attributes:
        lambda_init: Initial regularization parameter.
        lambda_min: Minimum allowed lambda (lower bound prevents under-regularization).
        lambda_max: Maximum allowed lambda (upper bound prevents gradient descent fallback).
        adaptation_rate: Multiplicative factor for lambda updates. Lambda is multiplied
            by this factor on cost increase, and divided on cost decrease.
    """

    lambda_init: float = 1e-6
    lambda_min: float = 1e-12
    lambda_max: float = 1.0
    adaptation_rate: float = 2.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.lambda_init <= 0:
            raise ValueError(f"lambda_init must be positive, got {self.lambda_init}")
        if self.lambda_min <= 0:
            raise ValueError(f"lambda_min must be positive, got {self.lambda_min}")
        if self.lambda_max <= self.lambda_min:
            raise ValueError(
                f"lambda_max ({self.lambda_max}) must exceed lambda_min ({self.lambda_min})"
            )
        if self.adaptation_rate <= 1.0:
            raise ValueError(
                f"adaptation_rate must be > 1.0, got {self.adaptation_rate}"
            )


class AdaptiveRegularizer:
    """Adaptive Tikhonov regularizer for Gauss-Newton optimization.

    Implements the damped Gauss-Newton (Levenberg-Marquardt) update:

        (J^T J + lambda * I) * delta = -J^T r

    with adaptive lambda that increases when the cost grows (indicating
    the quadratic model is poor) and decreases when the cost shrinks.
    """

    def __init__(self, config: RegularizationConfig | None = None) -> None:
        """Initialize regularizer.

        Args:
            config: Regularization configuration, or None for defaults.
        """
        self._config = config or RegularizationConfig()
        self._lambda = self._config.lambda_init

    @property
    def current_lambda(self) -> float:
        """Current regularization parameter value."""
        return self._lambda

    def compute_regularized_step(
        self,
        jacobian: np.ndarray,
        residuals: np.ndarray,
        lambda_: float | None = None,
    ) -> np.ndarray:
        """Compute Tikhonov-regularized Gauss-Newton step.

        Solves: (J^T J + lambda * I) * delta = -J^T r

        Args:
            jacobian: Jacobian matrix, shape (n_residuals, n_params).
            residuals: Residual vector, shape (n_residuals,).
            lambda_: Regularization parameter override. If None, uses
                the internally tracked value.

        Returns:
            Parameter update step, shape (n_params,).
        """
        if lambda_ is None:
            lambda_ = self._lambda

        J = np.asarray(jacobian, dtype=np.float64)
        r = np.asarray(residuals, dtype=np.float64)

        n_params = J.shape[1]
        JtJ = J.T @ J
        Jtr = J.T @ r

        # Tikhonov regularization: add lambda * I to diagonal
        regularized = JtJ + lambda_ * np.eye(n_params, dtype=np.float64)

        try:
            step = np.linalg.solve(regularized, -Jtr)
        except np.linalg.LinAlgError:
            logger.warning(
                "Singular matrix at lambda=%.2e, increasing regularization",
                lambda_,
            )
            # Fall back to pseudoinverse
            step = -np.linalg.pinv(regularized) @ Jtr

        logger.debug(
            "Regularized step: lambda=%.2e, ||step||=%.3e, ||residual||=%.3e",
            lambda_,
            float(np.linalg.norm(step)),
            float(np.linalg.norm(r)),
        )

        return step

    def adapt_lambda(
        self,
        cost_new: float,
        cost_old: float,
        lambda_: float | None = None,
    ) -> float:
        """Adapt the regularization parameter based on cost change.

        - If cost_new < cost_old: decrease lambda (trust Gauss-Newton more)
        - If cost_new >= cost_old: increase lambda (regularize more)

        Args:
            cost_new: Cost after proposed step.
            cost_old: Cost before proposed step.
            lambda_: Current lambda. If None, uses internally tracked value.

        Returns:
            Updated lambda value (also stored internally).
        """
        if lambda_ is None:
            lambda_ = self._lambda

        rate = self._config.adaptation_rate

        if cost_new < cost_old:
            # Good step: decrease regularization
            new_lambda = max(lambda_ / rate, self._config.lambda_min)
            logger.debug(
                "Cost decreased (%.3e -> %.3e), lambda: %.2e -> %.2e",
                cost_old,
                cost_new,
                lambda_,
                new_lambda,
            )
        else:
            # Bad step: increase regularization
            new_lambda = min(lambda_ * rate, self._config.lambda_max)
            logger.debug(
                "Cost increased (%.3e -> %.3e), lambda: %.2e -> %.2e",
                cost_old,
                cost_new,
                lambda_,
                new_lambda,
            )

        self._lambda = new_lambda
        return new_lambda

    def regularize_covariance(
        self,
        covariance: np.ndarray,
        lambda_: float | None = None,
    ) -> np.ndarray:
        """Regularize a near-singular covariance matrix.

        Adds lambda * I to the covariance to ensure positive definiteness.
        This is useful for post-fit uncertainty estimation when the raw
        covariance from (J^T J)^{-1} is ill-conditioned.

        Args:
            covariance: Covariance matrix, shape (n, n).
            lambda_: Regularization parameter. If None, uses internally
                tracked value.

        Returns:
            Regularized covariance matrix, shape (n, n).
        """
        if lambda_ is None:
            lambda_ = self._lambda

        cov = np.asarray(covariance, dtype=np.float64)
        n = cov.shape[0]

        if cov.shape != (n, n):
            raise ValueError(f"Covariance must be square, got shape {cov.shape}")

        regularized = cov + lambda_ * np.eye(n, dtype=np.float64)

        # Verify positive-definiteness via Cholesky
        try:
            np.linalg.cholesky(regularized)
        except np.linalg.LinAlgError:
            logger.warning(
                "Covariance still not positive definite at lambda=%.2e, "
                "using larger regularization",
                lambda_,
            )
            # Increase lambda until positive-definite
            trial_lambda = lambda_ * self._config.adaptation_rate
            while trial_lambda <= self._config.lambda_max:
                regularized = cov + trial_lambda * np.eye(n, dtype=np.float64)
                try:
                    np.linalg.cholesky(regularized)
                    logger.debug(
                        "Covariance positive definite at lambda=%.2e",
                        trial_lambda,
                    )
                    break
                except np.linalg.LinAlgError:
                    trial_lambda *= self._config.adaptation_rate
            else:
                logger.warning(
                    "Could not regularize covariance up to lambda_max=%.2e",
                    self._config.lambda_max,
                )

        return regularized

    def reset(self) -> None:
        """Reset lambda to initial value."""
        self._lambda = self._config.lambda_init
