"""Post-fit quality assessment for NLSQ results.

This module provides the heterodyne-side parity for homodyne's fit-quality
validation surface:

- :func:`classify_fit_quality` — legacy 3-bin fit-quality classifier.
- :class:`FitQualityValidator` — accumulates :class:`ValidationIssue` entries
  on a :class:`ValidationReport` for the chi-squared + bounds-proximity checks
  used by the older heterodyne validator stack.
- :class:`FitQualityConfig` / :class:`FitQualityReport` / :func:`validate_fit_quality`
  — the richer homodyne-compatible API.  The validator runs five purely
  data-driven checks (chi-squared, parameter significance, covariance
  condition number, restart-metadata, convergence status) and an optional
  physical-bounds-hit check.  All checks append to
  :attr:`FitQualityReport.warnings`; :attr:`FitQualityReport.passed` is derived
  from ``len(warnings) == 0`` at the end of the run, so callers can rely on
  the invariant ``passed == (warnings == [])``.

Design notes
------------

* This module is a **pure** validator.  No ``logger.warning(...)`` calls live
  inside the checks themselves; callers (adapter, orchestrator) decide when to
  emit log lines based on the returned :class:`FitQualityReport`.

* The functional :func:`validate_fit_quality` entry point is a thin wrapper
  around :class:`FitQualityValidator`'s extended ``.validate_extended()``
  method, so the per-check logic lives in one place.

* The "is this a scaling parameter?" decision routes through
  :data:`heterodyne.config.parameter_registry.DEFAULT_REGISTRY` rather than
  string heuristics, picking up new scaling parameter names automatically.

* :attr:`FitQualityReport.checks_performed` is a tri-valued mapping: ``True``
  means the check ran and passed, ``False`` means the check ran and failed,
  and ``None`` means the check could not run (skipped because of malformed
  diagnostics, missing arrays, or singular matrices).  Silent skips always
  append a warning so the ``passed`` invariant continues to hold.

* Failure-token ownership is partitioned between the two convergence
  checkers: restart-class tokens (``max_restarts``, ``max_iter``,
  ``max_iterations``, SciPy's ``"maximum function evaluations reached"``) are
  the responsibility of :meth:`FitQualityValidator._check_restart_metadata`;
  hard-failure tokens (``failed``, ``diverged``) belong to
  :meth:`FitQualityValidator._check_convergence_status`.  No token is checked
  by both methods, so a single failing run never produces two overlapping
  warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.optimization.nlsq.validation.result import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scaling-parameter membership: single registry-backed source of truth
# ---------------------------------------------------------------------------

# Scaling parameter base names (derived from the immutable registry at import
# time).  ``ParameterInfo.is_scaling`` is the authoritative flag.
_SCALING_NAMES: frozenset[str] = frozenset(
    info.name for info in DEFAULT_REGISTRY._parameters.values() if info.is_scaling
)


def _is_scaling_param(name: str) -> bool:
    """Return ``True`` if ``name`` is a per-angle scaling parameter.

    Resolves the parameter against :data:`DEFAULT_REGISTRY` when possible and
    falls back to a label-pattern check (``contrast_0``, ``contrast[0]``,
    bare ``contrast``) so per-angle expansion labels are also classified
    correctly without forcing every caller to canonicalise its labels.
    """

    try:
        info = DEFAULT_REGISTRY[name]
    except KeyError:
        info = None

    if info is not None:
        return bool(info.is_scaling)

    return any(
        name.startswith(f"{s}_") or name.startswith(f"{s}[") or name == s
        for s in _SCALING_NAMES
    )


def _is_physical_param(name: str) -> bool:
    """Return ``True`` for physical (non-scaling) parameter labels."""

    return not _is_scaling_param(name)


# ---------------------------------------------------------------------------
# Result-attribute helpers — tolerate homodyne ``.popt`` / heterodyne names
# ---------------------------------------------------------------------------


def _get_params(result: Any) -> np.ndarray | None:
    """Return the parameter vector from ``result`` or ``None``.

    Tries ``result.parameters`` first (heterodyne :class:`NLSQResult`), then
    falls back to ``result.popt`` (scipy / homodyne).  Returns ``None`` if
    neither attribute is present, leaving the caller to skip the check.
    """

    params = getattr(result, "parameters", None)
    if params is None:
        params = getattr(result, "popt", None)
    return params


def _get_param_names(result: Any) -> list[str] | None:
    """Return the parameter-name list from ``result`` or ``None``.

    Tries ``result.parameter_names`` first, falls back to ``result.param_names``.
    Returns ``None`` if neither is present so callers can synthesise
    ``param[i]`` labels.
    """

    names = getattr(result, "parameter_names", None)
    if names is None:
        names = getattr(result, "param_names", None)
    return list(names) if names is not None else None


def _coerce_status_text(value: Any) -> str:
    """Coerce any ``convergence_status`` / ``convergence_reason`` value to a
    lower-case string, tolerating bool / int / enum / None inputs."""

    if value is None:
        return ""
    return str(value).lower()


def _coerce_bool(value: object) -> bool:
    """Coerce YAML-style truthy values to ``bool``.

    Plain ``bool(x)`` is wrong for YAML config slices: ``bool("false")``
    returns ``True`` because Python sees a non-empty string.  This helper
    treats the strings ``"0"``, ``"false"``, ``"no"``, ``"off"`` and the empty
    string (case-insensitive, whitespace-stripped) as ``False`` and falls
    back to :class:`bool` for everything else.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _skip_check(
    report: FitQualityReport,
    name: str,
    reason: str,
    *,
    warn: bool = True,
) -> None:
    """Record that a check could not run.

    Sets ``checks_performed[name] = None`` (the "skipped" sentinel) and, by
    default, appends a warning so the report's ``passed`` invariant remains
    consistent: a malformed diagnostic always shows up in
    :attr:`FitQualityReport.warnings`.
    """

    report.checks_performed[name] = None
    if warn:
        report.warnings.append(f"Quality check '{name}' could not run: {reason}.")


# ---------------------------------------------------------------------------
# Legacy classifier (kept for backwards compatibility)
# ---------------------------------------------------------------------------


def classify_fit_quality(
    reduced_chi_squared: float | None,
    n_at_bounds: int = 0,
) -> str:
    """Classify fit quality into a 3-level flag.

    Thresholds:

    - ``"good"``     — reduced chi-squared < 1.5 **and** no parameters at bounds
    - ``"marginal"`` — 1.5 <= reduced chi-squared < 3.0, or any parameter at a bound
    - ``"poor"``     — reduced chi-squared >= 3.0 or unavailable

    These thresholds also feed the defaults of :class:`FitQualityConfig`
    (``chi2_good_threshold`` / ``chi2_acceptable_threshold``) so the package
    has a single source of truth for what counts as a "good" vs "acceptable"
    chi-squared.
    """

    if reduced_chi_squared is None:
        return "poor"
    if reduced_chi_squared < 1.5:
        return "marginal" if n_at_bounds > 0 else "good"
    if reduced_chi_squared < 3.0:
        return "marginal"
    return "poor"


# ---------------------------------------------------------------------------
# Homodyne-parity API: FitQualityConfig + FitQualityReport + FitQualityValidator
# ---------------------------------------------------------------------------


@dataclass
class FitQualityConfig:
    """Configuration for fit quality validation.

    Mirrors ``homodyne.optimization.nlsq.validation.fit_quality.FitQualityConfig``
    so result-saving code can be ported between the two packages.

    Defaults follow the legacy 1.5/3.0 thresholds used by
    :func:`classify_fit_quality` to keep one source of truth across the
    package.
    """

    enable: bool = True
    reduced_chi_squared_threshold: float = 10.0
    chi2_good_threshold: float = 1.5
    chi2_acceptable_threshold: float = 3.0
    min_parameter_significance: float = 2.0
    max_condition_number: float = 1e12
    warn_on_max_restarts: bool = True
    warn_on_bounds_hit: bool = True
    warn_on_convergence_failure: bool = True
    bounds_tolerance: float = 1e-9

    @classmethod
    def from_validation_config(
        cls, validation_config: dict[str, Any] | None
    ) -> FitQualityConfig:
        """Round-trip every :class:`FitQualityConfig` field from a dict.

        Unknown keys are ignored and missing keys fall back to the dataclass
        defaults, so the method is safe to feed with raw YAML config slices.
        Boolean fields go through :func:`_coerce_bool` so YAML strings like
        ``"false"`` round-trip to ``False`` instead of being silently
        truthified.
        """

        if validation_config is None:
            return cls()

        defaults = cls()
        return cls(
            enable=_coerce_bool(validation_config.get("enable", defaults.enable)),
            reduced_chi_squared_threshold=float(
                validation_config.get(
                    "reduced_chi_squared_threshold",
                    defaults.reduced_chi_squared_threshold,
                )
            ),
            chi2_good_threshold=float(
                validation_config.get(
                    "chi2_good_threshold", defaults.chi2_good_threshold
                )
            ),
            chi2_acceptable_threshold=float(
                validation_config.get(
                    "chi2_acceptable_threshold", defaults.chi2_acceptable_threshold
                )
            ),
            min_parameter_significance=float(
                validation_config.get(
                    "min_parameter_significance", defaults.min_parameter_significance
                )
            ),
            max_condition_number=float(
                validation_config.get(
                    "max_condition_number", defaults.max_condition_number
                )
            ),
            warn_on_max_restarts=_coerce_bool(
                validation_config.get(
                    "warn_on_max_restarts", defaults.warn_on_max_restarts
                )
            ),
            warn_on_bounds_hit=_coerce_bool(
                validation_config.get("warn_on_bounds_hit", defaults.warn_on_bounds_hit)
            ),
            warn_on_convergence_failure=_coerce_bool(
                validation_config.get(
                    "warn_on_convergence_failure",
                    defaults.warn_on_convergence_failure,
                )
            ),
            bounds_tolerance=float(
                validation_config.get("bounds_tolerance", defaults.bounds_tolerance)
            ),
        )


@dataclass
class FitQualityReport:
    """Report from :func:`validate_fit_quality`.

    The :attr:`passed` flag is recomputed from :attr:`warnings` at the end of
    a validation run, so the invariant ``passed == (warnings == [])`` always
    holds.

    :attr:`checks_performed` maps each check name to one of:

    - ``True``  — the check ran and passed
    - ``False`` — the check ran and failed
    - ``None``  — the check could not run (skipped)
    """

    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    checks_performed: dict[str, bool | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dictionary for persistence."""

        return {
            "quality_validation_passed": self.passed,
            "quality_warnings": list(self.warnings),
            "quality_checks": dict(self.checks_performed),
        }


def _classify_parameter_status(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    atol: float = 1e-9,
) -> list[str]:
    """Classify each parameter as ``at_lower_bound`` / ``at_upper_bound`` /
    ``active`` relative to its bounds."""

    statuses: list[str] = []
    for val, lb, ub in zip(values, lower, upper, strict=False):
        if abs(val - lb) < atol * (1.0 + abs(lb)):
            statuses.append("at_lower_bound")
        elif abs(val - ub) < atol * (1.0 + abs(ub)):
            statuses.append("at_upper_bound")
        else:
            statuses.append("active")
    return statuses


# Failure-token ownership is partitioned: restart-class tokens go to
# ``_check_restart_metadata``, hard-failure tokens go to
# ``_check_convergence_status``.  These two sets are intentionally disjoint so
# a single failing convergence_reason cannot trigger two warnings.
_RESTART_TOKENS: frozenset[str] = frozenset(
    {
        "max_restarts",
        "max_iter",
        "max_iterations",
        "maximum function evaluations reached",
    }
)

_HARD_FAIL_TOKENS: frozenset[str] = frozenset(
    {
        "failed",
        "diverged",
    }
)


class FitQualityValidator:
    """Assess fit quality.

    Two surfaces live on this class:

    * :meth:`validate` — the legacy heterodyne report-style API that yields a
      :class:`ValidationReport` populated with :class:`ValidationIssue` entries
      for the chi-squared + bounds-proximity checks.

    * :meth:`validate_extended` — the homodyne-parity report-style API used by
      :func:`validate_fit_quality`.  Accepts a :class:`FitQualityConfig`
      (defaults applied if omitted) and an optional bounds tuple; returns a
      :class:`FitQualityReport`.

    The class is the single source of truth; :func:`validate_fit_quality` is a
    thin functional wrapper around ``validate_extended``.
    """

    def __init__(
        self,
        chi2_warn: float = 10.0,
        chi2_fail: float = 100.0,
        edge_fraction: float = 0.005,
    ) -> None:
        self._chi2_warn = chi2_warn
        self._chi2_fail = chi2_fail
        self._edge_fraction = edge_fraction

    # ------------------------------------------------------------------
    # Legacy ValidationReport surface (kept for back-compat)
    # ------------------------------------------------------------------

    def validate(self, result: NLSQResult) -> ValidationReport:
        report = ValidationReport()
        self._legacy_check_chi_squared(result, report)
        self._legacy_check_bounds_proximity(result, report)

        if report.errors:
            report.is_valid = False
        return report

    def _legacy_check_chi_squared(
        self,
        result: NLSQResult,
        report: ValidationReport,
    ) -> None:
        chi2 = result.reduced_chi_squared
        if chi2 is None:
            return

        if chi2 > self._chi2_fail:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Very poor fit: reduced chi2 = {chi2:.2f} > {self._chi2_fail}",
                    "chi2_red",
                    chi2,
                )
            )
        elif chi2 > self._chi2_warn:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Mediocre fit: reduced chi2 = {chi2:.2f} > {self._chi2_warn}",
                    "chi2_red",
                    chi2,
                )
            )

    def _legacy_check_bounds_proximity(
        self,
        result: NLSQResult,
        report: ValidationReport,
    ) -> None:
        for name, value in zip(
            result.parameter_names,
            result.parameters,
            strict=True,
        ):
            if not _is_physical_param(name):
                continue
            try:
                info = DEFAULT_REGISTRY[name]
            except KeyError:
                continue

            span = info.max_bound - info.min_bound
            if span <= 0:
                continue

            frac_lo = (value - info.min_bound) / span
            frac_hi = (info.max_bound - value) / span

            if frac_lo < self._edge_fraction:
                report.issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        f"{name} = {value:.4e} near lower bound "
                        f"({frac_lo * 100:.2f}% from min={info.min_bound})",
                        f"bound_edge_{name}",
                        frac_lo,
                    )
                )
            elif frac_hi < self._edge_fraction:
                report.issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        f"{name} = {value:.4e} near upper bound "
                        f"({frac_hi * 100:.2f}% from max={info.max_bound})",
                        f"bound_edge_{name}",
                        frac_hi,
                    )
                )

    # ------------------------------------------------------------------
    # Homodyne-parity FitQualityReport surface
    # ------------------------------------------------------------------

    def validate_extended(
        self,
        result: Any,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        config: FitQualityConfig | None = None,
        param_labels: list[str] | None = None,
    ) -> FitQualityReport:
        """Run the homodyne-parity quality checks against ``result``."""

        if config is None:
            config = FitQualityConfig()

        if not config.enable:
            return FitQualityReport(passed=True, checks_performed={"enabled": False})

        report = FitQualityReport()

        self._check_chi_squared(result, config, report)
        self._check_parameter_significance(result, config, report, param_labels)
        self._check_condition_number(result, config, report)
        self._check_restart_metadata(result, config, report)
        if config.warn_on_bounds_hit and bounds is not None:
            self._check_physical_bounds(result, bounds, config, param_labels, report)
        self._check_convergence_status(result, config, report)

        # ``passed`` is derived; never set inline.  This guarantees the
        # invariant ``passed == (warnings == [])`` at the end of the run.
        report.passed = len(report.warnings) == 0
        return report

    # -- individual checks ---------------------------------------------

    @staticmethod
    def _check_chi_squared(
        result: Any,
        config: FitQualityConfig,
        report: FitQualityReport,
    ) -> None:
        """Reduced-chi² hard threshold + 3-bin quality classification.

        The hard-threshold and quality-bin paths are mutually exclusive: when
        ``reduced_chi²`` exceeds ``chi2_acceptable_threshold`` we fire exactly
        one warning ("threshold exceeded"), not two; the quality classifier
        only records ``chi2_quality=False`` in that case without appending a
        second redundant message.
        """

        reduced_chi_squared = getattr(result, "reduced_chi_squared", None)
        if reduced_chi_squared is None:
            _skip_check(
                report,
                "reduced_chi_squared",
                "result has no reduced_chi_squared attribute",
            )
            return

        threshold_failed = reduced_chi_squared > config.reduced_chi_squared_threshold
        report.checks_performed["reduced_chi_squared"] = not threshold_failed
        if threshold_failed:
            sigma_is_default = getattr(result, "sigma_is_default", False)
            if sigma_is_default:
                report.warnings.append(
                    f"Reduced chi-squared ({reduced_chi_squared:.4g}) exceeds threshold "
                    f"({config.reduced_chi_squared_threshold}), but sigma was not provided "
                    f"(using default). Chi-squared is not physically meaningful without "
                    f"experimental uncertainties; inspect residuals directly."
                )
            else:
                report.warnings.append(
                    f"Reduced chi-squared ({reduced_chi_squared:.4g}) exceeds threshold "
                    f"({config.reduced_chi_squared_threshold}). Consider reviewing fit quality."
                )

        # 3-bin quality classification, strictly below the acceptable
        # threshold = "acceptable" (matches the legacy ``classify_fit_quality``
        # convention that 3.0 is poor, not marginal).  Avoid double-counting:
        # when the hard-threshold path already fired a warning, we only flip
        # ``chi2_quality`` to False without appending another message.
        if reduced_chi_squared <= config.chi2_good_threshold:
            report.checks_performed["chi2_quality"] = True
        elif reduced_chi_squared < config.chi2_acceptable_threshold:
            report.checks_performed["chi2_quality"] = True
        else:
            report.checks_performed["chi2_quality"] = False
            if not threshold_failed:
                report.warnings.append(
                    f"Chi-squared quality: poor "
                    f"(reduced chi2 = {reduced_chi_squared:.4g} >= "
                    f"acceptable threshold = {config.chi2_acceptable_threshold:.4g})."
                )

    @staticmethod
    def _check_parameter_significance(
        result: Any,
        config: FitQualityConfig,
        report: FitQualityReport,
        param_labels: list[str] | None = None,
    ) -> None:
        """|param| / uncertainty ratio with NaN-safe masking.

        A parameter is treated as significant only when both its value and
        its uncertainty are finite *and* the uncertainty is strictly
        positive.  NaN parameters with finite uncertainty no longer slip
        through (the comparison ``nan < threshold`` would have evaluated to
        ``False`` and silently marked the parameter as significant).
        """

        params = _get_params(result)
        uncertainties = getattr(result, "uncertainties", None)
        if params is None or uncertainties is None:
            return  # both fields optional; nothing to check, nothing to skip

        try:
            params_arr = np.asarray(params, dtype=np.float64)
            uncert_arr = np.asarray(uncertainties, dtype=np.float64)
        except (TypeError, ValueError):
            _skip_check(
                report,
                "parameter_significance",
                "non-numeric parameters or uncertainties",
            )
            return

        if params_arr.shape != uncert_arr.shape or params_arr.size == 0:
            _skip_check(
                report,
                "parameter_significance",
                "empty or shape-mismatched parameters / uncertainties",
            )
            return

        valid = np.isfinite(params_arr) & np.isfinite(uncert_arr) & (uncert_arr > 0)
        if not np.any(valid):
            _skip_check(
                report,
                "parameter_significance",
                "no parameters have finite values and positive finite uncertainty",
            )
            return

        names = param_labels
        if names is None:
            names = _get_param_names(result)
        if names is None:
            names = [f"param[{i}]" for i in range(params_arr.size)]
        elif len(names) < params_arr.size:
            names = list(names) + [
                f"param[{i}]" for i in range(len(names), params_arr.size)
            ]

        # Warn explicitly about parameters with non-finite values that had to be
        # excluded from the significance comparison.  Without this, a NaN
        # parameter alongside otherwise-significant parameters would slip
        # through silently because the mask drops it before the comparison.
        nonfinite_param_mask = ~np.isfinite(params_arr)
        if np.any(nonfinite_param_mask):
            excluded_names = [
                names[i]
                for i in np.where(nonfinite_param_mask)[0].tolist()
                if i < len(names)
            ]
            n_excluded = int(np.sum(nonfinite_param_mask))
            joined_excl = ", ".join(excluded_names) if excluded_names else "(unnamed)"
            report.warnings.append(
                f"{n_excluded} parameter(s) have non-finite values and were excluded "
                f"from significance check: {joined_excl}."
            )

        significance = np.full(params_arr.shape, np.inf, dtype=np.float64)
        significance[valid] = np.abs(params_arr[valid]) / uncert_arr[valid]
        insignificant_mask = (significance < config.min_parameter_significance) & valid

        if np.any(insignificant_mask):
            failing_indices = np.where(insignificant_mask)[0]
            failing_names = [
                names[i] for i in failing_indices.tolist() if i < len(names)
            ]
            report.checks_performed["parameter_significance"] = False
            joined = ", ".join(failing_names) if failing_names else "(unnamed)"
            report.warnings.append(
                f"{int(np.sum(insignificant_mask))} parameter(s) below significance threshold "
                f"(|param/uncertainty| < {config.min_parameter_significance}): {joined}. "
                "These parameters may be poorly constrained."
            )
        else:
            report.checks_performed["parameter_significance"] = True

    @staticmethod
    def _check_condition_number(
        result: Any,
        config: FitQualityConfig,
        report: FitQualityReport,
    ) -> None:
        """Covariance-matrix condition number.

        Malformed covariance matrices (non-square, non-numeric, singular,
        non-finite condition number) are recorded with the "skipped" sentinel
        plus a warning, so callers can distinguish "no covariance available"
        from "check passed".
        """

        pcov = getattr(result, "covariance", None)
        if pcov is None:
            pcov = getattr(result, "pcov", None)
        if pcov is None:
            return  # field genuinely absent; nothing to skip

        try:
            pcov_arr = np.asarray(pcov, dtype=np.float64)
        except (TypeError, ValueError):
            _skip_check(report, "condition_number", "non-numeric covariance matrix")
            return

        if pcov_arr.ndim != 2 or pcov_arr.shape[0] != pcov_arr.shape[1]:
            _skip_check(report, "condition_number", "non-square covariance matrix")
            return

        if not np.all(np.isfinite(pcov_arr)):
            _skip_check(
                report,
                "condition_number",
                "non-finite entries in covariance matrix",
            )
            return

        try:
            cond = float(np.linalg.cond(pcov_arr))
        except (np.linalg.LinAlgError, ValueError):
            _skip_check(
                report,
                "condition_number",
                "singular or non-numeric covariance matrix",
            )
            return

        if not np.isfinite(cond):
            _skip_check(
                report,
                "condition_number",
                "singular/non-finite covariance matrix",
            )
            return

        if cond > config.max_condition_number:
            report.checks_performed["condition_number"] = False
            report.warnings.append(
                f"Covariance matrix condition number ({cond:.2e}) exceeds "
                f"threshold ({config.max_condition_number:.2e}). "
                "Parameters may be highly correlated or poorly determined."
            )
        else:
            report.checks_performed["condition_number"] = True

    @staticmethod
    def _check_restart_metadata(
        result: Any,
        config: FitQualityConfig,
        report: FitQualityReport,
    ) -> None:
        """Optimizer-restart / max-iteration check.

        Owns the restart-class tokens in :data:`_RESTART_TOKENS` (``max_restarts``,
        ``max_iter``, ``max_iterations``, SciPy's ``"maximum function
        evaluations reached"``).  Hard-failure tokens (``failed`` / ``diverged``)
        are deliberately *not* checked here — they belong to
        :meth:`_check_convergence_status`, so the two checkers never both fire
        on the same status string.

        The check runs for any optimizer (LM, TRF, NLSQ, CMA-ES) — the warning
        is phrased generically rather than naming CMA-ES.
        """

        if not config.warn_on_max_restarts:
            return

        # Primary source: ``result.convergence_reason`` (heterodyne).  Falls
        # back to ``metadata["cmaes_convergence_reason"]`` and then to
        # ``device_info["convergence_reason"]`` for cross-package
        # compatibility with homodyne result schemas.
        convergence_reason = _coerce_status_text(
            getattr(result, "convergence_reason", None)
        )

        if not convergence_reason:
            metadata = getattr(result, "metadata", None) or {}
            if isinstance(metadata, dict):
                convergence_reason = _coerce_status_text(
                    metadata.get("cmaes_convergence_reason", "")
                )

        if not convergence_reason:
            device_info = getattr(result, "device_info", None) or {}
            if isinstance(device_info, dict):
                convergence_reason = _coerce_status_text(
                    device_info.get("convergence_reason", "")
                )

        if not convergence_reason:
            return

        if convergence_reason in _RESTART_TOKENS:
            report.checks_performed["restart_metadata"] = False
            report.warnings.append(
                "Optimizer reached maximum restarts / iterations without "
                "convergence. Consider increasing the iteration limit or "
                "adjusting initial parameters."
            )
        else:
            report.checks_performed["restart_metadata"] = True

    @staticmethod
    def _check_physical_bounds(
        result: Any,
        bounds: tuple[np.ndarray, np.ndarray],
        config: FitQualityConfig,
        param_labels: list[str] | None,
        report: FitQualityReport,
    ) -> None:
        params = _get_params(result)
        if params is None:
            return  # genuinely absent

        try:
            params_arr = np.asarray(params, dtype=np.float64)
            lower_arr = np.asarray(bounds[0], dtype=np.float64)
            upper_arr = np.asarray(bounds[1], dtype=np.float64)
        except (TypeError, ValueError):
            _skip_check(report, "physical_bounds", "non-numeric params or bounds")
            return

        if params_arr.size == 0:
            _skip_check(report, "physical_bounds", "empty parameter vector")
            return

        if not (params_arr.shape == lower_arr.shape == upper_arr.shape):
            _skip_check(
                report,
                "physical_bounds",
                "shape mismatch between params and bounds",
            )
            return

        statuses = _classify_parameter_status(
            params_arr, lower_arr, upper_arr, config.bounds_tolerance
        )

        labels = param_labels
        if labels is None:
            labels = _get_param_names(result)

        at_bounds: list[tuple[str, str]] = []
        for i, status in enumerate(statuses):
            if status not in ("at_lower_bound", "at_upper_bound"):
                continue
            label = (
                labels[i] if labels is not None and i < len(labels) else f"param[{i}]"
            )
            if not _is_physical_param(label):
                continue
            at_bounds.append((label, status))

        report.checks_performed["physical_bounds"] = len(at_bounds) == 0
        if at_bounds:
            params_str = ", ".join(f"{label} ({status})" for label, status in at_bounds)
            report.warnings.append(
                f"Physical parameters at bounds: {params_str}. "
                "Consider expanding bounds or reviewing initial parameters."
            )

    @staticmethod
    def _check_convergence_status(
        result: Any,
        config: FitQualityConfig,
        report: FitQualityReport,
    ) -> None:
        """Convergence-status hard-failure check.

        Owns only :data:`_HARD_FAIL_TOKENS` (``failed`` / ``diverged``).
        Restart-class tokens are checked by :meth:`_check_restart_metadata`,
        so the same status string never produces two warnings.

        Reads the status from a 3-tier fallback chain identical in spirit to
        :meth:`_check_restart_metadata`: ``result.convergence_status`` →
        ``result.convergence_reason`` → ``metadata.get("convergence_status")``
        → ``metadata.get("convergence_reason")`` →
        ``device_info.get("convergence_status")`` →
        ``device_info.get("convergence_reason")``.
        """

        if not config.warn_on_convergence_failure:
            return

        status_text = _coerce_status_text(
            getattr(result, "convergence_status", None)
        ) or _coerce_status_text(getattr(result, "convergence_reason", None))

        if not status_text:
            metadata = getattr(result, "metadata", None) or {}
            if isinstance(metadata, dict):
                status_text = _coerce_status_text(
                    metadata.get("convergence_status", "")
                ) or _coerce_status_text(metadata.get("convergence_reason", ""))

        if not status_text:
            device_info = getattr(result, "device_info", None) or {}
            if isinstance(device_info, dict):
                status_text = _coerce_status_text(
                    device_info.get("convergence_status", "")
                ) or _coerce_status_text(device_info.get("convergence_reason", ""))

        if not status_text:
            return

        # Tokens that belong to the restart checker are ignored here so we
        # never fire two warnings for the same status string.
        if status_text in _RESTART_TOKENS:
            return

        passed = status_text not in _HARD_FAIL_TOKENS
        report.checks_performed["convergence_status"] = passed
        if not passed:
            report.warnings.append(
                f"Optimization did not converge successfully (status: {status_text})."
            )


# ---------------------------------------------------------------------------
# Functional entry point (thin wrapper around FitQualityValidator)
# ---------------------------------------------------------------------------


def validate_fit_quality(
    result: Any,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    config: FitQualityConfig | None = None,
    param_labels: list[str] | None = None,
) -> FitQualityReport:
    """Validate post-fit quality and return a :class:`FitQualityReport`.

    Thin convenience wrapper over
    :meth:`FitQualityValidator.validate_extended`; both yield identical
    results.  This function is the recommended entry point for callers that
    do not care about reusing a validator instance.
    """

    validator = FitQualityValidator()
    return validator.validate_extended(
        result=result,
        bounds=bounds,
        config=config,
        param_labels=param_labels,
    )


__all__ = [
    "classify_fit_quality",
    "FitQualityConfig",
    "FitQualityReport",
    "FitQualityValidator",
    "validate_fit_quality",
]
