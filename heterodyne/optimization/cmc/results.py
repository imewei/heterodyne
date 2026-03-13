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
                warnings.append(f"Low BFMI: {min(low_bfmi):.3f} < {min_bfmi}")

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

            lines.append(
                f"{name:18s} {mean:12.4e} {std:10.2e} {r_hat_str:>8s} {ess_str:>8s}"
            )

        lines.append("-" * 60)

        if self.wall_time_seconds is not None:
            lines.append(f"Wall time: {self.wall_time_seconds:.1f} s")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone functions operating on CMCResult
# ---------------------------------------------------------------------------


def cmc_result_to_arviz(result: CMCResult) -> Any:
    """Convert a CMCResult to an ArviZ InferenceData object.

    Samples stored in ``result.samples`` are reshaped to
    ``(num_chains, num_draws)`` when ``result.num_chains > 1`` so that
    ArviZ can compute per-chain diagnostics (R-hat, ESS).  When the
    result carries flat 1-D arrays the function treats the entire
    sequence as a single chain.

    Args:
        result: Completed CMC analysis result.

    Returns:
        ``arviz.InferenceData`` with a ``posterior`` group populated from
        ``result.samples`` and, when available, ``sample_stats`` populated
        from ``result.bfmi``.

    Raises:
        ImportError: If ArviZ is not installed.
        ValueError: If ``result.samples`` is None or empty.
    """
    try:
        import arviz as az  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "ArviZ is required for cmc_result_to_arviz. Install it with: uv add arviz"
        ) from None

    if not result.samples:
        raise ValueError(
            "CMCResult.samples is None or empty; cannot build InferenceData."
        )

    n_chains = max(result.num_chains, 1)
    posterior_dict: dict[str, np.ndarray] = {}

    for name, arr in result.samples.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            total = arr.shape[0]
            n_draws = total // n_chains
            if n_chains > 1 and total % n_chains == 0:
                # Reshape to (chains, draws)
                posterior_dict[name] = arr.reshape(n_chains, n_draws)
            else:
                # Fall back to single chain
                posterior_dict[name] = arr[np.newaxis, :]
        elif arr.ndim == 2:
            # Already (chains, draws)
            posterior_dict[name] = arr
        else:
            # Higher-dimensional parameter (e.g. covariance matrix per draw)
            posterior_dict[name] = arr

    idata = az.from_dict(posterior=posterior_dict)

    if result.bfmi is not None:
        sample_stats = {"energy": np.array(result.bfmi)}
        idata.add_groups({"sample_stats": sample_stats})

    return idata


def compare_cmc_nlsq(
    cmc_result: CMCResult,
    nlsq_result: Any,
    consistency_sigma: float = 2.0,
) -> dict[str, Any]:
    """Compare CMC posterior means with NLSQ point estimates.

    Parameters that appear in both results are compared. Parameters
    present in only one result are silently skipped.

    Args:
        cmc_result: Completed CMC result.
        nlsq_result: Completed NLSQ result (``NLSQResult`` instance).
        consistency_sigma: Number of posterior standard deviations within
            which the NLSQ estimate must fall to be flagged as consistent.
            Defaults to 2.0 (approximately 95 % credible interval).

    Returns:
        Dictionary with keys:

        - ``"common_parameters"`` — list of parameter names present in both.
        - ``"differences"`` — dict mapping name to ``(cmc_mean - nlsq_value)``.
        - ``"relative_deviations"`` — dict mapping name to
          ``abs(cmc_mean - nlsq_value) / cmc_std``.
        - ``"consistent"`` — dict mapping name to bool (True if within
          ``consistency_sigma`` posterior std of the CMC mean).
        - ``"n_consistent"`` — int count of consistent parameters.
        - ``"n_inconsistent"`` — int count of inconsistent parameters.
        - ``"consistency_sigma"`` — the threshold used.
    """
    cmc_means = cmc_result.params_dict()
    # NLSQResult.params_dict is a property, not a method
    nlsq_params: dict[str, float] = nlsq_result.params_dict

    common = sorted(set(cmc_means) & set(nlsq_params))

    differences: dict[str, float] = {}
    relative_deviations: dict[str, float] = {}
    consistent: dict[str, bool] = {}

    for name, cmc_val, nlsq_val, cmc_std in zip(
        common,
        [cmc_means[n] for n in common],
        [nlsq_params[n] for n in common],
        [
            float(cmc_result.posterior_std[cmc_result.parameter_names.index(n)])
            for n in common
        ],
        strict=True,
    ):
        diff = cmc_val - nlsq_val
        differences[name] = diff
        if cmc_std > 0.0:
            rel_dev = abs(diff) / cmc_std
        else:
            rel_dev = float("inf") if diff != 0.0 else 0.0
        relative_deviations[name] = rel_dev
        consistent[name] = rel_dev <= consistency_sigma

    n_consistent = sum(1 for v in consistent.values() if v)
    n_inconsistent = len(consistent) - n_consistent

    return {
        "common_parameters": common,
        "differences": differences,
        "relative_deviations": relative_deviations,
        "consistent": consistent,
        "n_consistent": n_consistent,
        "n_inconsistent": n_inconsistent,
        "consistency_sigma": consistency_sigma,
    }


def merge_shard_cmc_results(
    shard_results: list[CMCResult],
    parameter_names: list[str] | None = None,
) -> CMCResult:
    """Combine multiple shard CMCResults into a single consensus result.

    Uses inverse-variance weighting (precision weighting) to combine
    posterior means from independent shards, following the Consensus
    Monte Carlo methodology (Scott et al., 2016).  Diagnostics
    (R-hat, ESS, BFMI) are set to their worst-case values across shards
    so that failures are never hidden by averaging.

    Args:
        shard_results: Non-empty list of per-shard CMCResults.  Each must
            have the same ``parameter_names`` (or ``parameter_names``
            override must be supplied).
        parameter_names: Optional explicit parameter name list.  When
            supplied, only these parameters are included in the merged
            result; they must be present in every shard.

    Returns:
        A new ``CMCResult`` representing the consensus posterior.

    Raises:
        ValueError: If ``shard_results`` is empty or parameter names are
            inconsistent across shards when no override is given.
    """
    if not shard_results:
        raise ValueError("shard_results must be non-empty.")

    # Determine canonical parameter names
    if parameter_names is None:
        parameter_names = shard_results[0].parameter_names
        for i, sr in enumerate(shard_results[1:], start=1):
            if sr.parameter_names != parameter_names:
                raise ValueError(
                    f"Shard {i} has parameter_names {sr.parameter_names!r} "
                    f"but shard 0 has {parameter_names!r}. "
                    "Pass parameter_names explicitly to override."
                )

    n_params = len(parameter_names)

    # --- Inverse-variance weighted combination ---
    # precision_i = 1 / std_i^2  (per parameter)
    # combined_mean = sum(precision_i * mean_i) / sum(precision_i)
    # combined_std  = 1 / sqrt(sum(precision_i))
    precision_sum = np.zeros(n_params, dtype=np.float64)
    weighted_mean_sum = np.zeros(n_params, dtype=np.float64)

    for sr in shard_results:
        std = np.asarray(sr.posterior_std, dtype=np.float64)
        # Guard against zero std (degenerate shards)
        std = np.where(std > 0.0, std, np.finfo(np.float64).tiny)
        precision = 1.0 / (std**2)
        precision_sum += precision
        mean = np.asarray(sr.posterior_mean, dtype=np.float64)
        weighted_mean_sum += precision * mean

    combined_mean = weighted_mean_sum / precision_sum
    combined_std = 1.0 / np.sqrt(precision_sum)

    # --- Worst-case diagnostics ---
    r_hat_arrays = [sr.r_hat for sr in shard_results if sr.r_hat is not None]
    combined_r_hat: np.ndarray | None = None
    if r_hat_arrays:
        combined_r_hat = np.max(np.stack(r_hat_arrays, axis=0), axis=0)

    ess_bulk_arrays = [sr.ess_bulk for sr in shard_results if sr.ess_bulk is not None]
    combined_ess_bulk: np.ndarray | None = None
    if ess_bulk_arrays:
        combined_ess_bulk = np.min(np.stack(ess_bulk_arrays, axis=0), axis=0)

    ess_tail_arrays = [sr.ess_tail for sr in shard_results if sr.ess_tail is not None]
    combined_ess_tail: np.ndarray | None = None
    if ess_tail_arrays:
        combined_ess_tail = np.min(np.stack(ess_tail_arrays, axis=0), axis=0)

    all_bfmi: list[float] | None = None
    bfmi_lists = [sr.bfmi for sr in shard_results if sr.bfmi is not None]
    if bfmi_lists:
        all_bfmi = [min(bfmi_list) for bfmi_list in bfmi_lists]

    # --- Credible intervals: rebuild from combined mean/std (Gaussian approx) ---
    z95 = 1.959963985  # scipy.stats.norm.ppf(0.975)
    z89 = 1.598193423  # scipy.stats.norm.ppf(0.945)
    credible_intervals: dict[str, dict[str, float]] = {}
    for i, name in enumerate(parameter_names):
        mu = float(combined_mean[i])
        sigma = float(combined_std[i])
        credible_intervals[name] = {
            "lower_95": mu - z95 * sigma,
            "upper_95": mu + z95 * sigma,
            "lower_89": mu - z89 * sigma,
            "upper_89": mu + z89 * sigma,
        }

    # Convergence: all shards must pass
    convergence_passed = all(sr.convergence_passed for sr in shard_results)

    # --- Aggregate samples (concatenate across shards) ---
    combined_samples: dict[str, np.ndarray] | None = None
    if all(sr.samples is not None for sr in shard_results):
        combined_samples = {}
        for name in parameter_names:
            arrays = [
                np.asarray(sr.samples[name])  # type: ignore[index]
                for sr in shard_results
                if sr.samples is not None and name in sr.samples
            ]
            if arrays:
                combined_samples[name] = np.concatenate(arrays, axis=0)

    total_samples = sum(sr.num_samples for sr in shard_results)
    total_chains = sum(sr.num_chains for sr in shard_results)
    max_warmup = max(sr.num_warmup for sr in shard_results)
    total_wall_time: float | None = None
    wall_times = [
        sr.wall_time_seconds for sr in shard_results if sr.wall_time_seconds is not None
    ]
    if wall_times:
        total_wall_time = max(wall_times)  # Parallel shards: wall time = max shard

    return CMCResult(
        parameter_names=list(parameter_names),
        posterior_mean=combined_mean,
        posterior_std=combined_std,
        credible_intervals=credible_intervals,
        convergence_passed=convergence_passed,
        r_hat=combined_r_hat,
        ess_bulk=combined_ess_bulk,
        ess_tail=combined_ess_tail,
        bfmi=all_bfmi,
        samples=combined_samples,
        map_estimate=None,
        num_warmup=max_warmup,
        num_samples=total_samples,
        num_chains=total_chains,
        wall_time_seconds=total_wall_time,
        metadata={
            "n_shards": len(shard_results),
            "combination_method": "inverse_variance",
        },
    )


def cmc_result_summary_table(
    result: CMCResult,
    ci_level: str = "95",
    width: int = 80,
) -> str:
    """Format a CMCResult as a human-readable parameter summary table.

    The table includes columns for parameter name, posterior mean,
    posterior standard deviation, credible interval bounds, R-hat, and
    bulk ESS.  Missing diagnostics are shown as ``N/A``.

    Args:
        result: Completed CMC analysis result.
        ci_level: Credible interval level to display.  Must be ``"95"``
            or ``"89"``.  Defaults to ``"95"``.
        width: Total character width of the horizontal rule separators.
            Defaults to 80.

    Returns:
        Multi-line string containing the formatted table.

    Raises:
        ValueError: If ``ci_level`` is not ``"95"`` or ``"89"``.
    """
    if ci_level not in {"95", "89"}:
        raise ValueError(f"ci_level must be '95' or '89', got {ci_level!r}.")

    lower_key = f"lower_{ci_level}"
    upper_key = f"upper_{ci_level}"

    sep = "-" * width
    header_sep = "=" * width

    col_name = 18
    col_mean = 13
    col_std = 11
    col_ci = 25
    col_rhat = 8
    col_ess = 8

    header = (
        f"{'Parameter':<{col_name}}"
        f"{'Mean':>{col_mean}}"
        f"{'Std':>{col_std}}"
        f"{f'CI {ci_level}%':^{col_ci}}"
        f"{'R-hat':>{col_rhat}}"
        f"{'ESS':>{col_ess}}"
    )

    lines = [
        "CMC Posterior Summary",
        header_sep,
        f"Convergence : {'PASSED' if result.convergence_passed else 'FAILED'}",
        f"Chains      : {result.num_chains}",
        f"Samples     : {result.num_samples}",
        f"Warmup      : {result.num_warmup}",
    ]

    if result.wall_time_seconds is not None:
        lines.append(f"Wall time   : {result.wall_time_seconds:.1f} s")

    lines += ["", header, sep]

    for i, name in enumerate(result.parameter_names):
        mean = float(result.posterior_mean[i])
        std = float(result.posterior_std[i])

        ci_str = "N/A"
        if name in result.credible_intervals:
            ci = result.credible_intervals[name]
            lo = ci.get(lower_key)
            hi = ci.get(upper_key)
            if lo is not None and hi is not None:
                ci_str = f"[{lo:.3e}, {hi:.3e}]"

        r_hat_val = result.r_hat[i] if result.r_hat is not None else float("nan")
        ess_val = result.ess_bulk[i] if result.ess_bulk is not None else float("nan")

        r_hat_str = f"{r_hat_val:.3f}" if not np.isnan(r_hat_val) else "N/A"
        ess_str = f"{ess_val:.0f}" if not np.isnan(ess_val) else "N/A"

        row = (
            f"{name:<{col_name}}"
            f"{mean:>{col_mean}.4e}"
            f"{std:>{col_std}.2e}"
            f"{ci_str:^{col_ci}}"
            f"{r_hat_str:>{col_rhat}}"
            f"{ess_str:>{col_ess}}"
        )
        lines.append(row)

    lines.append(sep)

    if result.bfmi is not None:
        min_bfmi = min(result.bfmi)
        bfmi_flag = " [LOW]" if min_bfmi < 0.3 else ""
        lines.append(f"Min BFMI    : {min_bfmi:.3f}{bfmi_flag}")

    return "\n".join(lines)
