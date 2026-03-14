"""PBS/Torque job submission backend for Consensus Monte Carlo.

Submits per-shard MCMC sampling as PBS batch jobs, collects results from
completed jobs, and combines them.  Designed for HPC clusters running
PBS Professional or Torque where each shard runs on a separate node.

Usage::

    from heterodyne.optimization.cmc.backends.pbs import PBSBackend, PBSConfig

    cfg = PBSConfig(queue="large", walltime="04:00:00", nodes=1, ppn=8)
    backend = PBSBackend(pbs_config=cfg)
    samples = backend.run(model_fn, cmc_config, rng_key)
"""

from __future__ import annotations

import contextlib
import pickle  # Required: NumPyro model callables cannot be serialized as JSON.
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from heterodyne.optimization.cmc.backends.base import BackendCapabilities, CMCBackend
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax.numpy as jnp

    from heterodyne.optimization.cmc.config import CMCConfig

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BYTES_PER_FLOAT64: int = 8
_PBS_MEMORY_OVERHEAD_FACTOR: float = 8.0
_BYTES_PER_GB: float = 1024.0**3
_FILE_PREFIX: str = "heterodyne_cmc_pbs"

#: qstat exit code when a job ID has been purged from the database.
_QSTAT_UNKNOWN_JOB_EXIT: int = 35
_PBS_TERMINAL_STATES: frozenset[str] = frozenset({"C", "E", "F"})
_PBS_ACTIVE_STATES: frozenset[str] = frozenset({"Q", "W", "H", "R", "T", "S"})


# ---------------------------------------------------------------------------
# PBSConfig
# ---------------------------------------------------------------------------


@dataclass
class PBSConfig:
    """Configuration for PBS/Torque job submission.

    Attributes:
        queue: Target PBS queue name (e.g. ``"batch"``).
        walltime: Maximum wall-clock time in ``HH:MM:SS`` format.
        nodes: Number of nodes per shard job.
        ppn: Processors per node.
        memory: Memory request per node (e.g. ``"4gb"``).
        python_executable: Python interpreter accessible in the PBS job
            environment (full path or bare name like ``"python3"``).
        working_dir: Directory for temporary files.  Defaults to
            ``tempfile.gettempdir()`` when ``None``.
        extra_pbs_directives: Raw ``#PBS`` lines injected verbatim after
            the standard resource block.
        poll_interval: Seconds between ``qstat`` polls.
        max_retries: Re-submission attempts for failed shards.
        cleanup_on_success: Delete temporary files after successful runs.
    """

    queue: str = "batch"
    walltime: str = "01:00:00"
    nodes: int = 1
    ppn: int = 1
    memory: str = "4gb"
    python_executable: str = "python"
    working_dir: str | None = None
    extra_pbs_directives: list[str] = field(default_factory=list)
    poll_interval: float = 30.0
    max_retries: int = 2
    cleanup_on_success: bool = True


# ---------------------------------------------------------------------------
# ShardResult
# ---------------------------------------------------------------------------


@dataclass
class ShardResult:
    """Result from a single shard MCMC job.

    Attributes:
        shard_id: Zero-based shard index.
        samples: Posterior samples keyed by parameter name.
        job_id: PBS job identifier string.
        success: ``True`` when sampling completed without errors.
        error_message: Populated when ``success`` is ``False``.
    """

    shard_id: int
    samples: dict[str, Any]
    job_id: str
    success: bool = True
    error_message: str = ""


# ---------------------------------------------------------------------------
# Worker script (generated inline; no external script files required)
# ---------------------------------------------------------------------------

_WORKER_SCRIPT_TEMPLATE = '''\
#!/usr/bin/env python
"""PBS worker: loads shard data, runs MCMC, saves results."""
from __future__ import annotations
import pickle
import sys
from pathlib import Path
import jax
import numpy as np
jax.config.update("jax_platform_name", "cpu")
from numpyro.infer import MCMC, NUTS
from numpyro.infer import initialization as numpyro_init

_INIT_MAP = {
    "init_to_median": numpyro_init.init_to_median,
    "init_to_sample": numpyro_init.init_to_sample,
    "init_to_value": numpyro_init.init_to_value,
}

def main() -> None:
    data_path, result_path = Path(sys.argv[1]), Path(sys.argv[2])
    with data_path.open("rb") as fh:
        payload = pickle.load(fh)  # noqa: S301  # Trusted local file.
    model_fn = payload["model_fn"]
    config_dict = payload["config_dict"]
    init_params = payload.get("init_params")
    rng_key = jax.random.PRNGKey(payload["seed"])
    init_fn = _INIT_MAP.get(config_dict.get("init_strategy", "init_to_median"),
                            numpyro_init.init_to_median)
    kernel = NUTS(
        model_fn,
        target_accept_prob=config_dict["target_accept_prob"],
        max_tree_depth=config_dict.get("max_tree_depth", 10),
        dense_mass=config_dict.get("dense_mass", False),
        init_strategy=init_fn(),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=config_dict["num_warmup"],
        num_samples=config_dict["num_samples"],
        num_chains=config_dict.get("num_chains", 1),
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(rng_key, init_params=init_params, extra_fields=("energy",))
    np_samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    np.savez(result_path, **np_samples)
    print(f"Worker: saved {len(np_samples)} arrays to {result_path}")

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_pbs_available() -> bool:
    """Return True if ``qsub`` is present on PATH."""
    return shutil.which("qsub") is not None


def _submit_job(script_path: Path) -> str:
    """Call ``qsub`` on *script_path* and return the job ID string.

    Raises:
        RuntimeError: If ``qsub`` exits non-zero or returns an empty ID.
    """
    result = subprocess.run(
        ["qsub", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"qsub failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    job_id = result.stdout.strip()
    if not job_id:
        raise RuntimeError("qsub returned an empty job ID")
    return job_id


def _query_job_state(job_id: str) -> str | None:
    """Return the PBS job-state letter for *job_id*, or ``None`` if purged.

    Parses ``qstat -f <job_id>`` output for the ``job_state`` field.
    Returns ``None`` when the job has been purged from the accounting
    database (exit 35 or "Unknown Job" in stderr).
    """
    result = subprocess.run(
        ["qstat", "-f", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr_lower = result.stderr.lower()
        if (
            result.returncode == _QSTAT_UNKNOWN_JOB_EXIT
            or "unknown job" in stderr_lower
            or "invalid job id" in stderr_lower
        ):
            return None  # Job purged — caller checks result file for success.
        logger.warning(
            "qstat -f %s exited %d: %s",
            job_id,
            result.returncode,
            result.stderr.strip(),
        )
        return None

    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("job_state"):
            parts = stripped.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


def _cancel_job(job_id: str) -> None:
    """Send ``qdel`` to *job_id* (best-effort; errors are suppressed)."""
    with contextlib.suppress(Exception):
        subprocess.run(["qdel", job_id], capture_output=True, text=True, check=False)


def _build_pbs_script(
    *,
    job_name: str,
    pbs_cfg: PBSConfig,
    python_exe: str,
    worker_script_path: Path,
    data_path: Path,
    result_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> str:
    """Return a complete PBS job script as a string."""
    resource_line = (
        f"nodes={pbs_cfg.nodes}:ppn={pbs_cfg.ppn},"
        f"mem={pbs_cfg.memory},"
        f"walltime={pbs_cfg.walltime}"
    )
    extra_block = ""
    if pbs_cfg.extra_pbs_directives:
        extra_block = "\n" + "\n".join(pbs_cfg.extra_pbs_directives)
    return (
        "#!/bin/bash\n"
        f"#PBS -N {job_name}\n"
        f"#PBS -q {pbs_cfg.queue}\n"
        f"#PBS -l {resource_line}\n"
        f"#PBS -o {stdout_path}\n"
        f"#PBS -e {stderr_path}\n"
        f"{extra_block}\n"
        "cd $PBS_O_WORKDIR\n"
        f"{python_exe} {worker_script_path} {data_path} {result_path}\n"
    )


# ---------------------------------------------------------------------------
# PBSBackend
# ---------------------------------------------------------------------------


class PBSBackend(CMCBackend):
    """PBS/Torque backend for distributed CMC sampling.

    Each data shard is submitted as an independent PBS batch job.  The main
    process polls ``qstat`` until all jobs terminate, then reads per-shard
    ``.npz`` result files and concatenates the samples.

    Args:
        pbs_config: PBS resource and scheduling options.  Defaults to a
            ``PBSConfig()`` with ``queue="batch"`` when ``None``.
    """

    def __init__(self, pbs_config: PBSConfig | None = None) -> None:
        self._cfg = pbs_config or PBSConfig()
        self._work_dir = Path(self._cfg.working_dir or tempfile.gettempdir())
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._temp_paths: list[Path] = []

    # ------------------------------------------------------------------
    # CMCBackend abstract interface
    # ------------------------------------------------------------------

    def run(
        self,
        model: Callable[..., Any],
        config: CMCConfig,
        rng_key: jnp.ndarray,
        init_params: dict[str, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Submit per-shard PBS jobs, wait for completion, return combined samples.

        Args:
            model: NumPyro model function.  Must be picklable.
            config: CMC configuration with sampling hyperparameters.
            rng_key: JAX PRNG key used to derive per-shard integer seeds.
            init_params: Optional per-chain initial parameter values.

        Returns:
            Dictionary mapping parameter names to concatenated numpy arrays.

        Raises:
            RuntimeError: If PBS is unavailable or any shard fails.
            TimeoutError: If ``config.shard_timeout_seconds`` is exceeded.
        """
        import jax

        self.validate_resources()

        raw_shards = getattr(config, "num_shards", 1)
        num_shards = 1 if isinstance(raw_shards, str) else max(1, int(raw_shards))

        logger.info(
            "PBSBackend: submitting %d shard job(s) to queue '%s'",
            num_shards,
            self._cfg.queue,
        )

        config_dict = self._extract_config_dict(config)
        seeds = [
            int(jax.random.randint(jax.random.fold_in(rng_key, i), (), 0, 2**31 - 1))
            for i in range(num_shards)
        ]

        job_ids: list[str] = []
        for shard_id in range(num_shards):
            job_id = self.submit_shard(
                shard_data={"init_params": init_params},
                model_fn=model,
                config_dict=config_dict,
                shard_id=shard_id,
                seed=seeds[shard_id],
            )
            job_ids.append(job_id)
            logger.info("PBSBackend: shard %d submitted as job %s", shard_id, job_id)

        timeout = getattr(config, "shard_timeout_seconds", None)
        results = self.wait_for_jobs(job_ids, timeout=timeout)

        failed = [r for r in results if not r.success]
        if failed:
            ids = ", ".join(str(r.shard_id) for r in failed)
            msgs = "; ".join(r.error_message for r in failed)
            raise RuntimeError(
                f"PBSBackend: {len(failed)} shard(s) failed (ids: {ids}): {msgs}"
            )

        combined = self._combine_results(results)
        logger.info(
            "PBSBackend: combined %d shard(s), %d parameter(s)",
            len(results),
            len(combined),
        )

        if self._cfg.cleanup_on_success:
            self.cleanup(job_ids)

        return combined

    def get_capabilities(self) -> BackendCapabilities:
        """Return PBS backend capabilities."""
        return BackendCapabilities(
            supports_sharding=True,
            supports_parallel_chains=True,
            max_parallel_shards=256,
        )

    def validate_resources(self) -> None:
        """Check that ``qsub`` is on PATH.

        Raises:
            RuntimeError: If ``qsub`` is not found.
        """
        if not _check_pbs_available():
            raise RuntimeError(
                "PBSBackend: 'qsub' not found on PATH. "
                "Ensure PBS Professional or Torque is installed and the "
                "scheduler tools are in your shell PATH."
            )
        logger.debug("PBSBackend.validate_resources: qsub found")

    def estimate_memory(self, n_data: int, n_params: int, n_chains: int) -> float:
        """Estimate peak memory per PBS node (in GB) for a single shard.

        Because each shard runs on a separate node, the result is per-node
        only.  The estimate is conservative (upper-bound) to prevent OOM kills.
        """
        state_bytes = n_chains * 3 * n_params * _BYTES_PER_FLOAT64
        data_bytes = n_data * _BYTES_PER_FLOAT64
        return (state_bytes + data_bytes) * _PBS_MEMORY_OVERHEAD_FACTOR / _BYTES_PER_GB

    def cleanup(self, job_ids: list[str] | None = None) -> None:
        """Cancel active jobs and delete temporary files.

        Args:
            job_ids: PBS job IDs to cancel (best-effort).  ``None`` skips
                cancellation but still removes temporary files.
        """
        if job_ids:
            for jid in job_ids:
                if _query_job_state(jid) in _PBS_ACTIVE_STATES:
                    logger.debug("PBSBackend.cleanup: cancelling job %s", jid)
                    _cancel_job(jid)

        removed = 0
        for path in self._temp_paths:
            with contextlib.suppress(OSError):
                if path.is_file():
                    path.unlink()
                    removed += 1
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1
        self._temp_paths.clear()
        logger.debug("PBSBackend.cleanup: removed %d temporary path(s)", removed)

    # ------------------------------------------------------------------
    # Shard submission
    # ------------------------------------------------------------------

    def submit_shard(
        self,
        shard_data: dict[str, Any],
        model_fn: Callable[..., Any],
        config_dict: dict[str, Any],
        shard_id: int,
        seed: int,
    ) -> str:
        """Serialize shard payload, write PBS script, and submit via qsub.

        Args:
            shard_data: Auxiliary shard-specific data (e.g. ``init_params``).
            model_fn: Picklable NumPyro model function.
            config_dict: Flat NUTS hyperparameter dict.
            shard_id: Zero-based shard index (drives file naming).
            seed: Integer random seed for this shard.

        Returns:
            PBS job ID string.

        Raises:
            RuntimeError: If ``qsub`` fails.
        """
        tag = f"{_FILE_PREFIX}_shard{shard_id:04d}"

        worker_path = self._work_dir / f"{tag}_worker.py"
        worker_path.write_text(_WORKER_SCRIPT_TEMPLATE)
        self._temp_paths.append(worker_path)

        payload: dict[str, Any] = {
            "model_fn": model_fn,
            "config_dict": config_dict,
            "seed": seed,
            "init_params": shard_data.get("init_params"),
        }
        data_path = self._work_dir / f"{tag}_data.pkl"
        with data_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        self._temp_paths.append(data_path)

        # NumPy appends ".npz" to the stem automatically.
        result_stem = self._work_dir / f"{tag}_result"
        self._temp_paths.append(Path(str(result_stem) + ".npz"))

        stdout_path = self._work_dir / f"{tag}.o"
        stderr_path = self._work_dir / f"{tag}.e"
        self._temp_paths.extend([stdout_path, stderr_path])

        script_path = self._work_dir / f"{tag}.pbs"
        script_path.write_text(
            _build_pbs_script(
                job_name=f"hd_cmc_{shard_id:04d}",
                pbs_cfg=self._cfg,
                python_exe=self._cfg.python_executable,
                worker_script_path=worker_path,
                data_path=data_path,
                result_path=result_stem,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        )
        self._temp_paths.append(script_path)

        return _submit_job(script_path)

    # ------------------------------------------------------------------
    # Job monitoring
    # ------------------------------------------------------------------

    def wait_for_jobs(
        self,
        job_ids: list[str],
        timeout: float | None = None,
    ) -> list[ShardResult]:
        """Poll ``qstat`` until all jobs reach a terminal state.

        Jobs absent from ``qstat`` (purged from the accounting database) are
        treated as complete; the ``.npz`` file determines success or failure.

        Args:
            job_ids: PBS job IDs in shard order.
            timeout: Maximum seconds to wait (``None`` = unlimited).

        Returns:
            List of ``ShardResult`` in the same order as *job_ids*.

        Raises:
            TimeoutError: If *timeout* elapses before all jobs finish.
        """
        pending: dict[int, str] = dict(enumerate(job_ids))
        results: dict[int, ShardResult] = {}
        deadline = (time.monotonic() + timeout) if timeout is not None else None

        logger.info(
            "PBSBackend: waiting for %d job(s) (poll_interval=%.0fs)",
            len(job_ids),
            self._cfg.poll_interval,
        )

        while pending:
            if deadline is not None and time.monotonic() > deadline:
                remaining = list(pending.values())
                raise TimeoutError(
                    f"PBSBackend: timeout waiting for {len(remaining)} job(s): "
                    + ", ".join(remaining)
                )

            time.sleep(self._cfg.poll_interval)

            finished: list[int] = []
            for shard_id, job_id in pending.items():
                state = _query_job_state(job_id)
                if state is None or state in _PBS_TERMINAL_STATES:
                    result = self._collect_shard_result(shard_id, job_id)
                    results[shard_id] = result
                    finished.append(shard_id)
                    if result.success:
                        logger.info(
                            "PBSBackend: shard %d (job %s) completed",
                            shard_id,
                            job_id,
                        )
                    else:
                        logger.error(
                            "PBSBackend: shard %d (job %s) failed: %s",
                            shard_id,
                            job_id,
                            result.error_message,
                        )
                else:
                    logger.debug(
                        "PBSBackend: shard %d (job %s) state=%s",
                        shard_id,
                        job_id,
                        state,
                    )

            for shard_id in finished:
                del pending[shard_id]

        return [results[i] for i in range(len(job_ids))]

    # ------------------------------------------------------------------
    # Result collection and combination
    # ------------------------------------------------------------------

    def _collect_shard_result(self, shard_id: int, job_id: str) -> ShardResult:
        """Load the ``.npz`` result file written by the PBS worker."""
        import numpy as np

        tag = f"{_FILE_PREFIX}_shard{shard_id:04d}"
        result_npz = self._work_dir / f"{tag}_result.npz"

        if not result_npz.exists():
            return ShardResult(
                shard_id=shard_id,
                samples={},
                job_id=job_id,
                success=False,
                error_message=f"Result file not found: {result_npz}",
            )

        try:
            with np.load(result_npz, allow_pickle=False) as npz:
                samples: dict[str, Any] = {k: npz[k] for k in npz.files}
        except Exception as exc:  # noqa: BLE001
            return ShardResult(
                shard_id=shard_id,
                samples={},
                job_id=job_id,
                success=False,
                error_message=f"Failed to load {result_npz}: {exc}",
            )

        if not samples:
            return ShardResult(
                shard_id=shard_id,
                samples={},
                job_id=job_id,
                success=False,
                error_message=f"Result file {result_npz} contains no arrays",
            )

        return ShardResult(shard_id=shard_id, samples=samples, job_id=job_id)

    def _combine_results(self, results: list[ShardResult]) -> dict[str, Any]:
        """Concatenate posterior samples from all shards along axis 0."""
        import numpy as np

        combined: dict[str, list[Any]] = {}
        for result in results:
            for name, arr in result.samples.items():
                combined.setdefault(name, []).append(arr)
        return {name: np.concatenate(arrs, axis=0) for name, arrs in combined.items()}

    # ------------------------------------------------------------------
    # Higher-level convenience
    # ------------------------------------------------------------------

    def run_shards(
        self,
        shards: list[dict[str, Any]],
        model_fn: Callable[..., Any],
        config: CMCConfig,
        seeds: list[int],
    ) -> list[ShardResult]:
        """Submit and collect an explicit list of pre-partitioned shards.

        Use this when the caller has already split the data and needs to
        attach per-shard payloads.  For the standard pipeline use ``run()``.

        Args:
            shards: Per-shard data dicts passed verbatim to ``submit_shard``.
            model_fn: Picklable NumPyro model function.
            config: CMC configuration.
            seeds: One integer seed per shard.

        Returns:
            List of ``ShardResult`` in shard order.

        Raises:
            ValueError: If ``len(shards) != len(seeds)``.
            RuntimeError: If PBS is unavailable or submission fails.
            TimeoutError: If ``config.shard_timeout_seconds`` is exceeded.
        """
        if len(shards) != len(seeds):
            raise ValueError(
                f"run_shards: len(shards)={len(shards)} != len(seeds)={len(seeds)}"
            )

        self.validate_resources()
        config_dict = self._extract_config_dict(config)
        job_ids: list[str] = []

        for shard_id, (shard_data, seed) in enumerate(zip(shards, seeds, strict=True)):
            job_id = self.submit_shard(
                shard_data=shard_data,
                model_fn=model_fn,
                config_dict=config_dict,
                shard_id=shard_id,
                seed=seed,
            )
            job_ids.append(job_id)
            logger.info(
                "PBSBackend.run_shards: shard %d submitted as job %s",
                shard_id,
                job_id,
            )

        timeout = getattr(config, "shard_timeout_seconds", None)
        return self.wait_for_jobs(job_ids, timeout=timeout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_config_dict(self, config: CMCConfig) -> dict[str, Any]:
        """Flatten NUTS-relevant CMCConfig fields into a plain dict.

        The worker receives a plain dict (not the dataclass) to avoid pickle
        compatibility issues with forward references and optional dependencies.
        """
        return {
            "num_warmup": config.num_warmup,
            "num_samples": config.num_samples,
            "num_chains": 1,  # PBS mode: one chain per node.
            "target_accept_prob": config.target_accept_prob,
            "max_tree_depth": config.max_tree_depth,
            "dense_mass": config.dense_mass,
            "init_strategy": config.init_strategy,
        }
