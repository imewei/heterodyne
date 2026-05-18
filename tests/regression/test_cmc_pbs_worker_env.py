"""Regression for Codex finding C5: PBS worker template must satisfy Rule 8.

Rule 8: ``JAX_ENABLE_X64`` must be set BEFORE the first ``import jax`` in
any worker process.  The previous PBS worker template called ``import jax``
at L129 *before* configuring x64 or the persistent cache, which caused
downstream workers to silently run in float32 and re-JIT every kernel.

These checks operate on the textual template (which is materialised as
``<work_dir>/<tag>_worker.py`` per shard) because the worker is only
runnable on a real qsub host; importing/executing it here is not the goal.
"""

from __future__ import annotations

from heterodyne.optimization.cmc.backends import pbs


def _template() -> str:
    return pbs._WORKER_SCRIPT_TEMPLATE


class TestPBSWorkerEnvOrder:
    def test_sets_jax_enable_x64_before_import_jax(self) -> None:
        tpl = _template()
        env_pos = tpl.find("JAX_ENABLE_X64")
        import_pos = tpl.find("import jax")
        assert env_pos != -1, "Template must reference JAX_ENABLE_X64"
        assert import_pos != -1, "Template must reference import jax"
        assert env_pos < import_pos, (
            f"JAX_ENABLE_X64 must be set BEFORE 'import jax' "
            f"(env@{env_pos}, import@{import_pos})"
        )

    def test_calls_jax_config_update_enable_x64(self) -> None:
        tpl = _template()
        assert 'jax.config.update("jax_enable_x64", True)' in tpl

    def test_sets_compilation_cache_dir_when_env_present(self) -> None:
        tpl = _template()
        assert "JAX_COMPILATION_CACHE_DIR" in tpl
        assert "jax_compilation_cache_dir" in tpl
        assert "jax_persistent_cache_min_compile_time_secs" in tpl
        # Min compile time must be 0 so all kernels are cached.
        assert '"jax_persistent_cache_min_compile_time_secs", 0' in tpl

    def test_pbs_shell_exports_jax_enable_x64(self) -> None:
        """The PBS shell script must export JAX_ENABLE_X64=1 in the prelude."""
        from pathlib import Path

        from heterodyne.optimization.cmc.backends.pbs import (
            PBSConfig,
            _build_pbs_script,
        )

        cfg = PBSConfig()
        script = _build_pbs_script(
            job_name="test",
            pbs_cfg=cfg,
            python_exe="python",
            worker_script_path=Path("/tmp/worker.py"),
            data_path=Path("/tmp/data.pkl"),
            result_path=Path("/tmp/result"),
            stdout_path=Path("/tmp/out"),
            stderr_path=Path("/tmp/err"),
        )
        # The prelude must export the x64 flag before invoking python.
        export_pos = script.find("export JAX_ENABLE_X64")
        python_pos = script.find("python /tmp/worker.py")
        assert export_pos != -1, "PBS prelude must export JAX_ENABLE_X64"
        assert python_pos != -1, "PBS prelude must invoke the worker script"
        assert export_pos < python_pos
