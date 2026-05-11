"""Tests for NLSQ plotting helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.mark.unit
def test_plot_simulated_data_does_not_replace_model_time_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Display-only elapsed axes must not alter the model evaluation grid."""
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.viz.nlsq_plots import plot_simulated_data

    class FakeModel:
        def __init__(self) -> None:
            self._t = np.array([1.0, 2.0, 3.0])
            self.evaluation_t: np.ndarray | None = None

        @property
        def t(self) -> np.ndarray:
            return self._t

        def compute_correlation(
            self,
            *,
            phi_angle: float,
            contrast: float,
            offset: float,
        ) -> np.ndarray:
            _ = phi_angle, contrast, offset
            self.evaluation_t = np.asarray(self.t)
            return np.ones((len(self.t), len(self.t)))

    model = FakeModel()
    monkeypatch.setattr(HeterodyneModel, "from_config", lambda config: model)

    plot_simulated_data(
        config={},
        contrast=0.3,
        offset=1.0,
        phi_angles_str="0",
        plots_dir=tmp_path,
        data={"t1": np.array([0.0, 1.0, 2.0])},
    )

    assert np.array_equal(model._t, np.array([1.0, 2.0, 3.0]))
    assert model.evaluation_t is not None
    assert np.array_equal(model.evaluation_t, np.array([1.0, 2.0, 3.0]))


@pytest.mark.unit
@pytest.mark.regression
def test_plot_nlsq_fit_handles_diagonal_masked_residuals(tmp_path) -> None:
    """plot_nlsq_fit must not crash when residuals.size != c2_data.shape[0]**2.

    After _exclude_first_time_point and diagonal masking, residuals are
    1000*1000 - 1000 = 999000 flat elements, but c2_data.shape[0] == 1001.
    The old code tried residuals.reshape(1001, 1001) → ValueError.
    """
    from heterodyne.optimization.nlsq.results import NLSQResult
    from heterodyne.viz.nlsq_plots import plot_nlsq_fit, plot_residual_map

    # Simulate full (untrimmed) c2_data passed from plot_dispatch
    c2_data = np.ones((1001, 1001))
    # Simulate diagonal-masked flat residuals from the trimmed 1000x1000 fit
    masked_residuals = np.zeros(999_000)  # 1000*1000 - 1000 diagonal

    result = NLSQResult(
        parameters=np.zeros(3),
        parameter_names=["a", "b", "c"],
        success=False,
        message="Tier standard failed after 3 retries",
        fitted_correlation=np.ones((1000, 1000)),
        residuals=masked_residuals,
    )

    # Both functions must complete without ValueError
    fig1 = plot_nlsq_fit(c2_data, result, save_path=tmp_path / "fit.png")
    fig2 = plot_residual_map(result, c2_data, save_path=tmp_path / "resid.png")
    assert fig1 is not None
    assert fig2 is not None


@pytest.mark.unit
@pytest.mark.regression
def test_plot_nlsq_fit_handles_square_residuals(tmp_path) -> None:
    """plot_nlsq_fit reshape must still work when residuals are a perfect square."""
    from heterodyne.optimization.nlsq.results import NLSQResult
    from heterodyne.viz.nlsq_plots import plot_nlsq_fit

    n = 50
    c2_data = np.ones((n, n))
    result = NLSQResult(
        parameters=np.zeros(3),
        parameter_names=["a", "b", "c"],
        success=True,
        message="converged",
        fitted_correlation=np.ones((n, n)),
        residuals=np.zeros(n * n),
    )
    fig = plot_nlsq_fit(c2_data, result, save_path=tmp_path / "fit.png")
    assert fig is not None
