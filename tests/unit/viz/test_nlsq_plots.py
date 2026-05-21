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
def test_plot_nlsq_fit_uses_full_n_by_n_arrays(tmp_path) -> None:
    """plot_nlsq_fit must not crash when c2_data and fitted_correlation are both N×N.

    Under the t=0 boundary contract (loaded and plotted, excluded only from
    chi-square via the residual mask), the orchestrator passes the full
    N-length t array to the model so fitted_correlation is N×N and aligns
    with the loaded c2_data. The plotter must compute residuals as
    ``c2_data - fitted_correlation`` directly — no shape-mismatch dance, no
    NaN padding.
    """
    from heterodyne.optimization.nlsq.results import NLSQResult
    from heterodyne.viz.nlsq_plots import plot_nlsq_fit, plot_residual_map

    n = 1001
    c2_data = np.ones((n, n))
    # Residual flat vector size matches the off-diagonal mask: n * (n - 1).
    # Under the new contract, boundary entries are zero (masked); diagonal
    # entries are excluded by the existing off-diagonal slice.
    masked_residuals = np.zeros(n * (n - 1))

    result = NLSQResult(
        parameters=np.zeros(3),
        parameter_names=["a", "b", "c"],
        success=True,
        message="ok",
        fitted_correlation=np.ones((n, n)),
        residuals=masked_residuals,
    )

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
