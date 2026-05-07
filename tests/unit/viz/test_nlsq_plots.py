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
