"""Unit tests for heterodyne.viz.mcmc_arviz module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from heterodyne.viz.mcmc_arviz import (
    _create_empty_figure,
    _has_arviz,
    plot_arviz_pair,
    plot_arviz_posterior,
    plot_arviz_trace,
    to_inference_data,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:  # type: ignore[misc]
    """Close all matplotlib figures after each test."""
    yield  # type: ignore[misc]
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cmc_result(
    *,
    parameter_names: list[str] | None = None,
    samples: dict[str, np.ndarray] | None = None,
) -> object:
    """Create a lightweight mock CMCResult."""
    if parameter_names is None:
        parameter_names = ["alpha", "beta"]
    if samples is None:
        rng = np.random.default_rng(42)
        samples = {n: rng.standard_normal(200) for n in parameter_names}

    n_params = len(parameter_names)
    return SimpleNamespace(
        parameter_names=parameter_names,
        posterior_mean=np.zeros(n_params),
        posterior_std=np.ones(n_params),
        credible_intervals={
            n: {"hdi_3%": -1.0, "hdi_97%": 1.0} for n in parameter_names
        },
        convergence_passed=True,
        r_hat=np.ones(n_params),
        ess_bulk=np.full(n_params, 400.0),
        ess_tail=np.full(n_params, 300.0),
        bfmi=[0.9],
        samples=samples,
        num_chains=1,
    )


def _make_mock_idata() -> object:
    """Create a mock InferenceData-like object with a posterior group."""
    rng = np.random.default_rng(42)
    return SimpleNamespace(
        posterior=SimpleNamespace(
            data_vars={"alpha": rng.standard_normal((1, 200))},
        ),
    )


def _make_mock_axes_array() -> np.ndarray:
    """Create a numpy array of real matplotlib Axes (as ArviZ returns)."""
    fig, axes = plt.subplots(2, 2)
    return axes


# ---------------------------------------------------------------------------
# _has_arviz
# ---------------------------------------------------------------------------


class TestHasArviz:
    def test_returns_bool(self) -> None:
        result = _has_arviz()
        assert isinstance(result, bool)

    def test_returns_true_when_available(self) -> None:
        # ArviZ is a project dependency, so it should be available
        assert _has_arviz() is True


# ---------------------------------------------------------------------------
# _create_empty_figure
# ---------------------------------------------------------------------------


class TestCreateEmptyFigure:
    def test_returns_figure(self) -> None:
        fig = _create_empty_figure()
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self) -> None:
        fig = _create_empty_figure(title="Custom message")
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        texts = [t.get_text() for t in ax.texts]
        assert "Custom message" in texts

    def test_default_title(self) -> None:
        fig = _create_empty_figure()
        ax = fig.get_axes()[0]
        texts = [t.get_text() for t in ax.texts]
        assert "No data available" in texts

    def test_single_axes(self) -> None:
        fig = _create_empty_figure()
        axes = fig.get_axes()
        assert len(axes) == 1


# ---------------------------------------------------------------------------
# to_inference_data
# ---------------------------------------------------------------------------


class TestToInferenceData:
    def test_delegates_to_cmc_result_to_arviz(self) -> None:
        """Verify to_inference_data delegates to cmc_result_to_arviz."""
        result = _make_cmc_result()
        mock_idata = _make_mock_idata()

        with patch(
            "heterodyne.viz.mcmc_arviz.cmc_result_to_arviz",
            return_value=mock_idata,
            create=True,
        ) as _mock_fn:
            # Patch at the import site
            with patch(
                "heterodyne.optimization.cmc.results.cmc_result_to_arviz",
                return_value=mock_idata,
            ):
                idata = to_inference_data(result)  # type: ignore[arg-type]
                assert idata is mock_idata


# ---------------------------------------------------------------------------
# plot_arviz_trace
# ---------------------------------------------------------------------------


class TestPlotArvizTrace:
    def test_returns_figure(self) -> None:
        result = _make_cmc_result()
        mock_axes = _make_mock_axes_array()
        mock_idata = _make_mock_idata()

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=True),
            patch(
                "heterodyne.viz.mcmc_arviz.to_inference_data",
                return_value=mock_idata,
            ),
            patch("arviz.plot_trace", return_value=mock_axes),
        ):
            fig = plot_arviz_trace(result)  # type: ignore[arg-type]
            assert isinstance(fig, plt.Figure)

    def test_save_path(self, tmp_path: pytest.TempPathFactory) -> None:
        result = _make_cmc_result()
        mock_axes = _make_mock_axes_array()
        mock_idata = _make_mock_idata()
        save_file = tmp_path / "trace.png"  # type: ignore[operator]

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=True),
            patch(
                "heterodyne.viz.mcmc_arviz.to_inference_data",
                return_value=mock_idata,
            ),
            patch("arviz.plot_trace", return_value=mock_axes),
        ):
            _fig = plot_arviz_trace(result, save_path=save_file)  # type: ignore[arg-type]
            assert save_file.exists()  # type: ignore[union-attr]

    def test_var_names_passed_through(self) -> None:
        result = _make_cmc_result()
        mock_axes = _make_mock_axes_array()
        mock_idata = _make_mock_idata()

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=True),
            patch(
                "heterodyne.viz.mcmc_arviz.to_inference_data",
                return_value=mock_idata,
            ),
            patch("arviz.plot_trace", return_value=mock_axes) as mock_plot,
        ):
            plot_arviz_trace(result, var_names=["alpha"])  # type: ignore[arg-type]
            mock_plot.assert_called_once_with(
                mock_idata, var_names=["alpha"], compact=True
            )

    def test_fallback_when_arviz_missing(self) -> None:
        result = _make_cmc_result()
        mock_plot_trace = MagicMock(return_value=None)

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=False),
            patch(
                "heterodyne.viz.mcmc_plots.plot_trace",
                mock_plot_trace,
            ),
        ):
            plot_arviz_trace(result)  # type: ignore[arg-type]
            mock_plot_trace.assert_called_once()


# ---------------------------------------------------------------------------
# plot_arviz_posterior
# ---------------------------------------------------------------------------


class TestPlotArvizPosterior:
    def test_returns_figure(self) -> None:
        result = _make_cmc_result()
        mock_axes = _make_mock_axes_array()
        mock_idata = _make_mock_idata()

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=True),
            patch(
                "heterodyne.viz.mcmc_arviz.to_inference_data",
                return_value=mock_idata,
            ),
            patch("arviz.plot_posterior", return_value=mock_axes, create=True),
        ):
            fig = plot_arviz_posterior(result)  # type: ignore[arg-type]
            assert isinstance(fig, plt.Figure)

    def test_custom_hdi_prob(self) -> None:
        result = _make_cmc_result()
        mock_axes = _make_mock_axes_array()
        mock_idata = _make_mock_idata()

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=True),
            patch(
                "heterodyne.viz.mcmc_arviz.to_inference_data",
                return_value=mock_idata,
            ),
            patch(
                "arviz.plot_posterior", return_value=mock_axes, create=True
            ) as mock_plot,
        ):
            plot_arviz_posterior(result, hdi_prob=0.89)  # type: ignore[arg-type]
            mock_plot.assert_called_once_with(mock_idata, var_names=None, hdi_prob=0.89)

    def test_save_path(self, tmp_path: pytest.TempPathFactory) -> None:
        result = _make_cmc_result()
        mock_axes = _make_mock_axes_array()
        mock_idata = _make_mock_idata()
        save_file = tmp_path / "posterior.png"  # type: ignore[operator]

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=True),
            patch(
                "heterodyne.viz.mcmc_arviz.to_inference_data",
                return_value=mock_idata,
            ),
            patch("arviz.plot_posterior", return_value=mock_axes, create=True),
        ):
            plot_arviz_posterior(result, save_path=save_file)  # type: ignore[arg-type]
            assert save_file.exists()  # type: ignore[union-attr]

    def test_fallback_when_arviz_missing(self) -> None:
        result = _make_cmc_result()
        mock_plot_posterior = MagicMock(return_value=None)

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=False),
            patch(
                "heterodyne.viz.mcmc_plots.plot_posterior",
                mock_plot_posterior,
            ),
        ):
            plot_arviz_posterior(result)  # type: ignore[arg-type]
            mock_plot_posterior.assert_called_once()

    def test_handles_single_axes_return(self) -> None:
        """When az.plot_posterior returns a single Axes (not array)."""
        result = _make_cmc_result()
        _, single_ax = plt.subplots()
        mock_idata = _make_mock_idata()

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=True),
            patch(
                "heterodyne.viz.mcmc_arviz.to_inference_data",
                return_value=mock_idata,
            ),
            patch("arviz.plot_posterior", return_value=single_ax, create=True),
        ):
            fig = plot_arviz_posterior(result)  # type: ignore[arg-type]
            assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_arviz_pair
# ---------------------------------------------------------------------------


class TestPlotArvizPair:
    def test_returns_figure(self) -> None:
        result = _make_cmc_result()
        mock_axes = _make_mock_axes_array()
        mock_idata = _make_mock_idata()

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=True),
            patch(
                "heterodyne.viz.mcmc_arviz.to_inference_data",
                return_value=mock_idata,
            ),
            patch("arviz.plot_pair", return_value=mock_axes),
        ):
            fig = plot_arviz_pair(result)  # type: ignore[arg-type]
            assert isinstance(fig, plt.Figure)

    def test_var_names_passed_through(self) -> None:
        result = _make_cmc_result()
        mock_axes = _make_mock_axes_array()
        mock_idata = _make_mock_idata()

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=True),
            patch(
                "heterodyne.viz.mcmc_arviz.to_inference_data",
                return_value=mock_idata,
            ),
            patch("arviz.plot_pair", return_value=mock_axes) as mock_plot,
        ):
            plot_arviz_pair(result, var_names=["alpha"])  # type: ignore[arg-type]
            mock_plot.assert_called_once_with(
                mock_idata,
                var_names=["alpha"],
                kind="kde",
                marginals=True,
            )

    def test_save_path(self, tmp_path: pytest.TempPathFactory) -> None:
        result = _make_cmc_result()
        mock_axes = _make_mock_axes_array()
        mock_idata = _make_mock_idata()
        save_file = tmp_path / "pair.png"  # type: ignore[operator]

        with (
            patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=True),
            patch(
                "heterodyne.viz.mcmc_arviz.to_inference_data",
                return_value=mock_idata,
            ),
            patch("arviz.plot_pair", return_value=mock_axes),
        ):
            plot_arviz_pair(result, save_path=save_file)  # type: ignore[arg-type]
            assert save_file.exists()  # type: ignore[union-attr]

    def test_returns_none_when_arviz_missing(self) -> None:
        result = _make_cmc_result()
        with patch("heterodyne.viz.mcmc_arviz._has_arviz", return_value=False):
            fig = plot_arviz_pair(result)  # type: ignore[arg-type]
            assert fig is None
