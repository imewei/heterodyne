"""Documentation content validation tests.

Ensures that documentation files exist, are non-empty, and reference
key concepts consistently with the codebase.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# Documentation Files Exist
# ============================================================================


class TestDocumentationFilesExist:
    """Verify essential documentation files are present."""

    @pytest.mark.unit
    def test_claude_md_exists(self) -> None:
        """CLAUDE.md project instructions exist."""
        assert (PROJECT_ROOT / "CLAUDE.md").is_file()

    @pytest.mark.unit
    def test_readme_exists(self) -> None:
        """README exists (README.md or README.rst)."""
        has_readme = (
            (PROJECT_ROOT / "README.md").is_file()
            or (PROJECT_ROOT / "README.rst").is_file()
        )
        assert has_readme, "No README.md or README.rst found"

    @pytest.mark.unit
    def test_pyproject_toml_exists(self) -> None:
        """pyproject.toml exists."""
        assert (PROJECT_ROOT / "pyproject.toml").is_file()

    @pytest.mark.unit
    def test_master_template_exists(self) -> None:
        """Master config template exists."""
        template = (
            PROJECT_ROOT
            / "heterodyne"
            / "config"
            / "templates"
            / "heterodyne_master_template.yaml"
        )
        assert template.is_file()


# ============================================================================
# CLAUDE.md Content Validation
# ============================================================================


class TestClaudeMdContent:
    """Verify CLAUDE.md references key project concepts."""

    @pytest.fixture
    def claude_md(self) -> str:
        return (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    @pytest.mark.unit
    def test_mentions_14_parameters(self, claude_md: str) -> None:
        """CLAUDE.md documents the 14-parameter model."""
        assert "14" in claude_md

    @pytest.mark.unit
    def test_mentions_two_path_architecture(self, claude_md: str) -> None:
        """CLAUDE.md documents the two-path integral architecture."""
        assert "meshgrid" in claude_md.lower() or "Meshgrid" in claude_md

    @pytest.mark.unit
    def test_mentions_cpu_only(self, claude_md: str) -> None:
        """CLAUDE.md documents CPU-only constraint."""
        assert "CPU" in claude_md

    @pytest.mark.unit
    def test_mentions_nlsq(self, claude_md: str) -> None:
        """CLAUDE.md documents NLSQ optimization."""
        assert "NLSQ" in claude_md or "nlsq" in claude_md

    @pytest.mark.unit
    def test_mentions_cmc(self, claude_md: str) -> None:
        """CLAUDE.md documents CMC optimization."""
        assert "CMC" in claude_md or "cmc" in claude_md

    @pytest.mark.unit
    def test_mentions_analyzer_parameters(self, claude_md: str) -> None:
        """CLAUDE.md documents analyzer_parameters config section."""
        assert "analyzer_parameters" in claude_md or "temporal" in claude_md


# ============================================================================
# Template Content Validation
# ============================================================================


class TestTemplateContent:
    """Verify master template contains required sections."""

    @pytest.fixture
    def template(self) -> str:
        path = (
            PROJECT_ROOT
            / "heterodyne"
            / "config"
            / "templates"
            / "heterodyne_master_template.yaml"
        )
        return path.read_text(encoding="utf-8")

    @pytest.mark.unit
    def test_has_analyzer_parameters(self, template: str) -> None:
        """Template has analyzer_parameters section."""
        assert "analyzer_parameters:" in template

    @pytest.mark.unit
    def test_has_wavevector_q(self, template: str) -> None:
        """Template includes wavevector_q."""
        assert "wavevector_q:" in template

    @pytest.mark.unit
    def test_has_stator_rotor_gap(self, template: str) -> None:
        """Template includes stator_rotor_gap geometry."""
        assert "stator_rotor_gap:" in template

    @pytest.mark.unit
    def test_has_start_end_frame(self, template: str) -> None:
        """Template uses frame-based selection."""
        assert "start_frame:" in template
        assert "end_frame:" in template

    @pytest.mark.unit
    def test_has_all_parameter_groups(self, template: str) -> None:
        """Template documents all five parameter groups."""
        groups = ["reference", "sample", "velocity", "fraction", "angle"]
        for group in groups:
            assert group.lower() in template.lower(), (
                f"Missing parameter group: {group}"
            )
