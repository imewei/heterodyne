"""Tests for path validation utilities.

Tests resolve_path, validate_file_exists, validate_output_path,
and ensure_directory functions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from heterodyne.utils.path_validation import (
    PathValidationError,
    ensure_directory,
    resolve_path,
    validate_file_exists,
    validate_output_path,
)

# ============================================================================
# resolve_path Tests
# ============================================================================


class TestResolvePath:
    """Tests for resolve_path function."""

    @pytest.mark.unit
    def test_resolves_string_path(self) -> None:
        """Resolves string path to Path object."""
        result = resolve_path("/tmp")
        assert isinstance(result, Path)
        assert result.is_absolute()

    @pytest.mark.unit
    def test_resolves_path_object(self) -> None:
        """Resolves Path object to absolute Path."""
        result = resolve_path(Path("/tmp"))
        assert isinstance(result, Path)
        assert result.is_absolute()

    @pytest.mark.unit
    def test_expands_user_home(self) -> None:
        """Expands ~ to user home directory."""
        result = resolve_path("~")
        assert "~" not in str(result)
        assert result.is_absolute()

    @pytest.mark.unit
    def test_resolves_relative_path(self) -> None:
        """Resolves relative path to absolute."""
        result = resolve_path(".")
        assert result.is_absolute()
        assert result == Path.cwd()


# ============================================================================
# validate_file_exists Tests
# ============================================================================


class TestValidateFileExists:
    """Tests for validate_file_exists function."""

    @pytest.mark.unit
    def test_valid_existing_file(self) -> None:
        """Returns resolved path for existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            result = validate_file_exists(temp_path)
            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_file()
        finally:
            Path(temp_path).unlink()

    @pytest.mark.unit
    def test_nonexistent_file_raises(self) -> None:
        """Raises PathValidationError for nonexistent file."""
        with pytest.raises(PathValidationError) as excinfo:
            validate_file_exists("/nonexistent/path/to/file.txt")
        assert "not found" in str(excinfo.value)

    @pytest.mark.unit
    def test_directory_not_file_raises(self) -> None:
        """Raises PathValidationError when path is a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(PathValidationError) as excinfo:
                validate_file_exists(temp_dir)
            assert "not a file" in str(excinfo.value)

    @pytest.mark.unit
    def test_custom_description_in_error(self) -> None:
        """Custom description appears in error message."""
        with pytest.raises(PathValidationError) as excinfo:
            validate_file_exists("/nonexistent.txt", description="Config file")
        assert "Config file" in str(excinfo.value)


# ============================================================================
# validate_output_path Tests
# ============================================================================


class TestValidateOutputPath:
    """Tests for validate_output_path function."""

    @pytest.mark.unit
    def test_valid_new_file_path(self) -> None:
        """Returns resolved path for new file location."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.txt"
            result = validate_output_path(output_path)
            assert isinstance(result, Path)
            assert result.is_absolute()

    @pytest.mark.unit
    def test_creates_parent_directories(self) -> None:
        """Creates parent directories when create_parents=True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "dir" / "output.txt"
            result = validate_output_path(output_path, create_parents=True)
            assert result.parent.exists()

    @pytest.mark.unit
    def test_no_create_parents_raises_when_missing(self) -> None:
        """Raises when parent doesn't exist and create_parents=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nonexistent" / "output.txt"
            with pytest.raises(PathValidationError) as excinfo:
                validate_output_path(output_path, create_parents=False)
            assert "Parent directory does not exist" in str(excinfo.value)

    @pytest.mark.unit
    def test_directory_as_output_path_raises(self) -> None:
        """Raises when output path is an existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(PathValidationError) as excinfo:
                validate_output_path(temp_dir)
            assert "is a directory" in str(excinfo.value)


# ============================================================================
# ensure_directory Tests
# ============================================================================


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    @pytest.mark.unit
    def test_creates_new_directory(self) -> None:
        """Creates new directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_directory"
            assert not new_dir.exists()

            result = ensure_directory(new_dir)
            assert result.exists()
            assert result.is_dir()

    @pytest.mark.unit
    def test_creates_nested_directories(self) -> None:
        """Creates nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "a" / "b" / "c"
            result = ensure_directory(nested_dir)
            assert result.exists()
            assert result.is_dir()

    @pytest.mark.unit
    def test_existing_directory_unchanged(self) -> None:
        """Existing directory is returned unchanged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = ensure_directory(temp_dir)
            assert result.exists()
            assert result.is_dir()

    @pytest.mark.unit
    def test_returns_absolute_path(self) -> None:
        """Returns absolute resolved path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = ensure_directory(temp_dir)
            assert result.is_absolute()


# ============================================================================
# PathValidationError Tests
# ============================================================================


class TestPathValidationError:
    """Tests for PathValidationError exception."""

    @pytest.mark.unit
    def test_is_exception(self) -> None:
        """PathValidationError is an Exception."""
        error = PathValidationError("test error")
        assert isinstance(error, Exception)

    @pytest.mark.unit
    def test_message_preserved(self) -> None:
        """Error message is preserved."""
        message = "Path /foo/bar not found"
        error = PathValidationError(message)
        assert str(error) == message


# ============================================================================
# Integration Tests
# ============================================================================


class TestPathValidationIntegration:
    """Integration tests for path validation workflow."""

    @pytest.mark.integration
    def test_complete_workflow_existing_file(self) -> None:
        """Complete workflow: create file, validate, read."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file
            file_path = Path(temp_dir) / "test.txt"
            file_path.write_text("test content")

            # Validate it exists
            validated = validate_file_exists(file_path)
            assert validated.read_text() == "test content"

    @pytest.mark.integration
    def test_complete_workflow_output_file(self) -> None:
        """Complete workflow: validate output path, write."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Validate output path with nested directories
            output_path = Path(temp_dir) / "results" / "output.txt"
            validated = validate_output_path(output_path, create_parents=True)

            # Write to validated path
            validated.write_text("output content")
            assert validated.read_text() == "output content"

    @pytest.mark.integration
    def test_ensure_directory_then_write_file(self) -> None:
        """Ensure directory exists then write file to it."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = ensure_directory(Path(temp_dir) / "outputs")
            output_file = output_dir / "result.txt"
            output_file.write_text("result")
            assert output_file.read_text() == "result"
