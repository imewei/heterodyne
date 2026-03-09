"""Unit tests for heterodyne.io.json_utils."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from heterodyne.io.json_utils import (
    _sanitize_float,
    json_safe,
    json_serializer,
    load_json,
    save_json,
)


# ---------------------------------------------------------------------------
# _sanitize_float
# ---------------------------------------------------------------------------


class TestSanitizeFloat:
    """Tests for _sanitize_float."""

    def test_finite_float_passthrough(self) -> None:
        assert _sanitize_float(1.5) == 1.5

    def test_zero(self) -> None:
        assert _sanitize_float(0.0) == 0.0

    def test_negative(self) -> None:
        assert _sanitize_float(-3.14) == -3.14

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _sanitize_float(float("nan"))

    def test_positive_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _sanitize_float(float("inf"))

    def test_negative_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _sanitize_float(float("-inf"))


# ---------------------------------------------------------------------------
# json_safe — primitives
# ---------------------------------------------------------------------------


class TestJsonSafePrimitives:
    """Tests for json_safe on primitive types."""

    def test_none(self) -> None:
        assert json_safe(None) is None

    def test_bool_true(self) -> None:
        assert json_safe(True) is True

    def test_bool_false(self) -> None:
        assert json_safe(False) is False

    def test_int(self) -> None:
        assert json_safe(42) == 42

    def test_string(self) -> None:
        assert json_safe("hello") == "hello"

    def test_finite_float(self) -> None:
        assert json_safe(2.5) == 2.5

    def test_nan_float_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            json_safe(float("nan"))

    def test_inf_float_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            json_safe(float("inf"))


# ---------------------------------------------------------------------------
# json_safe — numpy types
# ---------------------------------------------------------------------------


class TestJsonSafeNumpy:
    """Tests for json_safe on numpy types."""

    def test_ndarray_1d(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = json_safe(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_ndarray_2d(self) -> None:
        arr = np.array([[1, 2], [3, 4]])
        result = json_safe(arr)
        assert result == [[1, 2], [3, 4]]

    def test_empty_ndarray(self) -> None:
        arr = np.array([])
        result = json_safe(arr)
        assert result == []

    def test_numpy_int(self) -> None:
        val = np.int64(7)
        result = json_safe(val)
        assert result == 7
        assert isinstance(result, int)

    def test_numpy_float(self) -> None:
        val = np.float64(3.14)
        result = json_safe(val)
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_numpy_bool(self) -> None:
        val = np.bool_(True)
        result = json_safe(val)
        assert result is True
        assert isinstance(result, bool)

    def test_complex_ndarray(self) -> None:
        arr = np.array([1 + 2j, 3 + 4j])
        result = json_safe(arr)
        assert result["__complex_array__"] is True
        assert result["shape"] == [2]
        assert len(result["data"]) == 2
        assert result["data"][0] == {"real": 1.0, "imag": 2.0}
        assert result["data"][1] == {"real": 3.0, "imag": 4.0}

    def test_complex_ndarray_2d(self) -> None:
        arr = np.array([[1 + 0j, 2 + 1j]])
        result = json_safe(arr)
        assert result["__complex_array__"] is True
        assert result["shape"] == [1, 2]
        assert len(result["data"]) == 2

    def test_numpy_complex_scalar(self) -> None:
        val = np.complex128(1 + 2j)
        result = json_safe(val)
        assert result == {"real": 1.0, "imag": 2.0}


# ---------------------------------------------------------------------------
# json_safe — Python complex
# ---------------------------------------------------------------------------


class TestJsonSafeComplex:
    """Tests for json_safe on Python complex numbers."""

    def test_python_complex(self) -> None:
        result = json_safe(3 + 4j)
        assert result == {"real": 3.0, "imag": 4.0}

    def test_python_complex_zero_imag(self) -> None:
        result = json_safe(5 + 0j)
        assert result == {"real": 5.0, "imag": 0.0}


# ---------------------------------------------------------------------------
# json_safe — Path and datetime
# ---------------------------------------------------------------------------


class TestJsonSafePathDatetime:
    """Tests for json_safe on Path and datetime objects."""

    def test_path(self) -> None:
        p = Path("/tmp/test/file.json")
        result = json_safe(p)
        assert result == "/tmp/test/file.json"
        assert isinstance(result, str)

    def test_datetime(self) -> None:
        dt = datetime(2026, 3, 9, 12, 30, 45)
        result = json_safe(dt)
        assert result == "2026-03-09T12:30:45"


# ---------------------------------------------------------------------------
# json_safe — nested structures
# ---------------------------------------------------------------------------


class TestJsonSafeNested:
    """Tests for json_safe on nested dicts and lists."""

    def test_nested_dict(self) -> None:
        data = {"a": np.int64(1), "b": np.array([2.0, 3.0])}
        result = json_safe(data)
        assert result == {"a": 1, "b": [2.0, 3.0]}

    def test_nested_list(self) -> None:
        data = [np.float64(1.0), "hello", None]
        result = json_safe(data)
        assert result == [1.0, "hello", None]

    def test_tuple_becomes_list(self) -> None:
        data = (1, np.int32(2), "three")
        result = json_safe(data)
        assert result == [1, 2, "three"]
        assert isinstance(result, list)

    def test_deeply_nested(self) -> None:
        data = {"level1": {"level2": {"value": np.float32(1.5)}}}
        result = json_safe(data)
        assert result == {"level1": {"level2": {"value": pytest.approx(1.5)}}}

    def test_empty_dict(self) -> None:
        assert json_safe({}) == {}

    def test_empty_list(self) -> None:
        assert json_safe([]) == []


# ---------------------------------------------------------------------------
# json_safe — fallback
# ---------------------------------------------------------------------------


class TestJsonSafeFallback:
    """Tests for json_safe fallback behavior."""

    def test_custom_object_str_fallback(self) -> None:
        class Foo:
            def __str__(self) -> str:
                return "I am Foo"

        result = json_safe(Foo())
        assert result == "I am Foo"

    def test_non_serializable_fallback(self) -> None:
        class BadObj:
            def __str__(self) -> str:
                raise RuntimeError("cannot stringify")

        result = json_safe(BadObj())
        assert "<non-serializable:" in result


# ---------------------------------------------------------------------------
# json_serializer
# ---------------------------------------------------------------------------


class TestJsonSerializer:
    """Tests for json_serializer."""

    def test_produces_valid_json(self) -> None:
        data = {"x": np.array([1, 2, 3]), "y": 4.5}
        result = json_serializer(data)
        parsed = json.loads(result)
        assert parsed["x"] == [1, 2, 3]
        assert parsed["y"] == 4.5

    def test_pretty_printed(self) -> None:
        data = {"key": "value"}
        result = json_serializer(data)
        # Pretty-printed means newlines are present
        assert "\n" in result

    def test_nan_raises_in_serializer(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            json_serializer({"val": float("nan")})

    def test_inf_raises_in_serializer(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            json_serializer({"val": float("inf")})


# ---------------------------------------------------------------------------
# save_json / load_json round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadJson:
    """Tests for save_json and load_json."""

    def test_round_trip(self, tmp_path: Path) -> None:
        data = {"alpha": 1.5, "names": ["a", "b"], "count": 10}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_round_trip_string_path(self, tmp_path: Path) -> None:
        data = {"key": "value"}
        path = str(tmp_path / "test2.json")
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "test.json"
        save_json({"x": 1}, path)
        assert path.exists()
        assert load_json(path) == {"x": 1}

    def test_numpy_round_trip(self, tmp_path: Path) -> None:
        data = {"params": np.array([1.0, 2.0, 3.0]), "n": np.int64(5)}
        path = tmp_path / "numpy.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded["params"] == [1.0, 2.0, 3.0]
        assert loaded["n"] == 5

    def test_atomic_write_no_partial_on_error(self, tmp_path: Path) -> None:
        """If serialization fails, file should not exist."""
        path = tmp_path / "fail.json"
        with pytest.raises(ValueError):
            save_json({"bad": float("nan")}, path)
        assert not path.exists()

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "overwrite.json"
        save_json({"v": 1}, path)
        save_json({"v": 2}, path)
        assert load_json(path) == {"v": 2}

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nope.json")
