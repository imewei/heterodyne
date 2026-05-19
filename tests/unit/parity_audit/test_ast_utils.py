"""Tests for safe AST helpers used by extractors."""

from __future__ import annotations

import ast

from tools.parity_audit.ast_utils import (
    int_constant,
    string_constant,
    string_list_from_node,
)


def test_string_list_from_node_handles_list() -> None:
    node = ast.parse("['a', 'b', 'c']", mode="eval").body
    assert string_list_from_node(node) == ["a", "b", "c"]


def test_string_list_from_node_handles_tuple() -> None:
    node = ast.parse("('x', 'y')", mode="eval").body
    assert string_list_from_node(node) == ["x", "y"]


def test_string_list_from_node_filters_non_strings() -> None:
    node = ast.parse("['a', 1, 'b', None]", mode="eval").body
    assert string_list_from_node(node) == ["a", "b"]


def test_string_list_from_node_returns_empty_for_unknown() -> None:
    node = ast.parse("x", mode="eval").body
    assert string_list_from_node(node) == []


def test_string_list_from_node_handles_none() -> None:
    assert string_list_from_node(None) == []


def test_string_constant_returns_value() -> None:
    node = ast.parse("'hello'", mode="eval").body
    assert string_constant(node) == "hello"


def test_string_constant_returns_none_for_non_string() -> None:
    node = ast.parse("123", mode="eval").body
    assert string_constant(node) is None


def test_int_constant_returns_value() -> None:
    node = ast.parse("42", mode="eval").body
    assert int_constant(node) == 42


def test_int_constant_returns_none_for_non_int() -> None:
    node = ast.parse("'42'", mode="eval").body
    assert int_constant(node) is None
