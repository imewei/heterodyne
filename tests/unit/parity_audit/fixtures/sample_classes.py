"""Synthetic input for class extractor tests."""

from __future__ import annotations

from dataclasses import dataclass


class PlainClass:
    def public_method(self, x: int) -> int:
        return x

    def _private(self) -> None:
        return None


class Subclass(PlainClass):
    def another(self) -> str:
        return ""


@dataclass(frozen=True)
class DataCls:
    name: str
    value: int = 0
    _private: str = "hidden"


class _PrivateClass:
    pass
