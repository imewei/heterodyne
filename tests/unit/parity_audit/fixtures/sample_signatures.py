"""Synthetic input for signature extractor tests. Do not import in real code."""

from __future__ import annotations


def public_fn(a: int, b: str = "x") -> bool:
    return True


def _private_fn(a: int) -> None:
    return None


async def async_public(a: int) -> int:
    return a


class Cls:
    def public_method(self, x: int) -> int:
        return x

    def _private_method(self) -> None:
        return None

    @staticmethod
    def static_method(x: int) -> int:
        return x


def fn_with_complex_types(
    items: list[dict[str, int]],
    *,
    callback: Callable[[int], int] | None = None,
) -> dict[str, list[int]]:
    return {}
