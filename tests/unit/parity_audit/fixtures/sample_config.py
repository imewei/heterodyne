"""Synthetic input for config extractor tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CMCConfig:
    target_accept: float = 0.8
    max_r_hat: float = 1.01
    nlsq_prior_width_factor: float = 1.0


def use_config(config: dict) -> tuple:
    a = config.get("optimization.cmc.dense_mass")
    b = config["optimization.nlsq.tolerance"]
    return a, b
