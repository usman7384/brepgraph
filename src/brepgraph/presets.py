"""Named feature sets."""

from __future__ import annotations

from .errors import UnknownFeatureError
from .registry import FeatureSet

uvnet = FeatureSet(
    nodes=["surface_type", "area", "centroid", "uv_grid"],
    edges=["curve_type", "length", "convexity", "dihedral_angle", "u_grid"],
)

_PRESETS = {"uvnet": uvnet}


def get(name: str) -> FeatureSet:
    try:
        return _PRESETS[name]
    except KeyError:
        raise UnknownFeatureError(f"unknown preset '{name}'; available: {sorted(_PRESETS)}") from None


def names() -> list[str]:
    return sorted(_PRESETS)
