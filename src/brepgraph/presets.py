"""Named feature sets."""

from __future__ import annotations

from .errors import UnknownFeatureError
from .registry import FeatureSet

uvnet = FeatureSet(
    nodes=["surface_type", "area", "centroid", "uv_grid"],
    edges=["curve_type", "length", "convexity", "dihedral_angle", "u_grid"],
)

# BrepMFR (Zhang et al., CAGD 111, 2024): 5x5 grids, loop / adjacency counts, and the face-pair
# proximity matrices (shortest-path distance, D2 and A3 histograms with 64 bins).
brepmfr = FeatureSet(
    nodes=["surface_type", "area", "num_loops", "num_adjacent_faces", ("uv_grid", {"size": 5})],
    edges=["curve_type", "length", "convexity", "dihedral_angle", ("u_grid", {"size": 5})],
    pairs=["shortest_path", "d2", "a3"],
)

_PRESETS = {"uvnet": uvnet, "brepmfr": brepmfr}


def get(name: str) -> FeatureSet:
    try:
        return _PRESETS[name]
    except KeyError:
        raise UnknownFeatureError(f"unknown preset '{name}'; available: {sorted(_PRESETS)}") from None


def names() -> list[str]:
    return sorted(_PRESETS)
