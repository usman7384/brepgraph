"""Optional transforms applied by the user or an adapter; never stored in the archive."""

from __future__ import annotations

import numpy as np

from ..errors import BrepGraphError

#: number of classes for the built-in categorical features
NUM_CLASSES = {"surface_type": 11, "curve_type": 9, "convexity": 3}


def _lookup(graph, name: str) -> np.ndarray:
    if name in graph.node_feats:
        return graph.node_feats[name]
    if name in graph.edge_feats:
        return graph.edge_feats[name]
    raise BrepGraphError(f"feature '{name}' not in graph")


def one_hot(graph, name: str, num_classes: int | None = None) -> np.ndarray:
    """``[N, C]`` float32 one-hot encoding of an integer feature."""
    arr = _lookup(graph, name)
    if arr.dtype.kind not in "iu":
        raise BrepGraphError(f"'{name}' is not categorical (dtype {arr.dtype})")
    c = num_classes or NUM_CLASSES.get(name) or int(arr.max(initial=-1)) + 1
    if arr.size and arr.max() >= c:
        raise BrepGraphError(f"'{name}' has class {int(arr.max())} but num_classes={c}")
    out = np.zeros((arr.shape[0], c), dtype=np.float32)
    out[np.arange(arr.shape[0]), arr] = 1.0
    return out


def standardize(graph, name: str, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-variance copy of a float feature (statistics over this graph only)."""
    arr = _lookup(graph, name).astype(np.float32)
    return (arr - arr.mean(axis=0)) / (arr.std(axis=0) + eps)
