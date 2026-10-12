"""PyTorch Geometric adapter (``pip install brepgraph[pyg]``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..errors import BrepGraphError, MissingExtraError

_RESERVED = {"x", "y", "pos", "edge_index", "edge_attr", "batch", "ptr", "num_nodes"}


def _tensor(torch, arr: np.ndarray):
    if arr.dtype.kind in "iu":  # integer indices and counts (incl. uint16) become torch.long
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


def _flat(torch, feats: dict, names: Sequence[str], n: int, what: str):
    parts = []
    for name in names:
        if name not in feats:
            raise BrepGraphError(f"{what} feature '{name}' not in graph; have {sorted(feats)}")
        parts.append(_tensor(torch, feats[name]).reshape(n, -1).float())
    return torch.cat(parts, dim=1)


def to_pyg(
    graph,
    x: Sequence[str] | None = None,
    edge_attr: Sequence[str] | None = None,
    pair_feats: Sequence[str] = (),
):
    """Return a ``torch_geometric.data.Data``.

    Every B-rep edge becomes two directed edges (``cat([ei, ei.flip(0)])``) and every edge feature
    is repeated unchanged for the reverse direction. All features are attached under their own
    names (``data.uv_grid``, ``data.area`` ...). ``x`` / ``edge_attr`` optionally concatenate flat
    features into ``data.x`` / ``data.edge_attr``. Avoid feature names containing ``face`` or
    ``index``: PyG's collation would concatenate them along the last dimension.

    Dense face-pair features (``[F, F, ...]``) are attached only when named in ``pair_feats``. PyG
    cannot batch graphs of different sizes with them out of the box; pad them in a custom collate.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise MissingExtraError(
            "to_pyg needs torch and torch_geometric: pip install 'brepgraph[pyg]'"
        ) from exc

    for name in (*graph.node_feats, *graph.edge_feats):
        if name in _RESERVED:
            raise BrepGraphError(f"feature name '{name}' collides with a PyG attribute")

    ei = torch.from_numpy(np.ascontiguousarray(graph.edge_index)).long()
    data = Data(edge_index=torch.cat([ei, ei.flip(0)], dim=1), num_nodes=graph.num_faces)
    for name, arr in graph.node_feats.items():
        setattr(data, name, _tensor(torch, arr))
    for name, arr in graph.edge_feats.items():
        t = _tensor(torch, arr)
        setattr(data, name, torch.cat([t, t], dim=0))
    for name in pair_feats:
        if name not in graph.pair_feats:
            raise BrepGraphError(f"pair feature '{name}' not in graph; have {sorted(graph.pair_feats)}")
        setattr(data, name, _tensor(torch, graph.pair_feats[name]))
    if x:
        data.x = _flat(torch, graph.node_feats, x, graph.num_faces, "node")
    if edge_attr:
        e = _flat(torch, graph.edge_feats, edge_attr, graph.num_edges, "edge")
        data.edge_attr = torch.cat([e, e], dim=0)
    return data
