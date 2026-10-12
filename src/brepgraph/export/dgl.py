"""DGL adapter (``pip install brepgraph[dgl]``)."""

from __future__ import annotations

import numpy as np

from ..errors import MissingExtraError


def _tensor(torch, arr: np.ndarray):
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.long() if arr.dtype.kind in "iu" else t.float()


def to_dgl(graph):
    """Return a ``dgl.DGLGraph`` with both directions of every B-rep edge.

    Node features go to ``ndata`` and edge features (repeated for the reverse direction) to
    ``edata`` under their own names. Face-pair features are not attached (dense ``[F, F, ...]``).
    """
    try:
        import dgl
        import torch
    except ImportError as exc:
        raise MissingExtraError("to_dgl needs torch and dgl: pip install 'brepgraph[dgl]'") from exc

    src = torch.from_numpy(np.ascontiguousarray(graph.edge_index[0])).long()
    dst = torch.from_numpy(np.ascontiguousarray(graph.edge_index[1])).long()
    g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), num_nodes=graph.num_faces)
    for name, arr in graph.node_feats.items():
        g.ndata[name] = _tensor(torch, arr)
    for name, arr in graph.edge_feats.items():
        t = _tensor(torch, arr)
        g.edata[name] = torch.cat([t, t], dim=0)
    return g
