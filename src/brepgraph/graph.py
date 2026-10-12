"""The framework-neutral graph object and its ``.npz`` serialisation (no pickle, no OCP handles)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import BrepGraphError, SchemaVersionError

SCHEMA_VERSION = "1.0.0"


@dataclass(eq=False)
class BrepGraph:
    """Face-adjacency graph of one solid. See ``docs/schema.md``.

    ``edge_index`` is ``[2, E]`` int64 with one column per retained B-rep edge, ``src < dst``;
    parallel edges are kept. ``node_feats[name]`` is ``[F, ...]`` and ``edge_feats[name]`` is
    ``[E, ...]`` with row ``i`` belonging to column ``i`` of ``edge_index``.
    """

    num_faces: int
    num_edges: int
    edge_index: np.ndarray
    node_feats: dict[str, np.ndarray]
    edge_feats: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        ei = self.edge_index
        if ei.shape != (2, self.num_edges) or ei.dtype != np.int64:
            raise BrepGraphError(f"edge_index must be int64 [2, {self.num_edges}], got {ei.shape}")
        if self.num_edges and (ei.min() < 0 or ei.max() >= self.num_faces or np.any(ei[0] >= ei[1])):
            raise BrepGraphError("edge_index must satisfy 0 <= src < dst < num_faces")
        for kind, feats, n in (
            ("node", self.node_feats, self.num_faces),
            ("edge", self.edge_feats, self.num_edges),
        ):
            for name, arr in feats.items():
                if arr.shape[:1] != (n,):
                    raise BrepGraphError(f"{kind} feature '{name}' has {arr.shape[:1]} rows, want {n}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "num_faces": self.num_faces,
            "num_edges": self.num_edges,
            "edge_index": self.edge_index,
            "node_feats": dict(self.node_feats),
            "edge_feats": dict(self.edge_feats),
            "meta": self.meta,
        }

    def to_pyg(self, x=None, edge_attr=None):
        """PyTorch Geometric ``Data`` (needs the ``pyg`` extra). See :func:`brepgraph.export.pyg.to_pyg`."""
        from .export.pyg import to_pyg

        return to_pyg(self, x=x, edge_attr=edge_attr)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a compressed ``.npz`` archive (loadable with ``allow_pickle=False``)."""
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.array(self.schema_version),
            "edge_index": self.edge_index,
            "meta": np.array(json.dumps(self.meta, sort_keys=True)),
        }
        arrays.update({f"node/{k}": v for k, v in self.node_feats.items()})
        arrays.update({f"edge/{k}": v for k, v in self.edge_feats.items()})
        with open(path, "wb") as fh:  # file object: numpy must not append '.npz' to the name
            np.savez_compressed(fh, **arrays)  # type: ignore[arg-type]  # numpy stubs type **kwds as bool

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> BrepGraph:
        with np.load(path, allow_pickle=False) as z:
            version = str(z["schema_version"])
            if version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
                raise SchemaVersionError(
                    f"{path}: schema {version} is incompatible with this library ({SCHEMA_VERSION})"
                )
            node = {k[5:]: z[k] for k in z.files if k.startswith("node/")}
            edge = {k[5:]: z[k] for k in z.files if k.startswith("edge/")}
            edge_index = z["edge_index"].astype(np.int64, copy=False)
            meta = json.loads(str(z["meta"]))
        g = cls(
            num_faces=int(next(iter(node.values())).shape[0]) if node else int(meta.get("num_faces", 0)),
            num_edges=int(edge_index.shape[1]),
            edge_index=edge_index,
            node_feats=node,
            edge_feats=edge,
            meta=meta,
            schema_version=version,
        )
        g.validate()
        return g

    def __repr__(self) -> str:
        def rows(feats):
            return "".join(f"\n    {k}: {tuple(v.shape)} {v.dtype}" for k, v in feats.items())

        src = self.meta.get("source")
        return (
            f"BrepGraph(source={src!r}, faces={self.num_faces}, edges={self.num_edges}, "
            f"schema={self.schema_version}\n  node_feats:{rows(self.node_feats)}"
            f"\n  edge_feats:{rows(self.edge_feats)}\n)"
        )
