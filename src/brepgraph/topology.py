"""Stage 2: deterministic face/edge numbering, edge->face map, edge classification.

Face and edge ids are the positions in an indexed map built with ``TopExp.MapShapes`` so the same
file and OCP version always give the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp

from ._compat import ShapeAncestorMap, ShapeIndexedMap, to_edge, to_face
from .errors import UnsupportedShapeError


@dataclass
class Topology:
    solid: Any
    face_map: Any
    edge_map: Any
    faces: list  # TopoDS_Face, index = face id
    edges: list  # TopoDS_Edge, every B-rep edge (index = position in edge_map - 1)
    edge_ids: np.ndarray  # [E] retained edges -> index into ``edges``
    edge_index: np.ndarray  # [2, E] int64 face ids, src < dst
    n_seam: int
    n_degenerated: int

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def num_edges(self) -> int:
        return int(self.edge_ids.shape[0])


def build_topology(solid) -> Topology:
    face_map, edge_map = ShapeIndexedMap(), ShapeIndexedMap()
    TopExp.MapShapes_s(solid, TopAbs_FACE, face_map)
    TopExp.MapShapes_s(solid, TopAbs_EDGE, edge_map)
    ancestors = ShapeAncestorMap()
    TopExp.MapShapesAndAncestors_s(solid, TopAbs_EDGE, TopAbs_FACE, ancestors)

    faces = [to_face(face_map.FindKey(i)) for i in range(1, face_map.Extent() + 1)]
    edges = [to_edge(edge_map.FindKey(i)) for i in range(1, edge_map.Extent() + 1)]

    kept, pairs = [], []
    n_seam = n_deg = 0
    for ei, edge in enumerate(edges):
        if BRep_Tool.Degenerated_s(edge):
            n_deg += 1
            continue
        owners = [face_map.FindIndex(f) - 1 for f in ancestors.FindFromKey(edge)]
        distinct = sorted(set(owners))
        if len(owners) == 1:
            raise UnsupportedShapeError(f"edge {ei} borders a single face (free edge: open shell)")
        if len(owners) > 2 or len(distinct) > 2:
            raise UnsupportedShapeError(f"edge {ei} is shared by {len(owners)} faces (non-manifold)")
        if len(distinct) == 1:  # same face on both sides: seam
            n_seam += 1
            continue
        kept.append(ei)
        pairs.append(distinct)

    edge_index = np.asarray(pairs, dtype=np.int64).T.copy() if pairs else np.zeros((2, 0), dtype=np.int64)
    return Topology(
        solid=solid,
        face_map=face_map,
        edge_map=edge_map,
        faces=faces,
        edges=edges,
        edge_ids=np.asarray(kept, dtype=np.int64),
        edge_index=edge_index,
        n_seam=n_seam,
        n_degenerated=n_deg,
    )


def bounding_box(solid) -> list[float]:
    """``[xmin, ymin, zmin, xmax, ymax, zmax]`` without using any tessellation."""
    box = Bnd_Box()
    BRepBndLib.Add_s(solid, box, False)
    lo, hi = box.CornerMin(), box.CornerMax()
    return [lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()]
