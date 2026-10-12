"""Tessellation helpers shared by area-uniform sampling (pair features) and visualization."""

from __future__ import annotations

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location

from .topology import bounding_box


def diagonal(solid) -> float:
    b = bounding_box(solid)
    return float(np.linalg.norm(np.array(b[3:]) - np.array(b[:3])))


def mesh_solid(solid, linear_ratio: float = 0.001, angular: float = 0.3) -> None:
    """Triangulate every face in place. ``linear_ratio`` is relative to the bounding-box diagonal."""
    BRepMesh_IncrementalMesh(solid, linear_ratio * max(diagonal(solid), 1e-12), False, angular, True)


def face_triangulation(face, orient: bool = False):
    """``(vertices[n,3] float64, triangles[m,3] int64)`` in world coordinates, or ``None``.

    With ``orient=True`` triangle winding is flipped for reversed faces so normals computed from
    the winding point outward.
    """
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None or tri.NbTriangles() == 0:
        return None
    trsf = loc.Transformation()
    verts = np.empty((tri.NbNodes(), 3))
    for k in range(tri.NbNodes()):
        p = tri.Node(k + 1).Transformed(trsf)
        verts[k] = (p.X(), p.Y(), p.Z())
    tris = np.empty((tri.NbTriangles(), 3), dtype=np.int64)
    for k in range(tri.NbTriangles()):
        tris[k] = tri.Triangle(k + 1).Get()
    tris -= 1
    if orient and face.Orientation() == TopAbs_REVERSED:
        tris = tris[:, [0, 2, 1]]
    return verts, tris


def sample_triangles(verts: np.ndarray, tris: np.ndarray, n: int, rng: np.random.Generator):
    """``n`` points uniformly distributed by area over a triangle mesh -> ``[n, 3]``."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if total <= 0:
        return np.repeat(a[:1], n, axis=0)
    pick = np.searchsorted(np.cumsum(area) / total, rng.random(n), side="right")
    pick = np.minimum(pick, len(tris) - 1)
    r1, r2 = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
    return (1 - r1) * a[pick] + r1 * (1 - r2) * b[pick] + r1 * r2 * c[pick]
