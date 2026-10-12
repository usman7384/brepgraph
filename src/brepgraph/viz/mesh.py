"""Re-load the source of a graph, check it matches the stored arrays, and tessellate it."""

from __future__ import annotations

import numpy as np
from OCP.BRepTools import BRepTools

from .. import io, meshing
from ..context import Context
from ..errors import BrepGraphError
from ..topology import build_topology


def reload_context(graph, source=None) -> Context:
    """Context of the solid a graph was built from.

    ``source`` (a path or OCP shape) overrides ``graph.meta['source_path']``. The reload uses the
    healing / unify / normalize settings recorded in ``meta``, then compares face count and the
    stored ``area`` / ``centroid`` arrays so a changed face order is caught instead of mis-coloured.
    """
    if source is None:
        source = graph.meta.get("source_path")
        if not source:
            raise BrepGraphError(
                "this graph has no source_path in meta (built from an in-memory shape, or loaded from an "
                "archive saved without it): pass source=<path or shape>"
            )
    shape, _, _ = io.read_source(source)
    index = graph.meta.get("solid_index")
    if index is None:
        raw = io.extract_solid(shape)
    else:
        solids = io.extract_solids(shape)
        if index >= len(solids):
            raise BrepGraphError(f"graph is solid {index} but the source has only {len(solids)} solids")
        raw = solids[index]
    solid, _ = io.prepare_solid(
        raw,
        unify_faces=bool(graph.meta.get("unify_faces", False)),
        normalize=bool(graph.meta.get("normalize", False)),
    )
    ctx = Context(build_topology(solid))
    if ctx.num_faces != graph.num_faces:
        raise BrepGraphError(
            f"source has {ctx.num_faces} faces but the graph has {graph.num_faces}: wrong source or changed settings"
        )
    if "area" in graph.node_feats:
        area = np.array([ctx.face_props(i).Mass() for i in range(ctx.num_faces)])
        if not np.allclose(area, graph.node_feats["area"], rtol=1e-4, atol=1e-6):
            raise BrepGraphError("stored face areas do not match the source: face order differs")
    if "centroid" in graph.node_feats:
        cen = np.array(
            [[c.X(), c.Y(), c.Z()] for c in (ctx.face_props(i).CentreOfMass() for i in range(ctx.num_faces))]
        )
        scale = max(ctx.diagonal, 1e-12)
        if not np.allclose(cen, graph.node_feats["centroid"], atol=1e-4 * scale):
            raise BrepGraphError("stored face centroids do not match the source: face order differs")
    return ctx


def tessellate(topology, linear_ratio: float = 0.002, angular: float = 0.5):
    """Triangulate all faces.

    Returns ``(vertices[N,3], triangles[M,3], face_id_per_triangle[M], face_id_per_vertex[N])``.
    ``linear_ratio`` is the linear deflection relative to the bounding-box diagonal (0.2 % default).
    Faces are not welded: every face owns its vertices, which keeps per-face colours crisp.
    """
    meshing.mesh_solid(topology.solid, linear_ratio, angular)
    verts, tris, tri_face, vert_face, offset = [], [], [], [], 0
    for fid, face in enumerate(topology.faces):
        mesh = meshing.face_triangulation(face, orient=True)
        if mesh is None:
            continue
        v, t = mesh
        verts.append(v)
        tris.append(t + offset)
        tri_face.append(np.full(len(t), fid, dtype=np.int64))
        vert_face.append(np.full(len(v), fid, dtype=np.int64))
        offset += len(v)
    BRepTools.Clean_s(topology.solid)  # do not leave the mesh on a caller-supplied shape
    if not verts:
        raise BrepGraphError("tessellation produced no triangles")
    return np.vstack(verts), np.vstack(tris), np.concatenate(tri_face), np.concatenate(vert_face)
