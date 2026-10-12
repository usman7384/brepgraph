"""Face (node) features: UV-Net style."""

from __future__ import annotations

import numpy as np
from OCP.gp import gp_Pnt2d
from OCP.TopAbs import TopAbs_IN, TopAbs_ON

from ..registry import node_feature


@node_feature("surface_type", dtype="int64", shape=())
def surface_type(ctx, i):
    """Index of OCP's GeomAbs_SurfaceType (0 plane, 1 cylinder, 2 cone, 3 sphere, 4 torus, 5 Bezier, 6 B-spline, 7 revolution, 8 extrusion, 9 offset, 10 other)."""
    return np.int64(int(ctx.surface(i).GetType()))


@node_feature("area", dtype="float32", shape=())
def area(ctx, i):
    """Face area in model units squared."""
    return np.float32(ctx.face_props(i).Mass())


@node_feature("centroid", dtype="float32", shape=(3,))
def centroid(ctx, i):
    """Centre of mass of the face."""
    c = ctx.face_props(i).CentreOfMass()
    return np.array([c.X(), c.Y(), c.Z()], dtype=np.float32)


@node_feature(
    "uv_grid",
    dtype="float32",
    shape=lambda p: (7, p["size"], p["size"]),
    defaults={"size": 10},
)
def uv_grid(ctx, i, size):
    """Channels-first UV sample grid: x, y, z, nx, ny, nz (outward unit normal), mask (1 = inside the trimmed face). Axis 1 is u, axis 2 is v."""
    u0, u1, v0, v1 = ctx.uv_bounds(i)
    clf = ctx.classifier(i)
    out = np.zeros((7, size, size), dtype=np.float32)
    for a, u in enumerate(np.linspace(u0, u1, size)):
        for b, v in enumerate(np.linspace(v0, v1, size)):
            point, normal = ctx.sample_surface(i, float(u), float(v))
            out[0:3, a, b] = point
            if normal is not None:
                out[3:6, a, b] = normal
                state = clf.Perform(gp_Pnt2d(float(u), float(v)))
                out[6, a, b] = 1.0 if state in (TopAbs_IN, TopAbs_ON) else 0.0
    return out


@node_feature("num_loops", dtype="int64", shape=())
def num_loops(ctx, i):
    """Number of boundary loops (wires) of the face; a face with holes has more than one."""
    return np.int64(ctx.num_wires(i))


@node_feature("num_adjacent_faces", dtype="int64", shape=())
def num_adjacent_faces(ctx, i):
    """Number of distinct faces sharing a retained edge with this face."""
    return np.int64(len(ctx.neighbors(i)))
