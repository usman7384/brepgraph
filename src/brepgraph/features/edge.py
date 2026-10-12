"""Edge features: UV-Net style."""

from __future__ import annotations

import numpy as np
from OCP.gp import gp_Pnt, gp_Vec

from ..registry import edge_feature


@edge_feature("curve_type", dtype="int64", shape=())
def curve_type(ctx, i):
    """Index of OCP's GeomAbs_CurveType (0 line, 1 circle, 2 ellipse, 3 hyperbola, 4 parabola, 5 Bezier, 6 B-spline, 7 offset, 8 other)."""
    return np.int64(int(ctx.curve(i).GetType()))


@edge_feature("length", dtype="float32", shape=())
def length(ctx, i):
    """Edge length in model units."""
    return np.float32(ctx.edge_props(i).Mass())


@edge_feature(
    "u_grid",
    dtype="float32",
    shape=lambda p: (6, p["size"]),
    defaults={"size": 10},
)
def u_grid(ctx, i, size):
    """Channels-first samples along the edge: x, y, z, tx, ty, tz (unit tangent). Points run along the edge's own B-rep orientation (reversed edges are walked from the last parameter to the first and the tangent follows)."""
    curve = ctx.curve(i)
    ts = np.linspace(curve.FirstParameter(), curve.LastParameter(), size)
    flip = ctx.edge_reversed(i)
    if flip:
        ts = ts[::-1]
    out = np.zeros((6, size), dtype=np.float32)
    p, d = gp_Pnt(), gp_Vec()
    for k, t in enumerate(ts):
        curve.D1(float(t), p, d)
        tan = np.array([d.X(), d.Y(), d.Z()])
        norm = np.linalg.norm(tan)
        if norm > 0:
            tan = tan / norm * (-1.0 if flip else 1.0)
        out[0:3, k] = (p.X(), p.Y(), p.Z())
        out[3:6, k] = tan
    return out


@edge_feature("dihedral_angle", dtype="float32", shape=())
def dihedral_angle(ctx, i):
    """Angle in radians (0..pi) between the two adjacent faces' outward normals at the edge midpoint."""
    return np.float32(ctx.dihedral(i)[0])


@edge_feature("convexity", dtype="int64", shape=(), defaults={"tol": 0.01})
def convexity(ctx, i, tol):
    """0 concave, 1 convex, 2 smooth (dihedral angle below ``tol`` radians)."""
    angle, s = ctx.dihedral(i)
    if angle < tol:
        return np.int64(2)
    return np.int64(1 if s < 0 else 0)
