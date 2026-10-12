"""Per-face / per-edge cache of OCP adaptors and properties handed to feature functions.

Feature functions receive a :class:`Context` instead of raw OCP shapes so that two features that
need the same adaptor or property object do not recompute it.
"""

from __future__ import annotations

import math

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.IntTools import IntTools_FClass2d
from OCP.TopAbs import TopAbs_EDGE, TopAbs_REVERSED, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer

from . import meshing
from .topology import Topology

#: nudge fractions (towards the parametric centre) tried at singular points (poles, apexes)
_NUDGES = (0.0, 1e-6, 1e-4, 1e-2)
_FACE_TOL = 1e-6


def _xyz(obj) -> np.ndarray:
    return np.array([obj.X(), obj.Y(), obj.Z()], dtype=np.float64)


class Context:
    def __init__(self, topology: Topology):
        self.topology = topology
        self._surface: dict[int, BRepAdaptor_Surface] = {}
        self._uv: dict[int, tuple[float, float, float, float]] = {}
        self._fprops: dict[int, GProp_GProps] = {}
        self._fclass: dict[int, IntTools_FClass2d] = {}
        self._curve: dict[int, BRepAdaptor_Curve] = {}
        self._eprops: dict[int, GProp_GProps] = {}
        self._face_edge_sign: dict[int, dict[int, int]] = {}
        self._dihedral: dict[int, tuple[float, float]] = {}
        self._jac: dict[int, float] = {}
        self._tris: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        self._meshed = False
        self._neighbors: list[set[int]] | None = None
        self._diag: float | None = None

    @property
    def num_faces(self) -> int:
        return self.topology.num_faces

    @property
    def num_edges(self) -> int:
        return self.topology.num_edges

    @property
    def diagonal(self) -> float:
        """Length of the whole solid's bounding-box diagonal."""
        if self._diag is None:
            self._diag = meshing.diagonal(self.topology.solid)
        return self._diag

    def neighbors(self, i: int) -> set[int]:
        """Ids of the faces sharing a retained edge with face ``i``."""
        if self._neighbors is None:
            self._neighbors = [set() for _ in range(self.num_faces)]
            for a, b in self.topology.edge_index.T:
                self._neighbors[int(a)].add(int(b))
                self._neighbors[int(b)].add(int(a))
        return self._neighbors[i]

    def num_wires(self, i: int) -> int:
        """Number of boundary loops of face ``i``."""
        exp, n = TopExp_Explorer(self.face(i), TopAbs_WIRE), 0
        while exp.More():
            n += 1
            exp.Next()
        return n

    def triangles(self, i: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Cached ``(vertices, triangles)`` of face ``i`` (meshes the solid on first use) or ``None``."""
        if not self._meshed:
            meshing.mesh_solid(self.topology.solid)
            self._meshed = True
        if i not in self._tris:
            self._tris[i] = meshing.face_triangulation(self.face(i))
        return self._tris[i]

    def release_mesh(self) -> None:
        """Remove the triangulation that :meth:`triangles` attached to the solid.

        Meshing mutates the OCP shape in place, which would otherwise leak into a shape the caller
        passed to ``load``.
        """
        if self._meshed:
            BRepTools.Clean_s(self.topology.solid)
            self._meshed = False
            self._tris.clear()

    def face(self, i: int):
        return self.topology.faces[i]

    def face_reversed(self, i: int) -> bool:
        return self.face(i).Orientation() == TopAbs_REVERSED

    def surface(self, i: int) -> BRepAdaptor_Surface:
        if i not in self._surface:
            self._surface[i] = BRepAdaptor_Surface(self.face(i), True)
        return self._surface[i]

    def uv_bounds(self, i: int) -> tuple[float, float, float, float]:
        if i not in self._uv:
            self._uv[i] = tuple(BRepTools.UVBounds_s(self.face(i)))
        return self._uv[i]

    def face_props(self, i: int) -> GProp_GProps:
        if i not in self._fprops:
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(self.face(i), props)
            self._fprops[i] = props
        return self._fprops[i]

    def _jacobian_ref(self, i: int) -> float:
        """Typical |dS/du x dS/dv| of the face: area / parametric extent (0 if degenerate)."""
        if i not in self._jac:
            u0, u1, v0, v1 = self.uv_bounds(i)
            extent = (u1 - u0) * (v1 - v0)
            self._jac[i] = self.face_props(i).Mass() / extent if extent > 0 else 0.0
        return self._jac[i]

    def classifier(self, i: int) -> IntTools_FClass2d:
        """Point-in-trimmed-face classifier in the face's (u, v) space."""
        if i not in self._fclass:
            self._fclass[i] = IntTools_FClass2d(self.face(i), _FACE_TOL)
        return self._fclass[i]

    def sample_surface(self, i: int, u: float, v: float):
        """Return ``(point[3], outward_unit_normal[3] | None)`` at parameters ``(u, v)``.

        The normal is flipped when the face is reversed in the B-rep. At singular points (sphere
        poles, cone apexes) the normal is taken a vanishing distance towards the parametric centre;
        ``None`` means no usable normal was found.
        """
        surf = self.surface(i)
        p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
        surf.D1(u, v, p, du, dv)
        point = _xyz(p)
        u0, u1, v0, v1 = self.uv_bounds(i)
        uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        # absolute floor: noise next to a pole/apex gives a tiny but sign-flipped cross product
        floor = 1e-8 * self._jacobian_ref(i)
        for frac in _NUDGES:
            if frac:
                surf.D1(u + frac * (uc - u), v + frac * (vc - v), p, du, dv)
            # scalar arithmetic: np.cross / np.linalg.norm on 3-vectors cost more than the OCP call
            ax, ay, az = du.X(), du.Y(), du.Z()
            bx, by, bz = dv.X(), dv.Y(), dv.Z()
            nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
            norm = math.sqrt(nx * nx + ny * ny + nz * nz)
            scale = math.sqrt(ax * ax + ay * ay + az * az) * math.sqrt(bx * bx + by * by + bz * bz)
            if scale > 0.0 and norm > 1e-9 * scale and norm > floor:
                sign = -1.0 if self.face_reversed(i) else 1.0
                k = sign / norm
                return point, np.array([nx * k, ny * k, nz * k])
        return point, None

    def edge(self, i: int):
        return self.topology.edges[int(self.topology.edge_ids[i])]

    def curve(self, i: int) -> BRepAdaptor_Curve:
        if i not in self._curve:
            self._curve[i] = BRepAdaptor_Curve(self.edge(i))
        return self._curve[i]

    def edge_props(self, i: int) -> GProp_GProps:
        if i not in self._eprops:
            props = GProp_GProps()
            BRepGProp.LinearProperties_s(self.edge(i), props)
            self._eprops[i] = props
        return self._eprops[i]

    def edge_faces(self, i: int) -> tuple[int, int]:
        a, b = self.topology.edge_index[:, i]
        return int(a), int(b)

    def edge_reversed(self, i: int) -> bool:
        """Orientation of the edge as stored in the solid's edge map."""
        return self.edge(i).Orientation() == TopAbs_REVERSED

    def _edge_sign_in_face(self, face_id: int, edge_id: int) -> int:
        """+1 / -1: edge traversed along / against its curve parameter in the face's outer view."""
        if face_id not in self._face_edge_sign:
            signs: dict[int, int] = {}
            exp = TopExp_Explorer(self.face(face_id), TopAbs_EDGE)
            while exp.More():
                e = exp.Current()
                eid = self.topology.edge_map.FindIndex(e) - 1
                signs[eid] = -1 if e.Orientation() == TopAbs_REVERSED else 1
                exp.Next()
            self._face_edge_sign[face_id] = signs
        return self._face_edge_sign[face_id][edge_id]

    def dihedral(self, i: int) -> tuple[float, float]:
        """``(angle, s)`` for retained edge ``i``.

        ``angle`` in [0, pi] is the angle between the two outward face normals at the edge midpoint.
        ``s = n_a . inward_b`` where ``inward_b = n_b x t_b`` points into face b (t_b is the edge
        tangent as traversed in face b, counter-clockwise seen from outside). ``s < 0`` means the
        far face bends away from face a's tangent plane, i.e. a convex edge.
        """
        if i in self._dihedral:
            return self._dihedral[i]
        edge = self.edge(i)
        fa, fb = self.edge_faces(i)
        curve = self.curve(i)
        tm = 0.5 * (curve.FirstParameter() + curve.LastParameter())
        normals = []
        for f in (fa, fb):
            pc = BRepAdaptor_Curve2d(edge, self.face(f))
            uv = pc.Value(0.5 * (pc.FirstParameter() + pc.LastParameter()))
            _, n = self.sample_surface(f, uv.X(), uv.Y())
            if n is None:
                raise ValueError(f"no surface normal at the midpoint of an edge on face {f}")
            normals.append(n)
        na, nb = normals
        p, d = gp_Pnt(), gp_Vec()
        curve.D1(tm, p, d)
        t = _xyz(d)
        norm_t = np.linalg.norm(t)
        t = t / norm_t if norm_t > 0 else t
        t *= self._edge_sign_in_face(fb, int(self.topology.edge_ids[i]))
        angle = float(np.arccos(np.clip(np.dot(na, nb), -1.0, 1.0)))
        s = float(np.dot(na, np.cross(nb, t)))
        self._dihedral[i] = (angle, s)
        return self._dihedral[i]
