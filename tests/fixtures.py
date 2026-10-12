"""Analytic shape builders and STEP writing helpers (OCP only)."""

from __future__ import annotations

from pathlib import Path

from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_Transform
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeTorus,
)
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shell

_to_edge = getattr(TopoDS, "Edge_s", None) or TopoDS.Edge


def _z_axis_at(x, y, z):
    return gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0, 0, 1))


def _translation(x, y, z):
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(x, y, z))
    return t


def box(dx=10.0, dy=20.0, dz=30.0):
    return BRepPrimAPI_MakeBox(dx, dy, dz).Shape()


def cylinder(r=5.0, h=10.0):
    return BRepPrimAPI_MakeCylinder(r, h).Shape()


def sphere(r=5.0):
    return BRepPrimAPI_MakeSphere(r).Shape()


def cone(r1=5.0, r2=0.0, h=10.0):
    return BRepPrimAPI_MakeCone(r1, r2, h).Shape()


def torus(major=10.0, minor=2.0):
    return BRepPrimAPI_MakeTorus(major, minor).Shape()


def half_cylinder(r=5.0, h=10.0):
    """Cylinder cut along its axis: the flat face and the curved face share two edges."""
    tool = BRepPrimAPI_MakeBox(gp_Pnt(0, -2 * r, -1), 2 * r, 4 * r, h + 2).Shape()
    return BRepAlgoAPI_Cut(cylinder(r, h), tool).Shape()


def pocket_box():
    """20x20x10 block with a 10x10x5 pocket cut from the top."""
    base = BRepPrimAPI_MakeBox(20.0, 20.0, 10.0).Shape()
    tool = BRepPrimAPI_MakeBox(gp_Pnt(5, 5, 5), 10.0, 10.0, 10.0).Shape()
    return BRepAlgoAPI_Cut(base, tool).Shape()


def filleted_box(radius=2.0, side=20.0):
    """Cube with one edge filleted."""
    base = BRepPrimAPI_MakeBox(side, side, side).Shape()
    maker = BRepFilletAPI_MakeFillet(base)
    maker.Add(radius, _to_edge(TopExp_Explorer(base, TopAbs_EDGE).Current()))
    return maker.Shape()


def tiny_fillet():
    """50 mm cube with a 0.02 mm fillet: a sliver face next to large faces."""
    return filleted_box(0.02, side=50.0)


def plate_with_hole(side=100.0, thick=10.0, r=20.0):
    base = BRepPrimAPI_MakeBox(side, side, thick).Shape()
    hole = BRepPrimAPI_MakeCylinder(_z_axis_at(side / 2, side / 2, -1.0), r, thick + 2.0).Shape()
    return BRepAlgoAPI_Cut(base, hole).Shape()


def holey_plate(n=5, pitch=10.0, r=2.0, thick=5.0):
    """Plate with an n x n grid of through holes, cut in a single boolean operation."""
    side = pitch * (n + 1)
    base = BRepPrimAPI_MakeBox(side, side, thick).Shape()
    tools, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(tools)
    for i in range(n):
        for j in range(n):
            axis = _z_axis_at(pitch * (i + 1), pitch * (j + 1), -1.0)
            builder.Add(tools, BRepPrimAPI_MakeCylinder(axis, r, thick + 2.0).Shape())
    return BRepAlgoAPI_Cut(base, tools).Shape()


def nurbs(shape):
    """Convert every surface and curve to B-spline."""
    return BRepBuilderAPI_NurbsConvert(shape, True).Shape()


def rigid_motion(shape, angle=0.7, move=(3.0, -4.0, 5.0)):
    rotation = gp_Trsf()
    rotation.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 2, 3)), angle)
    motion = _translation(*move)
    motion.Multiply(rotation)
    return BRepBuilderAPI_Transform(shape, motion, True).Shape()


def scaled(shape, s):
    t = gp_Trsf()
    t.SetScale(gp_Pnt(0, 0, 0), s)
    return BRepBuilderAPI_Transform(shape, t, True).Shape()


def two_boxes():
    """A compound of two separate solids."""
    compound, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, box(10, 10, 10))
    moved = BRepBuilderAPI_Transform(box(10, 10, 10), _translation(50, 0, 0), True).Shape()
    builder.Add(compound, moved)
    return compound


def open_shell():
    """A box with one face removed: a lone open shell."""
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    faces = TopExp_Explorer(box(), TopAbs_FACE)
    faces.Next()  # skip the first face
    while faces.More():
        builder.Add(shell, faces.Current())
        faces.Next()
    return shell


def has_triangulation(shape) -> bool:
    """True if any face of ``shape`` carries a mesh."""
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    faces = TopExp_Explorer(shape, TopAbs_FACE)
    while faces.More():
        face = getattr(TopoDS, "Face_s", None) or TopoDS.Face
        if BRep_Tool.Triangulation_s(face(faces.Current()), TopLoc_Location()) is not None:
            return True
        faces.Next()
    return False


def write_step(shape, path: Path) -> Path:
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"could not write {path}")
    return path
