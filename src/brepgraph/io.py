"""Stage 1: read a file, extract exactly one solid, validate and (optionally) heal it.

Units: the STEP reader converts to millimetres (OCCT default ``xstep.cascade.unit``). Nothing is
rescaled afterwards unless ``normalize=True`` is requested.
"""

from __future__ import annotations

import os
from pathlib import Path

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.IGESControl import IGESControl_Reader
from OCP.ShapeFix import ShapeFix_Shape
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS_Shape

from ._compat import ShapeIndexedMap, to_solid
from .errors import BrepGraphError, FileReadError, InvalidSolidError, UnsupportedShapeError

_STEP = {".step", ".stp"}
_IGES = {".iges", ".igs"}
_BREP = {".brep", ".brp"}


def _read_with(reader, path: Path) -> TopoDS_Shape:
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise FileReadError(f"{path}: the CAD reader reported failure (not a valid file?)")
    if reader.TransferRoots() == 0:
        raise FileReadError(f"{path}: no transferable entities found")
    shape = reader.OneShape()
    if shape.IsNull():
        raise FileReadError(f"{path}: reader produced an empty shape")
    return shape


def read_shape(path: str | os.PathLike[str]) -> TopoDS_Shape:
    """Read a STEP / IGES / BREP file into a ``TopoDS_Shape``."""
    p = Path(path)
    if not p.is_file():
        raise FileReadError(f"{p}: file not found")
    ext = p.suffix.lower()
    try:
        if ext in _STEP:
            return _read_with(STEPControl_Reader(), p)
        if ext in _IGES:
            return _read_with(IGESControl_Reader(), p)
        if ext in _BREP:
            shape = TopoDS_Shape()
            if not BRepTools.Read_s(shape, str(p), BRep_Builder()) or shape.IsNull():
                raise FileReadError(f"{p}: could not read BREP file")
            return shape
    except BrepGraphError:
        raise
    except Exception as exc:  # OCCT raises Standard_Failure subclasses
        raise FileReadError(f"{p}: {exc}") from exc
    raise FileReadError(f"{p}: unrecognised extension '{ext}' (use .step/.stp/.iges/.igs/.brep)")


def extract_solid(shape: TopoDS_Shape):
    """Return the single solid inside ``shape`` (unwrapping compounds)."""
    solids = ShapeIndexedMap()
    TopExp.MapShapes_s(shape, TopAbs_SOLID, solids)
    n = solids.Extent()
    if n == 0:
        raise UnsupportedShapeError("no solid found (open shell or surface body?)")
    if n > 1:
        raise UnsupportedShapeError(f"expected exactly one solid, found {n} (assembly/multi-body)")
    return to_solid(solids.FindKey(1))


def validate_and_heal(solid, *, heal: bool = True, unify_faces: bool = False):
    """Check validity, try ``ShapeFix_Shape`` once, optionally merge same-domain faces.

    Returns ``(solid, healed)``.
    """
    healed = False
    if not BRepCheck_Analyzer(solid).IsValid():
        if not heal:
            raise InvalidSolidError("solid failed BRepCheck_Analyzer and healing is disabled")
        fixer = ShapeFix_Shape(solid)
        fixer.Perform()
        try:
            fixed = extract_solid(fixer.Shape())
        except UnsupportedShapeError as exc:
            raise InvalidSolidError(f"healing did not produce a solid: {exc}") from exc
        if not BRepCheck_Analyzer(fixed).IsValid():
            raise InvalidSolidError("solid is still invalid after ShapeFix_Shape")
        solid, healed = fixed, True
    if unify_faces:
        unifier = ShapeUpgrade_UnifySameDomain(solid, True, True, True)
        unifier.Build()
        solid = extract_solid(unifier.Shape())
    return solid, healed


def center_and_scale(solid):
    """Translate the bounding-box centre to the origin and scale the longest side to 1."""
    box = Bnd_Box()
    BRepBndLib.Add_s(solid, box, False)
    lo, hi = box.CornerMin(), box.CornerMax()
    x0, y0, z0, x1, y1, z1 = lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()
    longest = max(x1 - x0, y1 - y0, z1 - z0)
    if longest <= 0:
        raise InvalidSolidError("solid has a degenerate bounding box")
    move = gp_Trsf()
    move.SetTranslation(gp_Vec(-(x0 + x1) / 2, -(y0 + y1) / 2, -(z0 + z1) / 2))
    scale = gp_Trsf()
    scale.SetScale(gp_Pnt(0, 0, 0), 1.0 / longest)
    scale.Multiply(move)  # scale * move: translate first
    return extract_solid(BRepBuilderAPI_Transform(solid, scale, True).Shape())


def load_solid(
    source,
    *,
    heal: bool = True,
    unify_faces: bool = False,
    normalize: bool = False,
):
    """Path or in-memory ``TopoDS_Shape`` -> validated single solid.

    Returns ``(solid, info)`` where ``info`` is a JSON-friendly dict for ``meta``.
    """
    if isinstance(source, TopoDS_Shape):
        shape, name, full = source, None, None
    else:
        shape, name, full = read_shape(source), Path(source).name, str(Path(source).resolve())
    solid, healed = validate_and_heal(extract_solid(shape), heal=heal, unify_faces=unify_faces)
    if normalize:
        solid = center_and_scale(solid)
    return solid, {
        "source": name,
        "source_path": full,
        "healed": healed,
        "unify_faces": unify_faces,
        "normalize": normalize,
    }
