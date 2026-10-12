"""Differences between OCP releases (7.9 vs 8.x), kept in one place."""

from __future__ import annotations

from importlib import metadata

from OCP.TopoDS import TopoDS

try:  # OCP 7.x
    from OCP.TopTools import (  # noqa: F401
        TopTools_IndexedDataMapOfShapeListOfShape as ShapeAncestorMap,
    )
    from OCP.TopTools import (  # noqa: F401
        TopTools_IndexedMapOfShape as ShapeIndexedMap,
    )
except ImportError:  # OCP 8.x exposes typed NCollection instances instead
    from OCP.collections import (  # noqa: F401
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeAncestorMap,
    )
    from OCP.collections import (  # noqa: F401
        IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeIndexedMap,
    )


def _caster(name: str):
    fn = getattr(TopoDS, f"{name}_s", None) or getattr(TopoDS, name)
    return fn


to_face = _caster("Face")
to_edge = _caster("Edge")
to_solid = _caster("Solid")


def ocp_version() -> str:
    for dist in ("cadquery-ocp", "cadquery-ocp-novtk"):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return "unknown"
