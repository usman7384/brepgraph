"""``show`` (whole part coloured by a feature) and ``show_face`` (single-face debug view)."""

from __future__ import annotations

import numpy as np

from ..errors import BrepGraphError, MissingExtraError
from .mesh import reload_context, tessellate

PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
    "#86bcb6",
]
CATEGORY_NAMES = {
    "surface_type": [
        "plane",
        "cylinder",
        "cone",
        "sphere",
        "torus",
        "bezier",
        "bspline",
        "revolution",
        "extrusion",
        "offset",
        "other",
    ],
}
CONVEXITY_COLORS = {0: ("concave", "#d62728"), 1: ("convex", "#2ca02c"), 2: ("smooth", "#1f77b4")}
_SCENE = {"aspectmode": "data"}
_MARGIN = {"l": 0, "r": 0, "t": 30, "b": 0}


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise MissingExtraError("visualization needs plotly: pip install 'brepgraph[viz]'") from exc
    return go


def _face_values(graph, color_by):
    """Resolve ``color_by`` to ``(values[F], kind, label, names)``; kind is 'categorical' or 'scalar'."""
    options = [n for n, a in graph.node_feats.items() if a.ndim == 1] + [
        f"({n!r}, <index 0..{a.shape[1] - 1}>)" for n, a in graph.node_feats.items() if a.ndim == 2
    ]
    if "uv_grid" in graph.node_feats:
        options.append("('uv_grid', 'mask_fraction')")

    def fail(msg):
        raise BrepGraphError(f"{msg}; valid color_by options: {options}")

    if isinstance(color_by, str):
        if color_by not in graph.node_feats:
            fail(f"unknown node feature '{color_by}'")
        arr = graph.node_feats[color_by]
        if arr.ndim != 1:
            fail(f"'{color_by}' has shape {arr.shape[1:]} per face and cannot be coloured directly")
        if arr.dtype.kind in "iu":
            return arr, "categorical", color_by, CATEGORY_NAMES.get(color_by)
        return arr.astype(float), "scalar", color_by, None
    name, sel = color_by
    if name not in graph.node_feats:
        fail(f"unknown node feature '{name}'")
    arr = graph.node_feats[name]
    if name == "uv_grid" and sel == "mask_fraction":
        return arr[:, 6].reshape(len(arr), -1).mean(axis=1), "scalar", "uv mask fraction", None
    if arr.ndim == 2 and isinstance(sel, (int, np.integer)) and 0 <= sel < arr.shape[1]:
        return arr[:, sel].astype(float), "scalar", f"{name}[{sel}]", None
    fail(f"cannot select {sel!r} from '{name}' of shape {arr.shape}")


def _edge_polylines(ctx, samples: int = 24) -> list[np.ndarray]:
    polylines = []
    for i in range(ctx.num_edges):
        curve = ctx.curve(i)
        params = np.linspace(curve.FirstParameter(), curve.LastParameter(), samples)
        points = [curve.Value(float(t)) for t in params]
        polylines.append(np.array([[p.X(), p.Y(), p.Z()] for p in points]))
    return polylines


def _line_trace(go, segments: list[np.ndarray], name: str, color: str, width: int):
    """One Scatter3d trace for several polylines, separated by ``None`` gaps."""
    coords: list[list] = [[], [], []]
    for seg in segments:
        for axis in range(3):
            coords[axis] += [*seg[:, axis], None]
    return go.Scatter3d(
        x=coords[0],
        y=coords[1],
        z=coords[2],
        mode="lines",
        name=name,
        line={"color": color, "width": width},
    )


def _surface_traces(go, ctx, values, kind, label, names, linear_ratio, angular):
    verts, tris, tri_face, vert_face = tessellate(ctx.topology, linear_ratio, angular)
    mesh = {
        "x": verts[:, 0],
        "y": verts[:, 1],
        "z": verts[:, 2],
        "i": tris[:, 0],
        "j": tris[:, 1],
        "k": tris[:, 2],
        "flatshading": True,
        "hoverinfo": "skip",
        "lighting": {"ambient": 0.6, "diffuse": 0.7, "specular": 0.1},
        "name": "faces",
    }
    if kind == "scalar":
        return [
            go.Mesh3d(
                intensity=values[vert_face],
                intensitymode="vertex",
                colorscale="Viridis",
                cmin=float(values.min()),
                cmax=float(values.max()),
                colorbar={"title": label},
                **mesh,
            )
        ]
    categories = sorted({int(v) for v in values})
    color_of = {c: PALETTE[c % len(PALETTE)] for c in categories}
    traces = [go.Mesh3d(facecolor=[color_of[int(values[f])] for f in tri_face], **mesh)]
    for c in categories:  # empty traces only carry the legend entries
        text = names[c] if names and c < len(names) else f"{label}={c}"
        traces.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={"size": 8, "color": color_of[c]},
                name=text,
            )
        )
    return traces


def _edge_traces(go, ctx, convexity):
    lines = _edge_polylines(ctx)
    if convexity is None:
        return [_line_trace(go, lines, "edges", "black", 4)]
    traces = []
    for code, (name, color) in CONVEXITY_COLORS.items():
        selected = [lines[k] for k in np.flatnonzero(convexity == code)]
        if selected:
            traces.append(_line_trace(go, selected, f"edge: {name}", color, 4))
    return traces


def _graph_overlay_traces(go, graph):
    centroids = graph.node_feats["centroid"]
    segments = [centroids[[a, b]] for a, b in graph.edge_index.T]
    nodes = go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode="markers+text",
        text=[str(i) for i in range(len(centroids))],
        name="face nodes",
        marker={"size": 4, "color": "black"},
    )
    return [_line_trace(go, segments, "graph edges", "black", 2), nodes]


def show(
    g,
    color_by="surface_type",
    *,
    edges: bool = True,
    graph: bool = False,
    source=None,
    linear_ratio: float = 0.002,
    angular: float = 0.5,
):
    """Plotly figure of the part with faces coloured by a node feature.

    ``color_by`` is a 1-D node feature name (integer features get a discrete palette and a legend,
    float features a colour bar), ``(name, k)`` to pick one column of a 2-D node feature, or
    ``("uv_grid", "mask_fraction")``. ``edges=True`` overlays the retained B-rep edges coloured by
    convexity; ``graph=True`` overlays the face-adjacency graph. The part is re-loaded from
    ``g.meta['source_path']`` (or ``source=``) and checked against the stored areas and centroids.
    ``linear_ratio`` is the tessellation deflection relative to the bounding-box diagonal.
    """
    go = _plotly()
    values, kind, label, names = _face_values(g, color_by)
    ctx = reload_context(g, source)
    fig = go.Figure()
    for trace in _surface_traces(go, ctx, values, kind, label, names, linear_ratio, angular):
        fig.add_trace(trace)
    if edges and ctx.num_edges:
        for trace in _edge_traces(go, ctx, g.edge_feats.get("convexity")):
            fig.add_trace(trace)
    if graph and "centroid" in g.node_feats and g.num_edges:
        for trace in _graph_overlay_traces(go, g):
            fig.add_trace(trace)
    fig.update_layout(
        scene=_SCENE, margin=_MARGIN, title=f"{g.meta.get('source') or 'part'}: coloured by {label}"
    )
    return fig


def show_face(graph, face: int, *, normal_scale: float | None = None):
    """Debug view of one face's UV samples: inside the trim (blue) vs outside (grey), plus normals."""
    go = _plotly()
    if "uv_grid" not in graph.node_feats:
        raise BrepGraphError("show_face needs the 'uv_grid' node feature")
    if not 0 <= face < graph.num_faces:
        raise BrepGraphError(f"face must be in [0, {graph.num_faces}), got {face}")
    grid = graph.node_feats["uv_grid"][face].reshape(7, -1)
    xyz, normals, inside = grid[:3], grid[3:6], grid[6] > 0.5
    span = float(np.ptp(xyz, axis=1).max()) or 1.0
    fig = go.Figure()
    for flag, name, color in (
        (True, "inside trim (mask=1)", "#1f77b4"),
        (False, "outside trim (mask=0)", "#bbbbbb"),
    ):
        sel = inside == flag
        if sel.any():
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[0, sel],
                    y=xyz[1, sel],
                    z=xyz[2, sel],
                    mode="markers",
                    name=name,
                    marker={"size": 4, "color": color},
                )
            )
    if inside.any():
        fig.add_trace(
            go.Cone(
                x=xyz[0, inside],
                y=xyz[1, inside],
                z=xyz[2, inside],
                u=normals[0, inside],
                v=normals[1, inside],
                w=normals[2, inside],
                anchor="tail",
                showscale=False,
                sizemode="absolute",
                sizeref=normal_scale or 0.08 * span,
                colorscale=[[0, "#d62728"], [1, "#d62728"]],
                name="outward normals",
            )
        )
    surface_type = graph.node_feats.get("surface_type")
    extra = f", surface_type={int(surface_type[face])}" if surface_type is not None else ""
    fig.update_layout(scene=_SCENE, margin=_MARGIN, title=f"face {face}{extra}")
    return fig
