"""``show_graph``: the face-adjacency graph on its own (nodes = faces, edges = shared B-rep edges).

Needs only the :class:`BrepGraph`; the source file is not re-loaded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import BrepGraphError
from .show import CONVEXITY_COLORS, PALETTE, _face_values, _plotly

LAYOUTS = ("spring", "spring3d", "centroid")


def spring_layout(n: int, edge_index: np.ndarray, dim: int = 2, iterations: int = 300, seed: int = 0):
    """Fruchterman-Reingold force-directed layout in ``dim`` dimensions, seeded and deterministic.

    Returns ``[n, dim]`` positions scaled to fit in [-1, 1]. Unconnected components stay near each
    other thanks to a weak pull towards the origin.
    """
    if n <= 1:
        return np.zeros((n, dim))
    pos = np.random.default_rng(seed).random((n, dim)) - 0.5
    k = 1.0 / np.sqrt(n)
    temp = 0.1
    a, b = edge_index
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=-1), 1e-9)
        disp = (delta * (k * k / dist**2)[..., None]).sum(axis=1)  # repulsion between all pairs
        if a.size:
            d = pos[a] - pos[b]
            pull = d * (np.maximum(np.linalg.norm(d, axis=1), 1e-9) / k)[:, None]  # attraction on edges
            np.add.at(disp, a, -pull)
            np.add.at(disp, b, pull)
        disp -= 0.05 * pos  # gravity
        length = np.maximum(np.linalg.norm(disp, axis=1), 1e-9)
        pos += disp / length[:, None] * np.minimum(length, temp)[:, None]
        temp = max(temp * 0.98, 1e-3)
    pos -= pos.mean(axis=0)
    return pos / max(np.abs(pos).max(), 1e-9)


def _positions(graph, layout: str, seed: int):
    if layout == "spring":
        return spring_layout(graph.num_faces, graph.edge_index, 2, seed=seed)
    if layout == "spring3d":
        return spring_layout(graph.num_faces, graph.edge_index, 3, seed=seed)
    if layout == "centroid":
        if "centroid" not in graph.node_feats:
            raise BrepGraphError("layout='centroid' needs the 'centroid' node feature")
        return graph.node_feats["centroid"].astype(float)
    raise BrepGraphError(f"unknown layout '{layout}'; valid layouts: {list(LAYOUTS)}")


def _fmt(arr: np.ndarray, i: int) -> str:
    v = arr[i]
    return str(int(v)) if arr.dtype.kind in "iu" else f"{float(v):.4g}"


def _hover(header: str, feats: dict, i: int) -> str:
    rows = [f"{name}: {_fmt(arr, i)}" for name, arr in feats.items() if arr.ndim == 1]
    return "<br>".join([header, *rows])


def _fan_out_parallel_edges(pos: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Midpoint of each edge, with parallel edges (same two faces) bent sideways so all stay visible."""
    dim = pos.shape[1]
    mids = (pos[edge_index[0]] + pos[edge_index[1]]) / 2
    groups: dict[tuple[int, int], list[int]] = {}
    for k, (a, b) in enumerate(edge_index.T):
        groups.setdefault((int(a), int(b)), []).append(k)
    span = float(np.ptp(pos, axis=0).max()) or 1.0
    for (a, b), ks in groups.items():
        if len(ks) < 2:
            continue
        d = pos[b] - pos[a]
        side = np.array([-d[1], d[0]]) if dim == 2 else np.cross(d, [0.0, 0.0, 1.0])
        if np.linalg.norm(side) < 1e-9:
            side = np.eye(dim)[0]
        side = side / np.linalg.norm(side)
        for r, k in enumerate(ks):
            mids[k] += side * (r - (len(ks) - 1) / 2) * 0.05 * span
    return mids


def show_graph(
    g,
    color_by="surface_type",
    *,
    layout: str = "spring",
    size_by: str | None = "area",
    labels: bool = True,
    seed: int = 0,
):
    """Plotly figure of the face-adjacency graph.

    Nodes are faces: coloured by ``color_by`` (same options as :func:`brepgraph.viz.show`) and sized
    by the scalar node feature ``size_by`` (``None`` for equal sizes). Edges are coloured by
    convexity (red concave, green convex, blue smooth) when that feature is present. Hovering a node
    or an edge midpoint lists all its scalar features. ``layout`` is ``"spring"`` (2-D
    force-directed), ``"spring3d"``, or ``"centroid"`` (3-D, nodes at the face centroids, so the
    graph sits inside the part's shape). Parallel edges between the same two faces are fanned out.
    """
    go = _plotly()
    values, kind, label, names = _face_values(g, color_by)
    pos = _positions(g, layout, seed)
    dim = pos.shape[1]
    Trace = go.Scatter if dim == 2 else go.Scatter3d

    def coords(p):
        return {ax: p[:, c].tolist() for c, ax in enumerate("xyz"[:dim])}

    fig = go.Figure()

    ei = g.edge_index
    n_edges = ei.shape[1]
    conv = g.edge_feats.get("convexity")
    if n_edges:
        mids = _fan_out_parallel_edges(pos, ei)
        classes = (
            [(c, *CONVEXITY_COLORS[c]) for c in CONVEXITY_COLORS]
            if conv is not None
            else [(None, "edge", "#555555")]
        )
        for code, cname, color in classes:
            sel = np.arange(n_edges) if code is None else np.flatnonzero(conv == code)
            if not len(sel):
                continue
            pts = np.concatenate(
                [np.stack([pos[ei[0, k]], mids[k], pos[ei[1, k]], np.full(dim, np.nan)]) for k in sel]
            )
            line = {ax: [None if np.isnan(v) else v for v in pts[:, c]] for c, ax in enumerate("xyz"[:dim])}
            fig.add_trace(
                Trace(
                    mode="lines",
                    hoverinfo="skip",
                    name=f"edge: {cname}",
                    line=dict(color=color, width=3),
                    **line,
                )
            )
        fig.add_trace(  # invisible markers at the midpoints carry the edge hover text
            Trace(
                mode="markers",
                marker=dict(size=7, color="rgba(0,0,0,0)"),
                hovertext=[
                    _hover(f"edge {k}: face {ei[0, k]} - face {ei[1, k]}", g.edge_feats, k)
                    for k in range(n_edges)
                ],
                hoverinfo="text",
                showlegend=False,
                name="edge info",
                **coords(mids),
            )
        )

    if size_by is not None and size_by in g.node_feats and g.node_feats[size_by].ndim == 1:
        s = np.sqrt(np.maximum(g.node_feats[size_by].astype(float), 0))
        sizes = 10 + 22 * (s - s.min()) / max(float(np.ptp(s)), 1e-12)
    else:
        sizes = np.full(g.num_faces, 16.0)
    hover = [_hover(f"face {i}", g.node_feats, i) for i in range(g.num_faces)]
    text = [str(i) for i in range(g.num_faces)]
    style: dict[str, Any] = dict(mode="markers+text" if labels else "markers", hoverinfo="text")
    if labels:
        style.update(textposition="middle center", textfont=dict(size=9, color="white"))
    outline = dict(width=1, color="black")
    if kind == "scalar":
        fig.add_trace(
            Trace(
                text=text if labels else None,
                hovertext=hover,
                name="faces",
                showlegend=False,
                marker=dict(
                    size=sizes.tolist(),
                    color=values.tolist(),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=label),
                    line=outline,
                ),
                **style,
                **coords(pos),
            )
        )
    else:
        for c in sorted({int(v) for v in values}):
            idx = np.flatnonzero(values == c)
            fig.add_trace(
                Trace(
                    text=[text[i] for i in idx] if labels else None,
                    hovertext=[hover[i] for i in idx],
                    name=names[c] if names and c < len(names) else f"{label}={c}",
                    marker=dict(size=sizes[idx].tolist(), color=PALETTE[c % len(PALETTE)], line=outline),
                    **style,
                    **coords(pos[idx]),
                )
            )

    title = f"{g.meta.get('source') or 'part'}: face graph, {g.num_faces} faces / {g.num_edges} edges, coloured by {label}"
    if dim == 2:
        fig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), plot_bgcolor="white"
        )
    else:
        fig.update_layout(scene=dict(aspectmode="data"))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return fig
