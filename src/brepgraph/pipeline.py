"""The only place the stages are wired together: solid -> topology -> features -> BrepGraph."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import __version__, io, presets
from ._compat import ocp_version
from .context import Context
from .errors import BrepGraphError, FeatureError
from .graph import BrepGraph
from .registry import FeatureSet, FeatureSpec
from .topology import bounding_box, build_topology


def _compute(spec: FeatureSpec, ctx: Context, n: int, params: dict[str, Any]) -> np.ndarray:
    expected = spec.expected_shape(params)
    out = np.zeros((n, *expected), dtype=spec.dtype)
    kind = "face" if spec.kind == "node" else "edge"
    for i in range(n):
        try:
            value = np.asarray(spec.fn(ctx, i, **params))
        except BrepGraphError:
            raise
        except Exception as exc:
            raise FeatureError(spec.name, i, f"{type(exc).__name__}: {exc}", kind) from exc
        if value.shape != expected:
            raise FeatureError(spec.name, i, f"returned shape {value.shape}, declared {expected}", kind)
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise FeatureError(spec.name, i, "returned NaN or infinite values", kind)
        out[i] = value
    return out


def _compute_pair(spec: FeatureSpec, ctx: Context, params: dict[str, Any]) -> np.ndarray:
    n = ctx.num_faces
    expected = (n, n, *spec.expected_shape(params))
    try:
        value = np.asarray(spec.fn(ctx, **params))
    except BrepGraphError:
        raise
    except Exception as exc:
        raise FeatureError(spec.name, None, f"{type(exc).__name__}: {exc}", "pair") from exc
    if value.shape != expected:
        raise FeatureError(spec.name, None, f"returned shape {value.shape}, declared {expected}", "pair")
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise FeatureError(spec.name, None, "returned NaN or infinite values", "pair")
    return value.astype(spec.dtype, copy=False)


def build_graph(
    solid,
    feature_set: FeatureSet,
    *,
    grid_size: int | None = None,
    source_info: dict[str, Any] | None = None,
) -> BrepGraph:
    topo = build_topology(solid)
    ctx = Context(topo)
    node_specs, edge_specs, pair_specs = feature_set.resolve(grid_size)

    try:
        node_feats = {s.name: _compute(s, ctx, topo.num_faces, p) for s, p in node_specs}
        edge_feats = {s.name: _compute(s, ctx, topo.num_edges, p) for s, p in edge_specs}
        pair_feats = {s.name: _compute_pair(s, ctx, p) for s, p in pair_specs}
    finally:
        ctx.release_mesh()

    meta: dict[str, Any] = dict(source_info or {})
    meta.update(
        {
            "bbox": bounding_box(solid),
            "features": {
                "node": [{"name": s.name, "version": s.version, "params": p} for s, p in node_specs],
                "edge": [{"name": s.name, "version": s.version, "params": p} for s, p in edge_specs],
                "pair": [{"name": s.name, "version": s.version, "params": p} for s, p in pair_specs],
            },
            "dropped_edges": {"seam": topo.n_seam, "degenerated": topo.n_degenerated},
            "library_version": __version__,
            "ocp_version": ocp_version(),
        }
    )
    graph = BrepGraph(
        num_faces=topo.num_faces,
        num_edges=topo.num_edges,
        edge_index=topo.edge_index,
        node_feats=node_feats,
        edge_feats=edge_feats,
        pair_feats=pair_feats,
        meta=meta,
    )
    graph.validate()
    return graph


def load(
    source: str | os.PathLike[str] | Any,
    features: FeatureSet | str | None = None,
    *,
    grid_size: int | None = None,
    heal: bool = True,
    unify_faces: bool = False,
    normalize: bool = False,
) -> BrepGraph:
    """STEP/IGES/BREP file path (or an OCP ``TopoDS_Shape``) -> :class:`BrepGraph`.

    ``features`` is a :class:`FeatureSet`, a preset name, or ``None`` for the ``uvnet`` preset.
    ``grid_size`` overrides the ``size`` parameter of every grid feature that does not set it.
    """
    if features is None:
        features = presets.uvnet
    elif isinstance(features, str):
        features = presets.get(features)
    solid, info = io.load_solid(source, heal=heal, unify_faces=unify_faces, normalize=normalize)
    return build_graph(solid, features, grid_size=grid_size, source_info=info)
