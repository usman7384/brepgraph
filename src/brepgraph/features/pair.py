"""Face-pair features (BrepMFR proximity matrices). Each returns the whole ``[F, F, ...]`` array.

Definitions follow Zhang et al., "BrepMFR", Comput. Aided Geom. Des. 111 (2024), section 3.1.3:

* ``shortest_path``: number of graph edges on the shortest path between two faces (their ``D_g``).
* ``d2``: for a face pair, ``samples`` random point pairs (one point on each face, area-uniform),
  distance divided by the solid's bounding-box diagonal, histogrammed into ``bins`` over [0, 1].
* ``a3``: the same for the angle formed by three random points. The paper does not say how the
  three points are drawn from two faces; here each point is drawn from the union of the two faces
  (face chosen with probability proportional to its area), the angle is taken at the middle point
  and divided by pi, then histogrammed over [0, 1].

Histograms are stored as frequencies by default (each ``[i, j]`` vector sums to 1). The ``dtype``
parameter trades precision for memory: ``"float16"`` halves it, and ``"uint16"`` stores the exact
integer counts used in the paper (each vector sums to ``samples``), also at half the memory. An
``[F, F, bins]`` histogram takes ``F * F * bins * 4`` bytes as float32 (about 200 MB for 900 faces
and 64 bins). Matrices are symmetric. Sampling is seeded, so a given file always yields the
same arrays.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .. import meshing
from ..registry import pair_feature

_HIST_DEFAULTS = {"bins": 64, "samples": 512, "seed": 0, "dtype": "float32"}
#: storage options for the histograms: frequencies (float32 / float16) or exact counts (uint16)
HIST_DTYPES = ("float32", "float16", "uint16")


def _hist_shape(p):
    return (p["bins"],)


def _hist_dtype(p):
    return p["dtype"]


def _hist_output(n: int, bins: int, samples: int, dtype: str) -> np.ndarray:
    if dtype not in HIST_DTYPES:
        raise ValueError(f"dtype must be one of {HIST_DTYPES}, got {dtype!r}")
    if dtype == "uint16" and samples > np.iinfo(np.uint16).max:
        raise ValueError(f"uint16 counts need samples <= 65535, got {samples}")
    return np.zeros((n, n, bins), dtype=dtype)


def _store(out: np.ndarray, i: int, counts: np.ndarray, samples: int) -> None:
    """Write histogram counts for row ``i`` (pairs ``(i, j >= i)``) as counts or frequencies."""
    out[i, i:] = counts if out.dtype.kind == "u" else counts / samples


def _face_points(ctx, samples: int, seed: int, sets: int) -> np.ndarray:
    """``[sets, F, samples, 3]`` area-uniform points per face, from the face's triangulation."""
    rng = np.random.default_rng(seed)
    out = np.empty((sets, ctx.num_faces, samples, 3))
    for i in range(ctx.num_faces):
        mesh = ctx.triangles(i)
        for s in range(sets):
            if mesh is None:  # no triangulation (tiny face): fall back to its centroid
                c = ctx.face_props(i).CentreOfMass()
                out[s, i] = (c.X(), c.Y(), c.Z())
            else:
                out[s, i] = meshing.sample_triangles(mesh[0], mesh[1], samples, rng)
    return out


def _bin_rows(values: np.ndarray, bins: int) -> np.ndarray:
    """``values`` in [0, 1], shape ``[rows, samples]`` -> histogram counts ``[rows, bins]``."""
    rows, samples = values.shape
    idx = np.clip((values * bins).astype(np.int64), 0, bins - 1)
    flat = (idx + np.arange(rows)[:, None] * bins).ravel()
    return np.bincount(flat, minlength=rows * bins).reshape(rows, bins)


def _mirror(out: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(out.shape[0], 1)
    out[iu[1], iu[0]] = out[iu[0], iu[1]]
    return out


@pair_feature("shortest_path", dtype="int64", shape=())
def shortest_path(ctx):
    """Number of edges on the shortest path between two faces in the face graph (0 on the diagonal, -1 if unreachable)."""
    n = ctx.num_faces
    out = np.full((n, n), -1, dtype=np.int64)
    for src in range(n):
        out[src, src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in ctx.neighbors(u):
                if out[src, v] < 0:
                    out[src, v] = out[src, u] + 1
                    queue.append(v)
    return out


@pair_feature("d2", dtype=_hist_dtype, shape=_hist_shape, defaults=_HIST_DEFAULTS)
def d2(ctx, bins, samples, seed, dtype):
    """D2 histogram per face pair: distance between random points on the two faces, divided by the solid's bounding-box diagonal (bins over [0, 1]; frequencies summing to 1, or counts with dtype="uint16")."""
    n, diag = ctx.num_faces, max(ctx.diagonal, 1e-12)
    out = _hist_output(n, bins, samples, dtype)
    pts = _face_points(ctx, samples, seed, 2)
    for i in range(n):
        dist = np.linalg.norm(pts[0, i][None] - pts[1, i:], axis=-1) / diag  # [n - i, samples]
        _store(out, i, _bin_rows(dist, bins), samples)
    return _mirror(out)


@pair_feature("a3", dtype=_hist_dtype, shape=_hist_shape, defaults=_HIST_DEFAULTS)
def a3(ctx, bins, samples, seed, dtype):
    """A3 histogram per face pair: angle (divided by pi) at the middle of three random points drawn from the union of the two faces (bins over [0, 1]; frequencies summing to 1, or counts with dtype="uint16")."""
    n = ctx.num_faces
    out = _hist_output(n, bins, samples, dtype)
    pts = _face_points(ctx, samples, seed, 3)
    area = np.array([ctx.face_props(i).Mass() for i in range(n)])
    pick = np.random.default_rng([seed, 1])
    for i in range(n):
        w = area[i] / np.maximum(area[i] + area[i:], 1e-300)  # P(point comes from face i)
        chosen = []
        for k in range(3):
            from_i = pick.random((n - i, samples)) < w[:, None]
            chosen.append(np.where(from_i[..., None], pts[k, i][None], pts[k, i:]))
        a, b, c = chosen
        u, v = a - b, c - b
        denom = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
        cos = np.einsum("rsk,rsk->rs", u, v) / np.where(denom > 0, denom, 1.0)
        angle = np.where(denom > 0, np.arccos(np.clip(cos, -1.0, 1.0)), 0.5 * np.pi) / np.pi
        _store(out, i, _bin_rows(angle, bins), samples)
    return _mirror(out)
