# Adding your own feature

A feature is a function `(ctx, index, **params) -> np.ndarray` for ONE face or ONE retained edge,
registered with a decorator that declares its name, dtype, shape and version.

```python
import numpy as np
import brepgraph as bg

@bg.node_feature("is_planar", dtype="int64", shape=(), version="1")
def is_planar(ctx, i):
    return int(ctx.surface(i).GetType() == 0)          # GeomAbs_Plane

@bg.edge_feature("half_length", dtype="float32", shape=(), defaults={"k": 0.5})
def half_length(ctx, i, k):
    return ctx.edge_props(i).Mass() * k

fs = bg.FeatureSet(
    nodes=["area", "is_planar"],
    edges=[("half_length", {"k": 0.25}), "length"],
)
g = bg.load("part.step", fs)
```

- `shape` may be a tuple or a callable of the resolved params, e.g. `lambda p: (7, p["size"], p["size"])`.
  A parameter called `size` is overridden by `load(..., grid_size=N)`. `dtype` may also be a callable of the params
  (the `d2` / `a3` histograms use this for their `dtype` option).
- The pipeline checks shape, casts to the declared dtype, and rejects NaN/inf with a `FeatureError`
  that names the feature and the face/edge index.
- The context caches adaptors and properties: `ctx.face(i)`, `ctx.surface(i)`, `ctx.uv_bounds(i)`,
  `ctx.face_props(i)`, `ctx.classifier(i)`, `ctx.sample_surface(i, u, v)`, `ctx.edge(i)`, `ctx.curve(i)`,
  `ctx.edge_props(i)`, `ctx.edge_faces(i)`, `ctx.dihedral(i)`, `ctx.neighbors(i)`, `ctx.num_wires(i)`,
  `ctx.triangles(i)` and `ctx.diagonal`. Prefer these to building OCP objects yourself.
- Names and versions are written to `graph.meta["features"]`, so a saved graph records how it was made.
- Registering a name twice raises `ValueError`. Avoid names containing `face` or `index` if you use the
  PyG adapter (PyG would batch them along the wrong dimension).
- Run `python scripts/gen_feature_docs.py` to refresh `docs/features.md` for the built-in features.
- **Parallel loading.** `bg.load_many` and `brepgraph batch` use fresh worker processes, which only know features
  registered when `brepgraph` and your modules are imported. Put custom features in an importable module (or at the
  top level of the main script) and import it before calling `load_many`. Features defined in a notebook cell are
  not visible to workers; use `workers=1` there.

## Face-pair features

`@bg.pair_feature(name, dtype=..., shape=..., defaults=...)` registers a function `(ctx, **params)` that returns
the whole `[F, F, *shape]` array at once (bulk computation is much faster than per-pair calls). Use
`ctx.triangles(i)` for tessellation-based sampling and `ctx.neighbors(i)` for adjacency; see `features/pair.py`.
The tessellation is removed from the solid when the graph is built, so callers' shapes are left untouched.
Dense outputs grow with `F * F`: allocate the result once in its final dtype rather than building a float64 copy.
