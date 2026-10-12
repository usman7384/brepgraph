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
  A parameter called `size` is overridden by `load(..., grid_size=N)`.
- The pipeline checks shape, casts to the declared dtype, and rejects NaN/inf with a `FeatureError`
  that names the feature and the face/edge index.
- The context caches adaptors and properties: `ctx.face(i)`, `ctx.surface(i)`, `ctx.uv_bounds(i)`,
  `ctx.face_props(i)`, `ctx.classifier(i)`, `ctx.sample_surface(i, u, v)`, `ctx.edge(i)`, `ctx.curve(i)`,
  `ctx.edge_props(i)`, `ctx.edge_faces(i)`, `ctx.dihedral(i)`. Prefer these to building OCP objects yourself.
- Names and versions are written to `graph.meta["features"]`, so a saved graph records how it was made.
- Registering a name twice raises `ValueError`. Avoid names containing `face` or `index` if you use the
  PyG adapter (PyG would batch them along the wrong dimension).
- Run `python scripts/gen_feature_docs.py` to refresh `docs/features.md` for the built-in features.
