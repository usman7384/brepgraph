# Graph schema (version 1.0.0)

A face-adjacency graph: one node per face, one graph edge per shared B-rep edge. Plain NumPy
arrays only; no OCP handles are ever serialised.

| Field | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `schema_version` | scalar | str | Semantic version of this layout; the loader rejects another major |
| `num_faces` | scalar | int | F |
| `num_edges` | scalar | int | E, retained B-rep edges |
| `edge_index` | [2, E] | int64 | Face ids joined by each edge, `src < dst`, stored once |
| `node_feats` | name -> [F, ...] | float32 / int64 | Per-face features |
| `edge_feats` | name -> [E, ...] | float32 / int64 | Per-edge features; row i belongs to column i of `edge_index` |
| `meta` | dict | JSON | Source name, bbox, feature names/versions/params, dropped-edge counts, library + OCP version, healing/normalise flags |

Default features and their shapes are in [features.md](features.md) (generated from the registry).

## Conventions

- **Channels first**: `uv_grid` is `[F, 7, size, size]` (axis 1 = u, axis 2 = v), `u_grid` is `[E, 6, size]`.
- **Normals are outward** from the solid; a face reversed in the B-rep has its surface normal flipped.
- **Mask** is 1 where a sample lies inside the trimmed face (the boundary counts as inside). Samples
  outside the trim are still evaluated on the underlying surface; the mask says whether to trust them.
  At singular points (sphere poles, cone apexes) the normal is evaluated a vanishing step towards the
  parametric centre; if none exists the sample has a zero normal and mask 0. So `mask == 1` implies a
  unit normal.
- **Seam edges** (same face on both sides) and **degenerated edges** are dropped. A sphere therefore
  has 1 face and 0 edges. Counts are recorded in `meta["dropped_edges"]`. This is not
  configurable yet, and has not been compared against UV-Net's own preprocessing.
- **Parallel edges** (two faces sharing several edges) are kept as separate columns.
- **Order** is deterministic: ids are positions in an indexed map from `TopExp.MapShapes`, so the same
  file and OCP version always give the same ids. A different OCP version may reorder them; `meta` stores
  the OCP version.
- **Edge tangents** in `u_grid` follow the edge's own B-rep orientation (the orientation stored in the
  solid's edge map), and adapters copy them unchanged onto the reverse direction.
- **Convexity**: 0 concave, 1 convex, 2 smooth (dihedral angle < `tol`, default 0.01 rad). The sign is
  fixed by tests (all box edges convex, pocket edges concave).
- **Dihedral angle**: angle between the two outward normals at the edge midpoint, 0..pi. It is 0 for a
  smooth join and pi/2 for both a convex and a concave right angle; use `convexity` to tell them apart.
- **Units**: whatever OCCT's reader produces (millimetres for STEP). No silent rescaling;
  `normalize=True` centres the bounding box on the origin and scales the longest side to 1.
- Categorical features are stored as integer indices; one-hot is a user-side transform.
