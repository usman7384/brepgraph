# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/). The graph layout has its own
`schema_version`, documented in `docs/schema.md`.

## [Unreleased]

## [0.2.0] - 2026-10-12

### Added
- BrepMFR-style features, implemented from the paper (Zhang et al., CAGD 111, 2024, section 3.1.3):
  face-pair matrices `shortest_path`, `d2` and `a3` (64-bin histograms), face features `num_loops` and
  `num_adjacent_faces`, and the `brepmfr` preset (`bg.load(path, "brepmfr")`).
- `pair_feats` on `BrepGraph` for dense `[F, F, ...]` features, the `@pair_feature` decorator, and
  `FeatureSet(pairs=[...])`. Graph schema 1.1.0; archives written with schema 1.0 still load.
- Visualization (`pip install brepgraph[viz]`): `bg.viz.show` (part coloured by any node feature, edges by
  convexity, optional graph overlay), `bg.viz.show_graph` (the face graph on its own, 2-D / 3-D / centroid
  layouts), `bg.viz.show_face` (UV samples, mask and normals) and `brepgraph view [--kind graph]`.
- DGL adapter `BrepGraph.to_dgl()`; the PyG adapter can attach pair features.
- `meta["source_path"]` records the resolved input path so the viewer can re-load the part.
- `examples/gnn_with_pyg.py`, `docs/using-with-gnns.md`, a demo notebook, mypy checks and coverage in CI.

## [0.1.0] - 2026-10-12

### Added
- STEP / IGES / BREP loading of a single solid, with validity check, `ShapeFix` healing, optional
  same-domain face merge, optional unit-box normalisation, and support for in-memory OCP shapes.
- Face features `surface_type`, `area`, `centroid`, `uv_grid`; edge features `curve_type`, `length`,
  `convexity`, `dihedral_angle`, `u_grid`; the `uvnet` preset.
- Feature registry (`@node_feature`, `@edge_feature`, `FeatureSet`).
- `BrepGraph` with `.npz` save / load (schema 1.0.0, no pickle), PyG adapter, `one_hot` / `standardize`
  transforms and the `brepgraph convert` / `info` CLI.
- Support for OCP 7.9 (Python 3.10) and 8.x (Python 3.11+).

[Unreleased]: https://github.com/usman7384/brepgraph/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/usman7384/brepgraph/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/usman7384/brepgraph/releases/tag/v0.1.0
