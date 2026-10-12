# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/). The graph layout has its own
`schema_version`, documented in `docs/schema.md`.

## [Unreleased]

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

[Unreleased]: https://github.com/usman7384/brepgraph/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/usman7384/brepgraph/releases/tag/v0.1.0
