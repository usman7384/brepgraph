# brepgraph

[![CI](https://github.com/usman7384/brepgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/usman7384/brepgraph/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Turn B-rep CAD files into ML-ready graphs. A STEP file goes in; a framework-neutral NumPy graph comes out.
Each **face** becomes a node (surface type, area, centroid, a UV grid of points, normals and trim mask) and
each shared **edge** becomes a graph edge (curve type, length, convexity, dihedral angle, a sampled curve
grid). A thin adapter hands the graph to PyTorch Geometric.

- `pip install` only: built on the OCP wheels from PyPI. No conda, no proprietary CAD kernel.
- Featurization only: no models, no training code.
- Features are registered functions, so you can add or mix your own (`docs/extending.md`).

## Installation

```bash
pip install brepgraph               # numpy + OCP
pip install "brepgraph[pyg]"        # + PyTorch Geometric adapter
```

Python 3.10 to 3.13 on Linux, macOS and Windows. PyTorch builds are CUDA- and version-specific, so they are not
pinned; install the build that matches your machine first.

## Quick start

```python
import brepgraph as bg

g = bg.load("part.step")                      # default `uvnet` feature set
print(g)                                      # every feature with its shape and dtype
g.num_faces, g.num_edges
g.node_feats["uv_grid"].shape                 # (F, 7, 10, 10): x y z nx ny nz mask
g.edge_feats["u_grid"].shape                  # (E, 6, 10):     x y z tx ty tz

g.save("part.npz")                            # plain NumPy archive, no pickle
g = bg.BrepGraph.load("part.npz")

data = g.to_pyg(x=("area", "centroid"))       # torch_geometric.data.Data, both edge directions
```

```bash
brepgraph convert part.step part.npz --preset uvnet
brepgraph info part.npz
```

All errors derive from `bg.BrepGraphError` (`FileReadError`, `UnsupportedShapeError`, `InvalidSolidError`,
`FeatureError`, `MissingExtraError`), so a batch job only needs one `try/except`.

## Features

| Kind | Name | Shape | dtype | Meaning |
| --- | --- | --- | --- | --- |
| face | `surface_type` | [F] | int64 | `GeomAbs_SurfaceType` index (11 values) |
| face | `area` | [F] | float32 | model units squared |
| face | `centroid` | [F, 3] | float32 | centre of mass |
| face | `uv_grid` | [F, 7, n, n] | float32 | x, y, z, outward normal, trim mask |
| edge | `curve_type` | [E] | int64 | `GeomAbs_CurveType` index (9 values) |
| edge | `length` | [E] | float32 | model units |
| edge | `convexity` | [E] | int64 | 0 concave, 1 convex, 2 smooth |
| edge | `dihedral_angle` | [E] | float32 | radians, 0 to pi |
| edge | `u_grid` | [E, 6, n] | float32 | x, y, z, unit tangent along the edge |

- Complete reference (generated from the registry): [docs/features.md](docs/features.md)
- Graph layout and conventions (normals, mask, seam edges, ordering, units): [docs/schema.md](docs/schema.md)
- Writing your own feature: [docs/extending.md](docs/extending.md)

## Known limitations

- One solid per file. Assemblies, multi-body files, open shells and non-manifold edges raise
  `UnsupportedShapeError`.
- Seam and degenerated edges are dropped (a sphere is 1 face and 0 edges). This is not configurable yet.
- Units are whatever OCCT produces (millimetres for STEP); nothing is rescaled unless `normalize=True`.
- Speed: every sample is a Python-to-C++ call. Expect roughly 30 ms per face with the default 10x10 grid.
- Face and edge order is deterministic for a given file and OCP version, but may differ between OCP versions.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests scripts && ruff format --check src tests scripts
mypy
pytest                       # fast suite;  `pytest -m slow` adds a 535-face part
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits and license

The feature ideas follow UV-Net (Jayaraman et al., CVPR 2021) and BRepNet (Lambourne et al., CVPR 2021). This
package is written independently against the papers and reuses none of their code. Geometry is by
[OpenCASCADE Technology](https://dev.opencascade.org) (LGPL 2.1 with exception) through the
[OCP](https://github.com/CadQuery/OCP) bindings (`cadquery-ocp`), installed as a separate dependency.

Released under the [MIT License](LICENSE).
