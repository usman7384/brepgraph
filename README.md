# brepgraph

[![CI](https://github.com/usman7384/brepgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/usman7384/brepgraph/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Turn B-rep CAD files into ML-ready graphs. A STEP file goes in; a framework-neutral NumPy graph comes out.
Each **face** becomes a node (surface type, area, centroid, a UV grid of points, normals and trim mask) and
each shared **edge** becomes a graph edge (curve type, length, convexity, dihedral angle, a sampled curve
grid). Thin adapters hand the graph to PyTorch Geometric or DGL.

- `pip install` only: built on the OCP wheels from PyPI. No conda, no proprietary CAD kernel.
- Featurization only: no models, no training code.
- Features are registered functions, so you can add or mix your own (`docs/extending.md`).
- Interactive Plotly views of the part, its face graph, and individual faces for debugging features.

## Installation

```bash
pip install brepgraph               # numpy + OCP
pip install "brepgraph[pyg]"        # + PyTorch Geometric adapter
pip install "brepgraph[dgl]"        # + DGL adapter
pip install "brepgraph[viz]"        # + Plotly visualization
```

Python 3.10 to 3.13 on Linux, macOS and Windows. PyTorch and DGL builds are CUDA- and version-specific, so they
are not pinned; install the build that matches your machine first.

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
dgl_graph = g.to_dgl()                        # DGL graph

g = bg.load("part.step", "brepmfr")           # BrepMFR-style set incl. face-pair D2 / A3 histograms
g.pair_feats["d2"].shape                      # (F, F, 64)
```

```bash
brepgraph convert part.step part.npz --preset uvnet
brepgraph info part.npz
brepgraph view part.step --color-by area      # interactive HTML next to the input
brepgraph view part.step --kind graph         # the face graph instead of the 3-D part
```

All errors derive from `bg.BrepGraphError` (`FileReadError`, `UnsupportedShapeError`, `InvalidSolidError`,
`FeatureError`, `MissingExtraError`), so a batch job only needs one `try/except`:

```python
for path in files:
    try:
        bg.load(path).save(path.with_suffix(".npz"))
    except bg.BrepGraphError as exc:
        print(path.name, "skipped:", exc)
```

## Features

Node, edge and pair features are stored as separate named arrays (`g.node_feats`, `g.edge_feats`,
`g.pair_feats`). Row `i` of a node array belongs to face `i`; row `k` of an edge array belongs to column `k` of
`g.edge_index`. Categorical features are integer indices.

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

The `brepmfr` preset follows Zhang et al., *BrepMFR*, CAGD 111 (2024): 5x5 grids plus face features
`num_loops` and `num_adjacent_faces`, and face-pair matrices `shortest_path` [F, F], `d2` and `a3`
[F, F, 64] (frequencies; rows sum to 1). The paper does not say how the three A3 points are drawn from two
faces; the choice made here is documented in `src/brepgraph/features/pair.py`. Pair features cost O(F^2) in
time and memory, so they are opt-in.

- Complete reference (generated from the registry): [docs/features.md](docs/features.md)
- Graph layout and conventions (normals, mask, seam edges, ordering, units): [docs/schema.md](docs/schema.md)
- Writing your own feature: [docs/extending.md](docs/extending.md)
- Feeding the graph to a GNN: [docs/using-with-gnns.md](docs/using-with-gnns.md) and
  [examples/gnn_with_pyg.py](examples/gnn_with_pyg.py)
- Demo notebook: [notebooks/demo.ipynb](notebooks/demo.ipynb)

## Visualization

```python
bg.viz.show(g, "surface_type").show()         # part coloured by a feature, edges coloured by convexity
bg.viz.show(g, "area", graph=True).show()     # continuous colours, plus the adjacency graph overlay
bg.viz.show_graph(g, layout="centroid").show()  # the face graph on its own
bg.viz.show_face(g, 12).show()                # one face's UV samples, trim mask and normals
```

`show` re-loads the source file recorded in the graph and checks it against the stored areas and centroids, so
a changed face order is reported instead of silently mis-coloured. Archives do not store the absolute input path
by default (it would leak local directory names); for a graph loaded from an archive pass `source=path`, or
save with `include_source_path=True`.

## Known limitations

- One solid per file. Assemblies, multi-body files, open shells and non-manifold edges raise
  `UnsupportedShapeError`.
- Seam and degenerated edges are dropped (a sphere is 1 face and 0 edges). This is not configurable yet.
- Units are whatever OCCT produces (millimetres for STEP); nothing is rescaled unless `normalize=True`.
- Speed: every sample is a Python-to-C++ call. Expect roughly 30 ms per face with the default 10x10 grid.
- Face and edge order is deterministic for a given file and OCP version, but may differ between OCP versions.
- A face graph can be disconnected (for example a solid with an internal cavity); unreachable pairs get
  `shortest_path = -1`.
- Not included: coedge graphs, an on-disk cache, batch processing, splitting multi-solid files.
- The DGL adapter was tested against a stub of the DGL API because no native DGL build was available on the
  development machine; run `pytest tests/test_v02.py` where `dgl` imports to exercise the real thing.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests scripts examples && ruff format --check src tests scripts examples
mypy
pytest                       # fast suite;  `pytest -m slow` adds a 535-face part
```

Features are tested against shapes with known answers (box, cylinder, sphere, cone, torus, pocketed and filleted
boxes, a plate with a hole), against invariants on every input (unit normals where the mask is 1, areas that sum
to the solid's surface area, rigid-motion and scale behaviour, determinism, round trips), and on harder
generated parts. Supported OCP lines are 7.9 (Python 3.10) and 8.x (Python 3.11+). See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

If this package is useful in your research, please cite it (see [CITATION.cff](CITATION.cff)) together with the
papers whose feature definitions it follows.

## Credits and license

The feature ideas follow UV-Net (Jayaraman et al., CVPR 2021), BRepNet (Lambourne et al., CVPR 2021) and
BrepMFR (Zhang et al., CAGD 2024). This package is written independently against the papers and reuses none of
their code. Geometry is by [OpenCASCADE Technology](https://dev.opencascade.org) (LGPL 2.1 with exception)
through the [OCP](https://github.com/CadQuery/OCP) bindings (`cadquery-ocp`), installed as a separate
dependency.

Released under the [MIT License](LICENSE).
