"""Varied, harder parts through the real loader: count snapshots and a large-part smoke test.

These are procedurally generated stand-ins for public datasets (see README, "Testing"). Drop real
``.step``/``.stp`` files into ``tests/data`` and ``test_data_dir`` will run the invariants on them.
"""

import time
from pathlib import Path

import numpy as np
import pytest

import brepgraph as bg
import fixtures as F

SNAPSHOTS = {  # name -> (faces, edges)
    "holey_plate_5": (lambda: F.holey_plate(5), (6 + 25, 12 + 25 * 2)),
    "nurbs_cylinder": (lambda: F.nurbs(F.cylinder()), (3, 2)),
    "tiny_fillet": (F.tiny_fillet, (7, 15)),
}


@pytest.mark.parametrize("name", SNAPSHOTS)
def test_count_snapshots(graph, name):
    build, (faces, edges) = SNAPSHOTS[name]
    g = graph(build())
    assert (g.num_faces, g.num_edges) == (faces, edges)


def _check(g):
    g.validate()
    for arr in (*g.node_feats.values(), *g.edge_feats.values()):
        if arr.dtype.kind == "f":
            assert np.isfinite(arr).all()
    grid = g.node_feats["uv_grid"]
    masked = np.moveaxis(grid, 1, -1)[grid[:, 6] > 0.5]
    np.testing.assert_allclose(np.linalg.norm(masked[:, 3:6], axis=1), 1.0, atol=1e-5)


@pytest.mark.slow
def test_large_part(graph):
    shape = F.holey_plate(23)  # 535 faces
    t0 = time.perf_counter()
    g = graph(shape)
    elapsed = time.perf_counter() - t0
    assert g.num_faces >= 500
    _check(g)
    print(f"\nlarge part: {g.num_faces} faces, {g.num_edges} edges in {elapsed:.1f}s")


@pytest.mark.data
def test_data_dir():
    files = sorted([*Path(__file__).parent.joinpath("data").glob("*.st*p")])
    if not files:
        pytest.skip("no files in tests/data")
    for path in files:
        _check(bg.load(path))
