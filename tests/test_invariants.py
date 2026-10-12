"""Properties that must hold on every input."""

import numpy as np
import pytest
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

import brepgraph as bg
import fixtures as F

SHAPES = {
    "box": F.box,
    "cylinder": F.cylinder,
    "sphere": F.sphere,
    "cone": F.cone,
    "torus": F.torus,
    "pocket": F.pocket_box,
    "fillet": F.filleted_box,
    "plate": F.plate_with_hole,
    "nurbs_pocket": lambda: F.nurbs(F.pocket_box()),
    "nurbs_cylinder": lambda: F.nurbs(F.cylinder()),
    "half_cylinder": F.half_cylinder,
    "tiny_fillet": F.tiny_fillet,
    "holey_plate": F.holey_plate,
}


def _total_area(shape):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    return props.Mass()


@pytest.fixture(params=sorted(SHAPES))
def shape(request):
    return SHAPES[request.param]()


def test_schema_and_values(graph, shape):
    g = graph(shape)
    g.validate()
    for feats in (g.node_feats, g.edge_feats):
        for name, arr in feats.items():
            spec = (bg.NODE_FEATURES if feats is g.node_feats else bg.EDGE_FEATURES)[name]
            assert arr.dtype == np.dtype(spec.dtype), name
            if arr.dtype.kind == "f":
                assert np.isfinite(arr).all(), name
    assert g.edge_index.dtype == np.int64
    assert g.node_feats["uv_grid"].shape == (g.num_faces, 7, 10, 10)
    assert g.edge_feats["u_grid"].shape == (g.num_edges, 6, 10)


def test_masked_normals_are_unit(graph, shape):
    grid = graph(shape).node_feats["uv_grid"]
    masked = np.moveaxis(grid, 1, -1)[grid[:, 6] > 0.5]  # [n, 7]
    assert len(masked) > 0
    np.testing.assert_allclose(np.linalg.norm(masked[:, 3:6], axis=1), 1.0, atol=1e-5)


def test_area_sum_matches_solid(graph, shape):
    g = graph(shape)
    assert g.node_feats["area"].astype(np.float64).sum() == pytest.approx(_total_area(shape), rel=1e-6)


def test_deterministic(graph, step, shape):
    path = step(shape)
    a, b = bg.load(path), bg.load(path)
    for k in a.node_feats:
        np.testing.assert_array_equal(a.node_feats[k], b.node_feats[k])
    for k in a.edge_feats:
        np.testing.assert_array_equal(a.edge_feats[k], b.edge_feats[k])
    np.testing.assert_array_equal(a.edge_index, b.edge_index)


def _scalar_summary(g):
    return {
        "area": np.sort(g.node_feats["area"]),
        "surface_type": np.sort(g.node_feats["surface_type"]),
        "mask": float(g.node_feats["uv_grid"][:, 6].sum()),
        "length": np.sort(g.edge_feats["length"]),
        "dihedral": np.sort(g.edge_feats["dihedral_angle"]),
        "convexity": np.bincount(g.edge_feats["convexity"], minlength=3),
        "curve_type": np.sort(g.edge_feats["curve_type"]),
    }


def test_rigid_motion_invariance(graph, shape):
    a, b = _scalar_summary(graph(shape)), _scalar_summary(graph(F.rigid_motion(shape)))
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-4, atol=1e-4, err_msg=k)


def test_rigid_motion_moves_points_and_normals(graph):
    shape = F.pocket_box()
    a, b = graph(shape), graph(F.rigid_motion(shape))
    # same topology, same order: centroids transform rigidly -> pairwise distances preserved
    ca, cb = a.node_feats["centroid"], b.node_feats["centroid"]
    da = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    db = np.linalg.norm(cb[:, None] - cb[None], axis=-1)
    np.testing.assert_allclose(da, db, atol=1e-3)
    na = a.node_feats["uv_grid"][:, 3:6].reshape(a.num_faces, 3, -1)
    nb = b.node_feats["uv_grid"][:, 3:6].reshape(b.num_faces, 3, -1)
    gram_a = np.einsum("fcn,gcn->fgn", na, na)  # per-sample normal dot products are rotation invariant
    gram_b = np.einsum("fcn,gcn->fgn", nb, nb)
    np.testing.assert_allclose(gram_a, gram_b, atol=1e-4)


def test_scale_covariance(graph, shape):
    s = 2.5
    a, b = graph(shape), graph(F.scaled(shape, s))
    np.testing.assert_allclose(np.sort(b.node_feats["area"]), np.sort(a.node_feats["area"]) * s**2, rtol=1e-4)
    np.testing.assert_allclose(
        np.sort(b.edge_feats["length"]), np.sort(a.edge_feats["length"]) * s, rtol=1e-4
    )
    np.testing.assert_allclose(
        np.sort(b.edge_feats["dihedral_angle"]), np.sort(a.edge_feats["dihedral_angle"]), atol=1e-4
    )


def test_normalize_option(graph):
    g = graph(F.box(10, 20, 30), normalize=True)
    lo, hi = np.array(g.meta["bbox"][:3]), np.array(g.meta["bbox"][3:])
    np.testing.assert_allclose(hi - lo, [1 / 3, 2 / 3, 1.0], atol=1e-6)
    np.testing.assert_allclose((hi + lo) / 2, 0, atol=1e-6)
    assert g.meta["normalize"] is True


def test_parallel_edges_are_kept(graph):
    g = graph(F.half_cylinder())
    assert g.num_faces == 4
    cols = [tuple(c) for c in g.edge_index.T]
    assert len(cols) == g.num_edges == 6
    assert len(set(cols)) < len(cols)  # the flat face and the curved face share two edges
