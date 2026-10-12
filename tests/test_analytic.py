"""Features checked against shapes with known answers."""

import math

import numpy as np

import fixtures as F

PI = math.pi
SURF = {"plane": 0, "cylinder": 1, "cone": 2, "sphere": 3, "torus": 4}
CURVE = {"line": 0, "circle": 1}


def _normals_points(g, face):
    grid = g.node_feats["uv_grid"][face]
    m = grid[6] > 0.5
    return grid[:3][:, m].T, grid[3:6][:, m].T


def test_box(graph):
    g = graph(F.box(10, 20, 30))
    assert (g.num_faces, g.num_edges) == (6, 12)
    nf, ef = g.node_feats, g.edge_feats
    assert (nf["surface_type"] == SURF["plane"]).all()
    np.testing.assert_allclose(sorted(nf["area"]), [200, 200, 300, 300, 600, 600], rtol=1e-6)
    assert (ef["curve_type"] == CURVE["line"]).all()
    np.testing.assert_allclose(sorted(ef["length"]), [10] * 4 + [20] * 4 + [30] * 4, rtol=1e-6)
    assert (ef["convexity"] == 1).all()
    np.testing.assert_allclose(ef["dihedral_angle"], PI / 2, atol=1e-6)
    assert (nf["uv_grid"][:, 6] == 1).all()  # mask is 1 everywhere
    centre = np.array([5.0, 10.0, 15.0])
    for f in range(6):
        pts, nrm = _normals_points(g, f)
        axis = np.abs(nrm[0]).round()
        assert sorted(axis) == [0, 0, 1]  # normal is an axis direction
        np.testing.assert_allclose(
            nrm, np.broadcast_to(nrm[0], nrm.shape), atol=1e-6
        )  # constant across the face
        assert np.dot(nrm[0], nf["centroid"][f] - centre) > 0  # outward
        np.testing.assert_allclose(pts @ nrm[0], float(pts[0] @ nrm[0]), atol=1e-4)  # points lie in-plane


def test_cylinder(graph):
    g = graph(F.cylinder(5, 10))
    assert (g.num_faces, g.num_edges) == (3, 2)
    assert g.meta["dropped_edges"]["seam"] == 1
    nf, ef = g.node_feats, g.edge_feats
    assert sorted(nf["surface_type"]) == [SURF["plane"], SURF["plane"], SURF["cylinder"]]
    lateral = nf["surface_type"] == SURF["cylinder"]
    np.testing.assert_allclose(nf["area"][lateral], 2 * PI * 5 * 10, rtol=1e-6)
    np.testing.assert_allclose(nf["area"][~lateral], PI * 25, rtol=1e-6)
    assert (ef["curve_type"] == CURVE["circle"]).all()
    np.testing.assert_allclose(ef["length"], 2 * PI * 5, rtol=1e-6)
    assert (ef["convexity"] == 1).all()
    np.testing.assert_allclose(ef["dihedral_angle"], PI / 2, atol=1e-5)
    # lateral normals point away from the axis
    f = int(np.flatnonzero(lateral)[0])
    pts, nrm = _normals_points(g, f)
    radial = pts[:, :2] / np.linalg.norm(pts[:, :2], axis=1, keepdims=True)
    np.testing.assert_allclose(nrm[:, :2], radial, atol=1e-5)
    np.testing.assert_allclose(nrm[:, 2], 0, atol=1e-5)


def test_sphere(graph):
    g = graph(F.sphere(5))
    assert (g.num_faces, g.num_edges) == (1, 0)
    assert g.meta["dropped_edges"] == {"seam": 1, "degenerated": 2}
    np.testing.assert_allclose(g.node_feats["area"], 4 * PI * 25, rtol=1e-6)
    for name, arr in g.edge_feats.items():
        assert arr.shape[0] == 0, name
    pts, nrm = _normals_points(g, 0)
    assert len(pts) > 0
    np.testing.assert_allclose(nrm, pts / 5.0, atol=1e-4)  # includes the poles
    assert np.isfinite(g.node_feats["uv_grid"]).all()


def test_cone_apex_has_no_nan(graph):
    g = graph(F.cone(5, 0, 10))
    assert g.num_faces == 2
    lateral = int(np.flatnonzero(g.node_feats["surface_type"] == SURF["cone"])[0])
    slant = math.hypot(5, 10)
    np.testing.assert_allclose(g.node_feats["area"][lateral], PI * 5 * slant, rtol=1e-6)
    grid = g.node_feats["uv_grid"][lateral]
    assert np.isfinite(grid).all()
    pts, nrm = _normals_points(g, lateral)
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1, atol=1e-5)
    # outward = away from the axis, and tilted up (toward +z) since the cone narrows upwards
    assert (np.einsum("ij,ij->i", nrm[:, :2], pts[:, :2]) >= -1e-6).all()
    assert (nrm[:, 2] > 0).all()


def test_torus(graph):
    R, r = 10.0, 2.0
    g = graph(F.torus(R, r))
    assert g.num_faces == 1
    np.testing.assert_allclose(g.node_feats["area"], 4 * PI**2 * R * r, rtol=1e-6)
    pts, nrm = _normals_points(g, 0)
    ring = np.c_[pts[:, :2] / np.linalg.norm(pts[:, :2], axis=1, keepdims=True) * R, np.zeros(len(pts))]
    np.testing.assert_allclose(nrm, (pts - ring) / r, atol=1e-4)


def test_pocket_box(graph):
    g = graph(F.pocket_box())
    assert (g.num_faces, g.num_edges) == (11, 24)
    conv = g.edge_feats["convexity"]
    assert (conv == 0).sum() == 8 and (conv == 1).sum() == 16 and (conv == 2).sum() == 0


def test_filleted_box(graph):
    g = graph(F.filleted_box(2.0))
    assert g.num_faces == 7
    smooth = g.edge_feats["convexity"] == 2
    assert smooth.sum() == 2
    assert (g.edge_feats["dihedral_angle"][smooth] < 1e-3).all()
    assert (g.node_feats["surface_type"] == SURF["cylinder"]).sum() == 1


def test_plate_with_hole_mask(graph):
    side, thick, r = 100.0, 10.0, 20.0
    g = graph(F.plate_with_hole(side, thick, r), grid_size=50)
    top = np.flatnonzero(np.isclose(g.node_feats["area"], side**2 - PI * r**2, rtol=1e-5))
    assert len(top) == 2  # top and bottom
    for f in top:
        mask = g.node_feats["uv_grid"][f, 6]
        assert abs((mask == 0).mean() - PI * r**2 / side**2) < 0.02


def test_nurbs_faces_work(graph):
    g = graph(F.nurbs(F.pocket_box()))
    assert (g.num_faces, g.num_edges) == (11, 24)
    assert (g.node_feats["surface_type"] == SURF["plane"]).sum() == 0  # all B-spline now
    assert (g.edge_feats["convexity"] == 0).sum() == 8


def test_u_grid_tangent_follows_point_order(graph):
    """Convention: samples and tangents run along the edge's own B-rep orientation (also when reversed)."""
    for shape in (F.pocket_box(), F.cylinder(), F.filleted_box()):
        g = graph(shape)
        grid = g.edge_feats["u_grid"]
        assert grid.shape[0] > 0
        np.testing.assert_allclose(np.linalg.norm(grid[:, 3:6], axis=1), 1.0, atol=1e-5)
        travel = grid[:, 0:3, -1] - grid[:, 0:3, 0]
        mid = grid[:, 3:6, grid.shape[2] // 2]
        closed = np.linalg.norm(travel, axis=1) < 1e-6  # full circles return to the start
        assert (np.einsum("ij,ij->i", travel, mid)[~closed] > 0).all()
