"""BrepMFR features, schema 1.1, DGL adapter, visualization."""

import math
import sys

import numpy as np
import pytest

import brepgraph as bg
import fixtures as F
from brepgraph import cli


def pairs(*names, **params):
    return bg.FeatureSet(nodes=["area"], pairs=[(n, params) if params else n for n in names])


# ---- node features ---------------------------------------------------------------------------
def test_loops_and_adjacency(graph):
    g = graph(F.box(), features=bg.FeatureSet(nodes=["num_loops", "num_adjacent_faces"]))
    assert (g.node_feats["num_loops"] == 1).all() and (g.node_feats["num_adjacent_faces"] == 4).all()
    g = graph(F.plate_with_hole(), features=bg.FeatureSet(nodes=["num_loops", "area", "surface_type"]))
    big = g.node_feats["area"] > 8000  # top and bottom: outer boundary + hole
    assert (g.node_feats["num_loops"][big] == 2).all() and big.sum() == 2
    g = graph(F.cylinder(), features=bg.FeatureSet(nodes=["num_adjacent_faces", "surface_type"]))
    assert g.node_feats["num_adjacent_faces"][g.node_feats["surface_type"] == 1][0] == 2


# ---- pair features ---------------------------------------------------------------------------
def test_shortest_path_on_a_box(graph):
    sp = graph(F.box(), features=pairs("shortest_path")).pair_feats["shortest_path"]
    assert sp.shape == (6, 6) and sp.dtype == np.int64
    assert (np.diag(sp) == 0).all() and (sp == sp.T).all()
    assert sorted(np.unique(sp)) == [0, 1, 2]
    assert ((sp == 2).sum(axis=1) == 1).all()  # exactly one opposite face each


def test_brepmfr_preset_shapes_and_sums(graph):
    g = graph(F.pocket_box(), features="brepmfr")
    n = g.num_faces
    assert g.node_feats["uv_grid"].shape == (n, 7, 5, 5) and g.edge_feats["u_grid"].shape == (
        g.num_edges,
        6,
        5,
    )
    for name in ("d2", "a3"):
        h = g.pair_feats[name]
        assert h.shape == (n, n, 64) and h.dtype == np.float32
        np.testing.assert_allclose(h.sum(-1), 1.0, atol=1e-5)
        np.testing.assert_array_equal(h, h.transpose(1, 0, 2))
    assert g.pair_feats["shortest_path"].shape == (n, n)
    assert g.meta["features"]["pair"][1] == {
        "name": "d2",
        "version": "1",
        "params": {"bins": 64, "samples": 512, "seed": 0, "dtype": "float32"},
    }


def test_d2_unit_square_analytic(graph):
    """Mean distance of two uniform points on a unit square is 0.5214; here divided by the cube's diagonal."""
    g = graph(F.box(1, 1, 1), features=pairs("d2", samples=20000))
    h = g.pair_feats["d2"]
    centres = (np.arange(64) + 0.5) / 64
    mean = h[0, 0] @ centres
    assert mean == pytest.approx(0.5214 / math.sqrt(3), abs=0.005)


def test_d2_between_opposite_faces_is_at_least_the_gap(graph):
    g = graph(F.box(1, 1, 1), features=pairs("d2", samples=2000))
    sp = graph(F.box(1, 1, 1), features=pairs("shortest_path")).pair_feats["shortest_path"]
    i, j = np.argwhere(sp == 2)[0]
    gap_bin = int(1 / math.sqrt(3) * 64) - 1  # distance >= 1 -> ratio >= 0.577
    assert g.pair_feats["d2"][i, j, :gap_bin].sum() == 0
    assert g.pair_feats["d2"][i, j].sum() == pytest.approx(1.0)


def test_a3_planar_region_has_mean_pi_over_three(graph):
    """Three exchangeable points in a plane: angle at the middle one averages pi/3, i.e. 1/3 after /pi."""
    g = graph(F.box(1, 1, 1), features=pairs("a3", samples=20000))
    centres = (np.arange(64) + 0.5) / 64
    assert g.pair_feats["a3"][0, 0] @ centres == pytest.approx(1 / 3, abs=0.01)


def test_seed_determinism(graph, step):
    path = step(F.pocket_box())
    a = bg.load(path, pairs("d2", "a3"))
    b = bg.load(path, pairs("d2", "a3"))
    c = bg.load(path, bg.FeatureSet(nodes=["area"], pairs=[("d2", {"seed": 1}), ("a3", {"seed": 1})]))
    for k in ("d2", "a3"):
        np.testing.assert_array_equal(a.pair_feats[k], b.pair_feats[k])
        assert not np.array_equal(a.pair_feats[k], c.pair_feats[k])
    assert c.meta["features"]["pair"][0]["params"]["seed"] == 1


def test_pair_feature_extension_and_errors(graph):
    reg = dict(bg.PAIR_FEATURES)
    try:
        bg.pair_feature("same_type", dtype="int64", shape=())(
            lambda ctx: (lambda t: (t[:, None] == t[None]).astype(np.int64))(
                np.array([int(ctx.surface(i).GetType()) for i in range(ctx.num_faces)])
            )
        )
        g = graph(F.cylinder(), features=bg.FeatureSet(pairs=["same_type"]))
        assert g.pair_feats["same_type"].shape == (3, 3)
        bg.pair_feature("wrong", dtype="float32", shape=())(lambda ctx: np.zeros((2, 2)))
        with pytest.raises(bg.FeatureError, match="wrong"):
            graph(F.cylinder(), features=bg.FeatureSet(pairs=["wrong"]))
        with pytest.raises(ValueError, match="already registered"):
            bg.pair_feature("d2", dtype="float32", shape=())(lambda ctx: None)
    finally:
        bg.PAIR_FEATURES.clear()
        bg.PAIR_FEATURES.update(reg)


# ---- schema 1.1 ------------------------------------------------------------------------------
def test_pair_feats_round_trip_and_schema_1_0_still_loads(graph, tmp_path):
    g = graph(F.pocket_box(), features="brepmfr")
    g.save(tmp_path / "g.npz")
    g2 = bg.BrepGraph.load(tmp_path / "g.npz")
    assert g2.schema_version == "1.1.0" and "pair/d2" not in g2.node_feats
    for k, v in g.pair_feats.items():
        np.testing.assert_array_equal(v, g2.pair_feats[k])
    assert "pair_feats:" in repr(g2) and "d2" in repr(g2)
    # an archive written by 0.1 (schema 1.0.0, no pair keys) must keep loading
    old = graph(F.box(), features="uvnet")
    old.schema_version = "1.0.0"
    old.save(tmp_path / "old.npz")
    assert bg.BrepGraph.load(tmp_path / "old.npz").pair_feats == {}


def test_validate_rejects_bad_pair_shape(graph):
    g = graph(F.box(), features="brepmfr")
    g.pair_feats["d2"] = g.pair_feats["d2"][:3]
    with pytest.raises(bg.BrepGraphError, match="pair feature"):
        g.validate()


def test_cli_info_lists_pair_features(step, tmp_path, capsys):
    out = tmp_path / "b.npz"
    assert cli.main(["convert", str(step(F.box())), str(out), "--preset", "brepmfr"]) == 0
    assert cli.main(["info", str(out)]) == 0
    text = capsys.readouterr().out
    assert "pair" in text and "d2" in text and "(6, 6, 64)" in text


# ---- adapters --------------------------------------------------------------------------------
def test_pyg_pair_feats(graph):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    g = graph(F.pocket_box(), features="brepmfr")
    d = g.to_pyg(pair_feats=("d2", "shortest_path"))
    assert d.d2.shape == (11, 11, 64) and d.d2.dtype == torch.float32
    assert d.shortest_path.dtype == torch.long
    with pytest.raises(bg.BrepGraphError, match="pair feature"):
        g.to_pyg(pair_feats=("nope",))


def test_dgl_adapter(graph):
    torch = pytest.importorskip("torch")
    try:
        import dgl  # noqa: F401
    except Exception:
        pytest.skip("dgl not importable here")
    g = graph(F.pocket_box())
    d = g.to_dgl()
    assert d.num_nodes() == 11 and d.num_edges() == 48
    assert d.ndata["uv_grid"].shape == (11, 7, 10, 10) and d.ndata["surface_type"].dtype == torch.long
    assert d.edata["u_grid"].shape == (48, 6, 10) and d.edata["convexity"].dtype == torch.long
    torch.testing.assert_close(d.edata["length"][:24], d.edata["length"][24:])
    src, dst = d.edges()
    assert torch.equal(src[:24], dst[24:]) and torch.equal(dst[:24], src[24:])


def test_dgl_missing_extra(graph, monkeypatch):
    monkeypatch.setitem(sys.modules, "dgl", None)
    with pytest.raises(bg.MissingExtraError, match=r"brepgraph\[dgl\]"):
        graph(F.box()).to_dgl()


# ---- visualization ---------------------------------------------------------------------------
@pytest.fixture
def plotly():
    return pytest.importorskip("plotly.graph_objects")


def test_tessellation_covers_every_face(plotly):
    from brepgraph.topology import build_topology
    from brepgraph.viz.mesh import tessellate

    for shape in (F.box(), F.pocket_box(), F.cylinder(), F.sphere(), F.filleted_box()):
        topo = build_topology(shape)
        verts, tris, tri_face, vert_face = tessellate(topo)
        assert set(tri_face) == set(range(topo.num_faces))
        assert tris.max() < len(verts) and len(vert_face) == len(verts)


def test_show_variants(graph, plotly):
    for shape in (F.box(), F.pocket_box(), F.filleted_box()):
        g = graph(shape)
        fig = bg.viz.show(g)  # categorical: surface_type
        assert isinstance(fig, plotly.Figure)
        names = [t.name for t in fig.data]
        assert "plane" in names and "faces" in names
        assert any(n and n.startswith("edge:") for n in names)
    g = graph(F.pocket_box())
    fig = bg.viz.show(g, "area", graph=True)  # scalar -> colour bar + graph overlay
    assert fig.data[0].colorbar.title.text == "area"
    assert "graph edges" in [t.name for t in fig.data] and "face nodes" in [t.name for t in fig.data]
    fig = bg.viz.show(g, ("uv_grid", "mask_fraction"), edges=False)
    assert all(not (t.name or "").startswith("edge:") for t in fig.data)
    g5 = graph(F.pocket_box(), features="brepmfr")
    assert isinstance(bg.viz.show(g5, "num_loops"), plotly.Figure)


def test_show_bad_color_by_lists_options(graph, plotly):
    g = graph(F.box())
    with pytest.raises(bg.BrepGraphError, match="valid color_by options.*surface_type"):
        bg.viz.show(g, "uv_grid")
    with pytest.raises(bg.BrepGraphError, match="unknown node feature"):
        bg.viz.show(g, "nope")


def test_show_detects_wrong_source_and_needs_source_for_memory_shapes(graph, plotly):
    g = graph(F.pocket_box())
    with pytest.raises(bg.BrepGraphError, match="faces but the graph has"):
        bg.viz.show(g, source=F.box())
    mem = bg.load(F.pocket_box())
    with pytest.raises(bg.BrepGraphError, match="source_path"):
        bg.viz.show(mem)
    assert isinstance(bg.viz.show(mem, source=F.pocket_box()), plotly.Figure)
    # same face count, different geometry -> areas differ
    with pytest.raises(bg.BrepGraphError, match="do not match"):
        bg.viz.show(bg.load(F.box(10, 20, 30)), source=F.box(11, 20, 30))


def test_show_face_debug_view(graph, plotly):
    g = graph(F.plate_with_hole(), grid_size=12)
    top = int(
        np.argmax(g.node_feats["area"] * (g.node_feats["num_loops"] if "num_loops" in g.node_feats else 1))
    )
    fig = bg.viz.show_face(g, top)
    names = [t.name for t in fig.data]
    assert "inside trim (mask=1)" in names and "outside trim (mask=0)" in names and "outward normals" in names
    with pytest.raises(bg.BrepGraphError):
        bg.viz.show_face(g, 99)


def test_viz_missing_extra(graph, monkeypatch):
    g = graph(F.box())
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(bg.MissingExtraError, match=r"brepgraph\[viz\]"):
        bg.viz.show(g)


def test_cli_view_writes_html(step, tmp_path, plotly):
    src = step(F.pocket_box())
    out = tmp_path / "view.html"
    assert cli.main(["view", str(src), "--color-by", "area", "--out", str(out), "--graph"]) == 0
    html = out.read_text(encoding="utf-8")
    assert "plotly" in html.lower() and len(html) > 10_000
    assert cli.main(["view", str(src), "--color-by", "uv_grid"]) == 1


def test_dgl_adapter_logic_against_a_stub(graph, monkeypatch):
    """Real dgl has no Windows wheel for current torch; exercise to_dgl's own logic with an API stub."""
    torch = pytest.importorskip("torch")

    class StubGraph:
        def __init__(self, edges, num_nodes):
            self.src, self.dst = edges
            self.n = num_nodes
            self.ndata, self.edata = {}, {}

    stub = type(sys)("dgl")
    stub.graph = lambda edges, num_nodes: StubGraph(edges, num_nodes)
    monkeypatch.setitem(sys.modules, "dgl", stub)
    g = graph(F.pocket_box())
    d = g.to_dgl()
    assert d.n == 11 and len(d.src) == 48
    assert torch.equal(d.src[:24], d.dst[24:]) and torch.equal(d.dst[:24], d.src[24:])
    assert d.src.dtype == torch.long
    assert d.ndata["uv_grid"].shape == (11, 7, 10, 10) and d.ndata["surface_type"].dtype == torch.long
    assert d.edata["u_grid"].shape == (48, 6, 10) and d.edata["convexity"].dtype == torch.long
    torch.testing.assert_close(d.edata["length"][:24], d.edata["length"][24:])
    assert (d.src[:24].numpy() == g.edge_index[0]).all()


# ---- graph view ------------------------------------------------------------------------------
def test_spring_layout_is_deterministic_and_bounded():
    from brepgraph.viz.graph_view import spring_layout

    ei = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    a, b = spring_layout(5, ei, 2), spring_layout(5, ei, 2)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (5, 2) and np.abs(a).max() <= 1 + 1e-9
    assert not np.array_equal(a, spring_layout(5, ei, 2, seed=1))
    assert spring_layout(1, np.zeros((2, 0), int), 3).shape == (1, 3)
    # a chain: neighbours end up closer than the two ends
    assert np.linalg.norm(a[0] - a[1]) < np.linalg.norm(a[0] - a[4])


def test_show_graph_variants(graph, plotly):
    g = graph(F.pocket_box())
    for layout in ("spring", "spring3d", "centroid"):
        fig = bg.viz.show_graph(g, layout=layout)
        names = [t.name for t in fig.data]
        assert "plane" in names and "edge: convex" in names and "edge: concave" in names
        node_points = sum(len(t.x) for t in fig.data if t.name in ("plane", "cylinder"))
        assert node_points == g.num_faces  # every face is drawn exactly once
        edge_info = next(t for t in fig.data if t.name == "edge info")
        assert len(edge_info.x) == g.num_edges
        assert (type(fig.data[0]).__name__ == "Scatter") == (layout == "spring")
    fig = bg.viz.show_graph(g, "area", size_by=None, labels=False)
    assert fig.data[-1].marker.colorbar.title.text == "area"
    hover = fig.data[-1].hovertext[0]
    assert "face 0" in hover and "area:" in hover and "surface_type:" in hover


def test_show_graph_needs_no_source_and_handles_edge_cases(graph, plotly):
    mem = bg.load(F.pocket_box())  # in-memory shape: no source_path, show() would refuse
    assert isinstance(bg.viz.show_graph(mem), plotly.Figure)
    sphere = graph(F.sphere())  # one node, no edges
    fig = bg.viz.show_graph(sphere)
    assert sum(len(t.x) for t in fig.data) == 1
    half = graph(F.half_cylinder())  # parallel edges between the same two faces
    fig = bg.viz.show_graph(half, layout="spring")
    assert len(next(t for t in fig.data if t.name == "edge info").x) == 6
    with pytest.raises(bg.BrepGraphError, match="valid layouts"):
        bg.viz.show_graph(half, layout="nope")
    with pytest.raises(bg.BrepGraphError, match="valid color_by"):
        bg.viz.show_graph(half, "uv_grid")
    light = bg.load(F.box(), bg.FeatureSet(nodes=["surface_type"]))  # no convexity, no centroid
    assert any(t.name == "edge: edge" for t in bg.viz.show_graph(light).data)
    with pytest.raises(bg.BrepGraphError, match="centroid"):
        bg.viz.show_graph(light, layout="centroid")


def test_cli_view_graph(step, tmp_path, plotly):
    out = tmp_path / "graph.html"
    assert cli.main(["view", str(step(F.pocket_box())), "--kind", "graph", "--out", str(out)]) == 0
    assert len(out.read_text(encoding="utf-8")) > 10_000


# ---- no side effects on caller-supplied shapes -------------------------------------------------
def test_load_does_not_leave_a_mesh_on_the_callers_shape():
    shape = F.pocket_box()
    assert not F.has_triangulation(shape)
    bg.load(shape, "brepmfr")  # pair features tessellate internally
    assert not F.has_triangulation(shape)


def test_show_does_not_leave_a_mesh_on_the_callers_shape(plotly):
    shape = F.pocket_box()
    g = bg.load(shape)
    bg.viz.show(g, source=shape)
    assert not F.has_triangulation(shape)


def test_release_mesh_clears_cached_triangles():
    from brepgraph.context import Context
    from brepgraph.topology import build_topology

    ctx = Context(build_topology(F.box()))
    assert ctx.triangles(0) is not None and F.has_triangulation(ctx.topology.solid)
    ctx.release_mesh()
    assert not F.has_triangulation(ctx.topology.solid)
    assert ctx.triangles(0) is not None  # transparently re-meshes


def test_viewing_a_loaded_archive_needs_source_or_embedded_path(step, tmp_path, plotly):
    shape = F.pocket_box()
    path = step(shape)
    bg.load(path).save(tmp_path / "plain.npz")
    loaded = bg.BrepGraph.load(tmp_path / "plain.npz")
    with pytest.raises(bg.BrepGraphError, match="saved without it"):
        bg.viz.show(loaded)
    assert isinstance(bg.viz.show(loaded, source=path), plotly.Figure)
    bg.load(path).save(tmp_path / "kept.npz", include_source_path=True)
    assert isinstance(bg.viz.show(bg.BrepGraph.load(tmp_path / "kept.npz")), plotly.Figure)
