"""Errors, registry, serialisation, CLI and adapters."""

import json
import sys

import numpy as np
import pytest
from OCP.BRepTools import BRepTools

import brepgraph as bg
import fixtures as F
from brepgraph import cli, io, registry
from brepgraph.export import transforms


# ---- errors ----------------------------------------------------------------------------------
def test_text_file_renamed_step(tmp_path):
    p = tmp_path / "fake.step"
    p.write_text("this is not a STEP file\n")
    with pytest.raises(bg.FileReadError):
        bg.load(p)


def test_missing_and_unknown_extension(tmp_path):
    with pytest.raises(bg.FileReadError):
        bg.load(tmp_path / "nope.step")
    (tmp_path / "a.xyz").write_text("x")
    with pytest.raises(bg.FileReadError):
        bg.load(tmp_path / "a.xyz")


def test_two_solids_rejected(step):
    with pytest.raises(bg.UnsupportedShapeError, match="found 2"):
        bg.load(step(F.two_boxes()))


def test_open_shell_is_unsupported(step):
    # a lone shell contains no solid, so the loader reports UnsupportedShapeError
    with pytest.raises(bg.UnsupportedShapeError, match="no solid"):
        bg.load(step(F.open_shell()))


def test_invalid_solid_raises(monkeypatch):
    class Bad:
        def __init__(self, *_):
            pass

        def IsValid(self):
            return False

    monkeypatch.setattr(io, "BRepCheck_Analyzer", Bad)
    with pytest.raises(bg.InvalidSolidError):
        bg.load(F.box(), heal=False)
    with pytest.raises(bg.InvalidSolidError):  # healing "runs" but the re-check still fails
        bg.load(F.box())


def test_all_errors_share_a_base():
    for exc in (
        bg.FileReadError,
        bg.UnsupportedShapeError,
        bg.InvalidSolidError,
        bg.FeatureError,
        bg.MissingExtraError,
        bg.UnknownFeatureError,
        bg.SchemaVersionError,
    ):
        assert issubclass(exc, bg.BrepGraphError)


# ---- readers ---------------------------------------------------------------------------------
def test_in_memory_shape_and_brep_file(tmp_path):
    g1 = bg.load(F.pocket_box())
    path = tmp_path / "pocket.brep"
    assert BRepTools.Write_s(F.pocket_box(), str(path))
    g2 = bg.load(path)
    assert (g1.num_faces, g1.num_edges) == (g2.num_faces, g2.num_edges) == (11, 24)
    assert g1.meta["source"] is None and g2.meta["source"] == "pocket.brep"


# ---- registry --------------------------------------------------------------------------------
@pytest.fixture
def clean_registry():
    nodes, edges = dict(registry.NODE_FEATURES), dict(registry.EDGE_FEATURES)
    yield
    registry.NODE_FEATURES.clear()
    registry.NODE_FEATURES.update(nodes)
    registry.EDGE_FEATURES.clear()
    registry.EDGE_FEATURES.update(edges)


def test_duplicate_registration_raises(clean_registry):
    with pytest.raises(ValueError, match="already registered"):
        bg.node_feature("area", dtype="float32", shape=())(lambda ctx, i: 0.0)


def test_custom_feature_is_recorded_in_meta(clean_registry):
    @bg.node_feature("is_planar", dtype="int64", shape=(), version="7")
    def is_planar(ctx, i):
        return int(ctx.surface(i).GetType() == 0)

    @bg.edge_feature("half_length", dtype="float32", shape=(), defaults={"k": 0.5})
    def half_length(ctx, i, k):
        return ctx.edge_props(i).Mass() * k

    fs = bg.FeatureSet(nodes=["is_planar", "area"], edges=[("half_length", {"k": 0.25}), "length"])
    g = bg.load(F.box(10, 10, 10), fs)
    assert g.node_feats["is_planar"].tolist() == [1] * 6
    np.testing.assert_allclose(g.edge_feats["half_length"], 2.5)
    assert {"name": "is_planar", "version": "7", "params": {}} in g.meta["features"]["node"]
    assert set(g.node_feats) == {"is_planar", "area"}


def test_wrong_shape_names_the_feature(clean_registry):
    bg.node_feature("bad_shape", dtype="float32", shape=(3,))(lambda ctx, i: np.zeros(4))
    with pytest.raises(bg.FeatureError, match="bad_shape.*shape"):
        bg.load(F.box(), bg.FeatureSet(nodes=["bad_shape"]))


def test_feature_exception_and_nan_become_feature_errors(clean_registry):
    def boom(ctx, i):
        raise RuntimeError("kaput")

    bg.node_feature("boom", dtype="float32", shape=())(boom)
    bg.node_feature("nan", dtype="float32", shape=())(lambda ctx, i: float("nan"))
    with pytest.raises(bg.FeatureError, match=r"boom.*face 0.*kaput"):
        bg.load(F.box(), bg.FeatureSet(nodes=["boom"]))
    with pytest.raises(bg.FeatureError, match="NaN"):
        bg.load(F.box(), bg.FeatureSet(nodes=["nan"]))


def test_unknown_names_list_the_registered_ones():
    with pytest.raises(bg.UnknownFeatureError, match="area"):
        bg.FeatureSet(nodes=["nope"])
    with pytest.raises(bg.UnknownFeatureError, match="size"):
        bg.FeatureSet(nodes=[("uv_grid", {"bogus": 1})]).resolve()
    with pytest.raises(bg.UnknownFeatureError, match="uvnet"):
        bg.load(F.box(), "nope")


def test_grid_size_and_explicit_params():
    g = bg.load(F.box(), bg.FeatureSet(nodes=[("uv_grid", {"size": 4})], edges=["u_grid"]), grid_size=7)
    assert g.node_feats["uv_grid"].shape == (6, 7, 4, 4)
    assert g.edge_feats["u_grid"].shape == (12, 6, 7)


def test_feature_docs_do_not_drift():
    from pathlib import Path

    doc = Path(__file__).parent.parent / "docs" / "features.md"
    assert doc.read_text(encoding="utf-8") == registry.describe()


# ---- npz -------------------------------------------------------------------------------------
def test_npz_round_trip(tmp_path):
    g = bg.load(F.pocket_box())
    path = tmp_path / "g.npz"
    g.save(path)
    assert path.exists()  # no '.npz' appended / name mangled
    g2 = bg.BrepGraph.load(path)
    assert (g2.num_faces, g2.num_edges, g2.schema_version) == (
        g.num_faces,
        g.num_edges,
        g.schema_version,
    )
    np.testing.assert_array_equal(g.edge_index, g2.edge_index)
    for k, v in g.node_feats.items():
        np.testing.assert_array_equal(v, g2.node_feats[k])
        assert v.dtype == g2.node_feats[k].dtype
    for k, v in g.edge_feats.items():
        np.testing.assert_array_equal(v, g2.edge_feats[k])
    assert {k: v for k, v in g.meta.items() if k != "source_path"} == g2.meta
    json.dumps(g.meta)  # meta is plain JSON


def test_archives_do_not_embed_the_local_path_by_default(step, tmp_path):
    g = bg.load(step(F.box()))
    assert g.meta["source_path"]  # kept in memory so the viewer works on a freshly loaded graph
    g.save(tmp_path / "private.npz")
    assert "source_path" not in bg.BrepGraph.load(tmp_path / "private.npz").meta
    assert str(tmp_path) not in (tmp_path / "private.npz").read_bytes().decode("latin-1")
    g.save(tmp_path / "keep.npz", include_source_path=True)
    assert bg.BrepGraph.load(tmp_path / "keep.npz").meta["source_path"] == g.meta["source_path"]


def test_zero_edge_graph_round_trips(tmp_path):
    g = bg.load(F.sphere())
    g.save(tmp_path / "s.npz")
    g2 = bg.BrepGraph.load(tmp_path / "s.npz")
    assert g2.num_edges == 0 and g2.edge_feats["u_grid"].shape == (0, 6, 10)


def test_graphs_without_node_features_round_trip(tmp_path):
    """num_faces must come from the archive, not from the first node feature."""
    for name, features in {
        "edges": bg.FeatureSet(edges=["length"]),
        "pairs": bg.FeatureSet(pairs=["shortest_path"]),
        "empty": bg.FeatureSet(),
    }.items():
        g = bg.load(F.pocket_box(), features)
        g.save(tmp_path / f"{name}.npz")
        g2 = bg.BrepGraph.load(tmp_path / f"{name}.npz")
        assert (g2.num_faces, g2.num_edges) == (11, 24), name
        np.testing.assert_array_equal(g.edge_index, g2.edge_index)


def test_archive_without_num_faces_still_loads(tmp_path):
    """Archives written before num_faces was stored fall back to the first node feature."""
    g = bg.load(F.box())
    g.save(tmp_path / "new.npz")
    with np.load(tmp_path / "new.npz", allow_pickle=False) as z:
        legacy = {k: z[k] for k in z.files if k != "num_faces"}
    np.savez_compressed(tmp_path / "old.npz", **legacy)
    assert bg.BrepGraph.load(tmp_path / "old.npz").num_faces == 6


def test_incompatible_schema_rejected(tmp_path):
    g = bg.load(F.box())
    g.schema_version = "2.0.0"
    g.save(tmp_path / "g.npz")
    with pytest.raises(bg.SchemaVersionError):
        bg.BrepGraph.load(tmp_path / "g.npz")


def test_repr_lists_every_feature():
    text = repr(bg.load(F.box()))
    for name in (*bg.presets.uvnet.resolve()[0], *bg.presets.uvnet.resolve()[1]):
        assert name[0].name in text
    assert "(6, 7, 10, 10)" in text


# ---- transforms ------------------------------------------------------------------------------
def test_one_hot_and_standardize():
    g = bg.load(F.pocket_box())
    oh = transforms.one_hot(g, "convexity")
    assert oh.shape == (24, 3) and oh.sum() == 24 and oh.dtype == np.float32
    assert transforms.one_hot(g, "surface_type").shape == (11, 11)
    z = transforms.standardize(g, "area")
    assert abs(z.mean()) < 1e-5
    with pytest.raises(bg.BrepGraphError):
        transforms.one_hot(g, "area")


# ---- CLI -------------------------------------------------------------------------------------
def test_cli_convert_and_info(step, tmp_path, capsys):
    out = tmp_path / "box.npz"
    assert cli.main(["convert", str(step(F.box())), str(out), "--grid-size", "5"]) == 0
    assert "6 faces, 12 edges" in capsys.readouterr().out
    assert bg.BrepGraph.load(out).node_feats["uv_grid"].shape == (6, 7, 5, 5)
    assert cli.main(["info", str(out)]) == 0
    assert "uv_grid" in capsys.readouterr().out


def test_cli_embed_source_path_flag(step, tmp_path):
    src, plain, kept = step(F.box()), tmp_path / "a.npz", tmp_path / "b.npz"
    assert cli.main(["convert", str(src), str(plain)]) == 0
    assert cli.main(["convert", str(src), str(kept), "--embed-source-path"]) == 0
    assert "source_path" not in bg.BrepGraph.load(plain).meta
    assert "source_path" in bg.BrepGraph.load(kept).meta


def test_cli_error_is_one_line_exit_1(tmp_path, capsys):
    bad = tmp_path / "x.step"
    bad.write_text("nope")
    assert cli.main(["convert", str(bad), str(tmp_path / "o.npz")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("brepgraph: FileReadError") and "Traceback" not in err and err.count("\n") == 1


# ---- adapters --------------------------------------------------------------------------------
def test_missing_extra_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    with pytest.raises(bg.MissingExtraError, match=r"brepgraph\[pyg\]"):
        bg.load(F.box()).to_pyg()


def test_pyg_adapter():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    g = bg.load(F.pocket_box())
    d = g.to_pyg(x=("area", "centroid"), edge_attr=("length", "dihedral_angle"))
    assert d.num_nodes == 11 and d.edge_index.shape == (2, 48)
    assert d.edge_index.dtype == torch.long
    assert d.uv_grid.shape == (11, 7, 10, 10) and d.uv_grid.dtype == torch.float32
    assert d.u_grid.shape == (48, 6, 10) and d.convexity.shape == (48,)
    assert d.surface_type.dtype == torch.long and d.convexity.dtype == torch.long
    assert d.x.shape == (11, 4) and d.edge_attr.shape == (48, 2)
    # reverse direction carries the same edge features
    torch.testing.assert_close(d.length[:24], d.length[24:])
    assert torch.equal(d.edge_index[:, 24:], d.edge_index[:, :24].flip(0))
    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader([d, d], batch_size=2)))
    assert batch.uv_grid.shape == (22, 7, 10, 10) and batch.u_grid.shape == (96, 6, 10)
    assert int(batch.edge_index.max()) == 21
