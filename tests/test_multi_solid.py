"""One graph per solid: load_all, the CLI flag, and re-loading the right solid in the viewer."""

import warnings

import numpy as np
import pytest

import brepgraph as bg
import fixtures as F
from brepgraph import cli


def _summary(g):
    return (g.num_faces, g.num_edges, sorted(np.round(g.node_feats["area"], 3).tolist()))


def test_load_still_rejects_several_solids_and_points_to_load_all(step):
    with pytest.raises(bg.UnsupportedShapeError, match="load_all"):
        bg.load(step(F.two_boxes()))


def test_load_all_returns_one_graph_per_solid(step):
    path = step(F.compound(F.box(10, 20, 30), F.pocket_box()))
    graphs = bg.load_all(path)
    assert len(graphs) == 2
    assert [g.meta["solid_index"] for g in graphs] == [0, 1]
    assert {g.meta["num_solids"] for g in graphs} == {2}
    assert all(g.meta["source"] == path.name for g in graphs)
    # each graph matches loading that solid on its own
    expected = sorted([_summary(bg.load(F.box(10, 20, 30))), _summary(bg.load(F.pocket_box()))])
    assert sorted(_summary(g) for g in graphs) == expected


def test_load_all_is_deterministic_and_works_for_single_solids(step):
    path = step(F.compound(F.box(), F.cylinder(), F.sphere()))
    a, b = bg.load_all(path), bg.load_all(path)
    assert [_summary(g) for g in a] == [_summary(g) for g in b]
    single = bg.load_all(step(F.pocket_box()))
    assert len(single) == 1 and single[0].meta["num_solids"] == 1
    assert _summary(single[0]) == _summary(bg.load(F.pocket_box()))


def test_repeated_instances_are_separate_solids(step):
    graphs = bg.load_all(step(F.two_boxes()))  # the same box placed twice
    assert len(graphs) == 2
    assert _summary(graphs[0]) == _summary(graphs[1])
    assert not np.allclose(graphs[0].node_feats["centroid"], graphs[1].node_feats["centroid"])


def test_load_all_normalizes_each_solid_on_its_own(step):
    graphs = bg.load_all(step(F.compound(F.box(1, 1, 1), F.box(100, 50, 50))), normalize=True)
    for g in graphs:
        lo, hi = np.array(g.meta["bbox"][:3]), np.array(g.meta["bbox"][3:])
        assert np.max(hi - lo) == pytest.approx(1.0, abs=1e-6)


def _fail_on_pocket(ctx, i):
    if ctx.num_faces == 11:  # only the pocketed box has 11 faces
        raise RuntimeError("deliberate failure")
    return 0.0


@pytest.fixture
def failing_feature():
    saved = dict(bg.NODE_FEATURES)
    bg.node_feature("fail_on_pocket", dtype="float32", shape=())(_fail_on_pocket)
    yield bg.FeatureSet(nodes=["fail_on_pocket", "area"])
    bg.NODE_FEATURES.clear()
    bg.NODE_FEATURES.update(saved)


def test_errors_raise_or_skip(step, failing_feature):
    path = step(F.compound(F.box(), F.pocket_box(), F.cylinder()))
    with pytest.raises(bg.FeatureError, match="fail_on_pocket"):
        bg.load_all(path, failing_feature)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graphs = bg.load_all(path, failing_feature, errors="skip")
    assert [g.meta["solid_index"] for g in graphs] == [0, 2]
    assert any("skipping solid 1 of 3" in str(w.message) for w in caught)
    with pytest.raises(ValueError, match="errors"):
        bg.load_all(path, errors="ignore")


def test_cli_all_solids(step, tmp_path, capsys):
    src = step(F.compound(F.box(), F.pocket_box()))
    out = tmp_path / "asm.npz"
    assert cli.main(["convert", str(src), str(out), "--all-solids"]) == 0
    assert not out.exists()
    first, second = tmp_path / "asm.solid0.npz", tmp_path / "asm.solid1.npz"
    assert first.exists() and second.exists()
    assert bg.BrepGraph.load(second).meta["solid_index"] == 1
    single = tmp_path / "one.npz"
    assert cli.main(["convert", str(step(F.box())), str(single), "--all-solids"]) == 0
    assert single.exists()  # single-solid files keep the plain name
    assert cli.main(["convert", str(src), str(tmp_path / "x.npz")]) == 1  # without the flag
    assert "load_all" in capsys.readouterr().err


def test_viewer_reloads_the_right_solid(step):
    go = pytest.importorskip("plotly.graph_objects")
    shape = F.compound(F.box(10, 20, 30), F.pocket_box())
    path = step(shape)
    for g in bg.load_all(path):
        assert isinstance(bg.viz.show(g), go.Figure)  # area/centroid check passes for the right solid
    second = bg.load_all(shape)[1]
    assert isinstance(bg.viz.show(second, source=shape), go.Figure)
    second.meta["solid_index"] = 5
    with pytest.raises(bg.BrepGraphError, match="only 2 solids"):
        bg.viz.show(second, source=shape)
