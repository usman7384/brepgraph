"""Parallel loading: results, error reporting, crash isolation and the batch CLI."""

import json

import numpy as np
import pytest

import batch_helpers
import brepgraph as bg
import fixtures as F
from brepgraph import cli
from brepgraph.batch import CRASH_MESSAGE, find_cad_files, run_parallel


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    (root / "sub").mkdir(parents=True)
    F.write_step(F.box(), root / "box.step")
    F.write_step(F.pocket_box(), root / "sub" / "pocket.stp")
    F.write_step(F.compound(F.box(), F.cylinder()), root / "asm.step")
    (root / "broken.step").write_text("not a step file")
    (root / "notes.txt").write_text("ignored")
    return root


def test_run_parallel_keeps_input_order():
    items = list(range(20))
    assert run_parallel(batch_helpers.square, items, error_result=None, workers=3) == [x * x for x in items]
    assert run_parallel(batch_helpers.square, items, error_result=None, workers=1) == [x * x for x in items]


def test_a_crashing_worker_only_fails_its_own_item():
    items = [f"item{i}" for i in range(12)]
    items[5] = "CRASH-5"
    results = run_parallel(
        batch_helpers.crash_on_marked, items, error_result=lambda item, msg: f"error:{msg}", workers=3
    )
    assert results[5] == f"error:{CRASH_MESSAGE}"
    assert results[:5] + results[6:] == [f"ok:{x}" for x in items if "CRASH" not in x]


def test_load_many_matches_load_and_reports_failures(dataset):
    files = [p for p, _ in find_cad_files([dataset])]
    results = bg.load_many(files, workers=2)
    assert [r.source for r in results] == [str(p) for p in files]  # input order
    by_name = {r.source.split("\\")[-1].split("/")[-1]: r for r in results}
    assert not by_name["broken.step"].ok and by_name["broken.step"].error.startswith("FileReadError")
    assert not by_name["asm.step"].ok and "load_all" in by_name["asm.step"].error
    box = by_name["box.step"]
    assert box.ok and len(box.graphs) == 1
    expected = bg.load(dataset / "box.step")
    for k, v in expected.node_feats.items():
        np.testing.assert_array_equal(v, box.graphs[0].node_feats[k])


def test_load_many_all_solids_and_in_process_mode(dataset):
    results = bg.load_many([dataset / "asm.step", dataset / "box.step"], workers=1, all_solids=True)
    assert [len(r.graphs) for r in results] == [2, 1]
    assert [g.meta["solid_index"] for g in results[0].graphs] == [0, 1]


def test_load_many_passes_feature_options(dataset):
    (result,) = bg.load_many([dataset / "box.step"], "brepmfr", workers=1, grid_size=3)
    assert result.graphs[0].node_feats["uv_grid"].shape == (6, 7, 5, 5)  # preset sets size=5 explicitly
    assert result.graphs[0].pair_feats["d2"].shape == (6, 6, 64)


def test_find_cad_files(dataset):
    found = {rel.as_posix() for _, rel in find_cad_files([dataset])}
    assert found == {"asm.step", "box.step", "broken.step", "sub/pocket.stp"}


def test_cli_batch(dataset, tmp_path, capsys):
    out = tmp_path / "out"
    code = cli.main(["batch", str(dataset), "-o", str(out), "--workers", "2", "--all-solids"])
    assert code == 1  # broken.step failed
    assert (out / "box.npz").exists() and (out / "sub" / "pocket.npz").exists()
    assert (out / "asm.solid0.npz").exists() and (out / "asm.solid1.npz").exists()
    report = {
        r["source"].replace("\\", "/").split("/")[-1]: r
        for r in json.loads((out / "report.json").read_text())
    }
    assert report["broken.step"]["ok"] is False and "FileReadError" in report["broken.step"]["error"]
    assert report["box.step"]["outputs"][0]["faces"] == 6
    assert "4 files: 3 converted (4 graphs), 1 failed" in capsys.readouterr().out
    assert bg.BrepGraph.load(out / "sub" / "pocket.npz").num_faces == 11


def test_cli_batch_all_ok_and_errors(dataset, tmp_path, capsys):
    out = tmp_path / "ok"
    assert (
        cli.main(["batch", str(dataset / "box.step"), str(dataset / "sub"), "-o", str(out), "--workers", "1"])
        == 0
    )
    assert (out / "box.npz").exists() and (out / "pocket.npz").exists()
    capsys.readouterr()
    assert cli.main(["batch", str(tmp_path / "missing"), "-o", str(out)]) == 1
    assert "missing: no such file or folder" in capsys.readouterr().err
    (tmp_path / "empty").mkdir()
    assert cli.main(["batch", str(tmp_path / "empty"), "-o", str(out)]) == 1
    assert "no .step/.stp/.iges/.igs/.brep files found" in capsys.readouterr().err
