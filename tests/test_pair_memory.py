"""Storage options for the dense face-pair histograms."""

import numpy as np
import pytest

import brepgraph as bg
import fixtures as F


def _pairs(dtype, samples=512):
    return bg.FeatureSet(
        pairs=[("d2", {"dtype": dtype, "samples": samples}), ("a3", {"dtype": dtype, "samples": samples})]
    )


@pytest.fixture(scope="module")
def graphs():
    shape = F.pocket_box()
    return {dtype: bg.load(shape, _pairs(dtype)) for dtype in ("float32", "float16", "uint16")}


def test_default_is_unchanged_float32_frequencies(graphs):
    default = bg.load(F.pocket_box(), "brepmfr")
    for name in ("d2", "a3"):
        np.testing.assert_array_equal(default.pair_feats[name], graphs["float32"].pair_feats[name])
        assert default.pair_feats[name].dtype == np.float32


@pytest.mark.parametrize("name", ["d2", "a3"])
def test_counts_and_half_precision_agree_with_float32(graphs, name):
    ref = graphs["float32"].pair_feats[name]
    counts = graphs["uint16"].pair_feats[name]
    half = graphs["float16"].pair_feats[name]
    assert counts.dtype == np.uint16 and half.dtype == np.float16
    assert (counts.sum(-1) == 512).all()  # exact counts, as in the paper
    np.testing.assert_allclose(counts / 512, ref, atol=1e-7)
    np.testing.assert_allclose(half.astype(np.float32), ref, atol=1e-3)
    assert counts.nbytes == half.nbytes == ref.nbytes // 2
    np.testing.assert_array_equal(counts, counts.transpose(1, 0, 2))


def test_meta_records_the_storage_choice(graphs):
    params = {f["name"]: f["params"] for f in graphs["uint16"].meta["features"]["pair"]}
    assert params["d2"]["dtype"] == params["a3"]["dtype"] == "uint16"


def test_invalid_storage_options_are_rejected():
    with pytest.raises(bg.FeatureError, match="dtype must be one of"):
        bg.load(F.box(), bg.FeatureSet(pairs=[("d2", {"dtype": "int8"})]))
    with pytest.raises(bg.FeatureError, match="samples <= 65535"):
        bg.load(F.box(), bg.FeatureSet(pairs=[("a3", {"dtype": "uint16", "samples": 70000})]))


def test_compact_histograms_round_trip_through_npz(graphs, tmp_path):
    for dtype in ("float16", "uint16"):
        graphs[dtype].save(tmp_path / f"{dtype}.npz")
        loaded = bg.BrepGraph.load(tmp_path / f"{dtype}.npz")
        assert loaded.pair_feats["d2"].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(loaded.pair_feats["d2"], graphs[dtype].pair_feats["d2"])


def test_pyg_accepts_count_histograms(graphs):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    data = graphs["uint16"].to_pyg(pair_feats=("d2",))
    assert data.d2.dtype == torch.long and int(data.d2[0, 0].sum()) == 512
