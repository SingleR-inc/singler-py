from singler._utils import (
    _stable_intersect,
    _stable_union,
    _clean_matrix,
)
import numpy
import summarizedexperiment


def test_intersect():
    # Preserves the order in the first argument.
    out = _stable_intersect(["B", "C", "A", "D", "E"], ["A", "C", "E"])
    assert out == ["C", "A", "E"]

    # Works with more than 2 entries.
    out = _stable_intersect(["B", "C", "A", "D", "E"], ["A", "C", "E"], ["E", "A"])
    assert out == ["A", "E"]

    # Handles duplicates gracefully.
    out = _stable_intersect(
        ["B", "B", "C", "A", "D", "D", "E"], ["A", "A", "C", "E", "F", "F"]
    )
    assert out == ["C", "A", "E"]

    # Handles None-ness.
    out = _stable_intersect(
        ["B", None, "C", "A", None, "D", "E"], ["A", None, "C", "E", None, "F"]
    )
    assert out == ["C", "A", "E"]

    # Empty list.
    assert _stable_intersect() == []


def test_union():
    out = _stable_union(
        ["B", "C", "A", "D", "E"],
        [
            "A",
            "C",
            "E",
            "F",
        ],
    )
    assert out == ["B", "C", "A", "D", "E", "F"]

    # Works with more than 2 entries.
    out = _stable_union(["B", "C", "A", "D", "E"], ["A", "C", "K", "E"], ["G", "K"])
    assert out == ["B", "C", "A", "D", "E", "K", "G"]

    # Handles duplicates gracefully.
    out = _stable_union(
        ["B", "B", "C", "A", "D", "D", "E"], ["F", "A", "A", "C", "E", "F"]
    )
    assert out == ["B", "C", "A", "D", "E", "F"]

    # Handles None-ness.
    out = _stable_union(
        ["B", None, "C", "A", None, "D", "E"], ["A", None, "C", "E", None, "F"]
    )
    assert out == ["B", "C", "A", "D", "E", "F"]

    # Empty list.
    assert _stable_union() == []


def test_clean_matrix():
    out = numpy.random.rand(20, 10)
    features = ["FEATURE_" + str(i) for i in range(out.shape[0])]

    out2, feats = _clean_matrix(
        out, features, assay_type=None, check_missing=True, num_threads=1
    )
    assert feats == features
    assert (out2[1, :] == out[1, :]).all()
    assert (out2[:, 2] == out[:, 2]).all()

    out2, feats = _clean_matrix(
        out, features, assay_type=None, check_missing=False, num_threads=1
    )
    assert feats == features
    assert (out2[3, :] == out[3, :]).all()
    assert (out2[:, 4] == out[:, 4]).all()

    tmp = numpy.copy(out)
    tmp[0, 5] = numpy.nan
    out2, feats = _clean_matrix(
        tmp, features, assay_type=None, check_missing=True, num_threads=1
    )
    assert feats == features[1:]
    assert (out2[2, :] == out[3, :]).all()
    assert (out2[:, 4] == out[1:, 4]).all()

    se = summarizedexperiment.SummarizedExperiment({"counts": out})
    out2, feats = _clean_matrix(
        se, features, assay_type="counts", check_missing=True, num_threads=1
    )
    assert feats == features
    assert (out2[1, :] == out[1, :]).all()
    assert (out2[:, 2] == out[:, 2]).all()

    se2 = se.set_row_names(features)
    out2, feats = _clean_matrix(
        se2, None, assay_type="counts", check_missing=True, num_threads=1
    )
    assert feats.as_list() == features
    assert (out2[1, :] == out[1, :]).all()
    assert (out2[:, 2] == out[:, 2]).all()
