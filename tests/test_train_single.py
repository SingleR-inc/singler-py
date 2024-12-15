import singler
import numpy


def test_train_single_basic():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]
    markers = singler.get_classic_markers(ref, labels, features)

    built = singler.train_single(ref, labels, features, markers=markers)
    assert built.num_labels() == 5
    assert built.num_markers() < len(features)
    assert built.features == features
    assert built.labels.as_list() == ["A", "B", "C", "D", "E"]

    all_markers = built.marker_subset()
    assert len(all_markers) == built.num_markers()
    feat_set = set(features)
    for m in all_markers:
        assert m in feat_set

    # Same results when run in parallel.
    pbuilt = singler.train_single(
        ref, labels, features, markers=markers, num_threads=2
    )
    assert all_markers == pbuilt.marker_subset()


def test_train_single_markers():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]
    built = singler.train_single(ref, labels, features)

    markers = singler.get_classic_markers(ref, labels, features)
    mbuilt = singler.train_single(ref, labels, features, markers=markers)
    assert built.markers == mbuilt.markers


def test_train_single_dedup():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]
    features[0] = "1"
    built = singler.train_single(ref, labels, features)

    assert built.features == features[1:] # duplicates are ignored
    assert built._full_data.shape[0] == len(built.features)
    assert (built._full_data[0, :] == ref[0, :]).all()
    assert (built._full_data[1, :] == ref[2, :]).all()


def test_train_single_missing_label():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", None, "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]
    built = singler.train_single(ref, labels, features)
    assert built._full_data.shape[1] == len(labels) - 1
    assert (built._full_data[0,:] == ref[0,[0,1,2,3,5,6,7,8,9]]).all()


def test_train_single_restricted():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]

    keep = range(1, ref.shape[0], 3)
    restricted = [str(i) for i in keep]
    built = singler.train_single(
        ref, labels, features, restrict_to=set(restricted)
    )
    assert built.features == features

    expected = singler.train_single(ref[keep,:], labels, restricted)
    assert built.markers == expected.markers
    assert built.marker_subset() == expected.marker_subset()

    # Check that the actual C++ content is the same.
    test = numpy.random.rand(10000, 50)
    output = singler.classify_single(test, built)
    expected_output = singler.classify_single(test[keep,:], expected)
    assert (output.column("delta") == expected_output.column("delta")).all()
    assert output.column("best") == expected_output.column("best")
