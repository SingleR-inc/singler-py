import singler
import numpy
import biocutils


def test_train_single_basic():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]
    markers = singler.get_classic_markers(ref, labels, features)

    built = singler.train_single(ref, labels, features, markers=markers)
    assert built.num_labels() == 5
    assert built.num_markers() < len(features)
    assert built.features == features
    assert built.labels == ["A", "B", "C", "D", "E"]

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
    assert (output.get_column("delta") == expected_output.get_column("delta")).all()
    assert output.get_column("best") == expected_output.get_column("best")


def test_train_single_scranpy():
    ref = numpy.random.rand(10000, 1000)
    labels = ["A", "B", "C", "D"] * 250
    features = ["gene_" + str(i) for i in range(ref.shape[0])]

    import scranpy
    effects = scranpy.score_markers(ref, labels, all_pairwise=True)

    def verify(ref_markers, effect_sizes, hard_limit, extra):
        all_labels = sorted(list(ref_markers.keys()))
        assert all_labels == sorted(effects.groups)

        for g1, group1 in enumerate(effects.groups):
            current_markers = ref_markers[group1]
            assert all_labels == sorted(list(current_markers.keys()))

            for g2, group2 in enumerate(effects.groups):
                if g1 == g2:
                    assert len(current_markers[group2]) == 0
                else:
                    my_effects = effect_sizes[g2, g1, :]
                    assert len(my_effects) == 10000
                    my_markers = current_markers[group2]
                    assert len(my_markers) > 0
                    my_markers_set = set(my_markers)
                    is_chosen = numpy.array([f in my_markers_set for f in features])
                    min_chosen = my_effects[is_chosen].min() 
                    assert min_chosen >= my_effects[numpy.logical_not(is_chosen)].max()
                    assert min_chosen > hard_limit
                    if extra is not None:
                        extra(group1, group2, my_markers)

    built = singler.train_single(ref, labels, features, marker_method="auc")
    verify(built.markers, effects.auc, 0.5, extra=None)

    built = singler.train_single(ref, labels, features, marker_method="cohens_d")
    def extra_cohen(n, n2, my_markers):
        assert len(my_markers) <= 10
        markerref = ref[biocutils.match(my_markers, features),:]
        left = markerref[:,[n == l for l in labels]].mean(axis=1)
        right = markerref[:,[n2 == l for l in labels]].mean(axis=1)
        assert (left > right).all()
    verify(built.markers, effects.cohens_d, 0, extra=extra_cohen)

    built = singler.train_single(ref, labels, features, marker_method="cohens_d", num_de=10000)
    def extra_cohen(n, n2, my_markers):
        assert len(my_markers) > 10
        markerref = ref[biocutils.match(my_markers, features),:]
        left = markerref[:,[n == l for l in labels]].mean(axis=1)
        right = markerref[:,[n2 == l for l in labels]].mean(axis=1)
        assert (left > right).all()
    verify(built.markers, effects.cohens_d, 0, extra=extra_cohen)

    # Responds to threshold specification.
    thresh_effects = scranpy.score_markers(ref, labels, threshold=1, all_pairwise=True)
    def extra_threshold(n, n2, my_markers):
        markerref = ref[biocutils.match(my_markers, features),:]
        left = markerref[:,[n == l for l in labels]].mean(axis=1)
        right = markerref[:,[n2 == l for l in labels]].mean(axis=1)
        assert (left > right + 1).all()
    verify(built.markers, effects.cohens_d, 0, extra=extra_cohen)


def test_train_single_aggregate():
    ref = numpy.random.rand(10000, 1000)
    labels = ["A", "B", "C", "D"] * 250
    features = ["gene_" + str(i) for i in range(ref.shape[0])]

    built = singler.train_single(ref, labels, features)
    aggr = singler.train_single(ref, labels, features, aggregate=True)
    assert aggr.markers == built.markers
    assert aggr.labels == built.labels
    assert aggr.features == built.features
    assert aggr.marker_subset() == built.marker_subset()
    assert aggr._full_data.shape[0] == built._full_data.shape[0]
    assert aggr._full_data.shape[1] < built._full_data.shape[1]
