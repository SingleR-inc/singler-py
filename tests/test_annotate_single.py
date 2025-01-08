import singler
import numpy
import biocutils


def test_annotate_single_sanity():
    numpy.random.seed(123456) # ensure we don't get surprised by different results.

    ref = numpy.random.rand(10000, 10) + 1
    ref[:2000, :2] = 0
    ref[2000:4000, 2:4] = 0
    ref[4000:6000, 4:6] = 0
    ref[6000:8000, 6:8] = 0
    ref[8000:, 8:] = 0
    labels = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]

    test = numpy.random.rand(10000, 5) + 1
    test[2000:4000, 0] = 0  # B
    test[6000:8000, 1] = 0  # D
    test[:2000, 2] = 0  # A
    test[8000:, 3] = 0  # E
    test[4000:6000, 4] = 0  # C

    all_features = [str(i) for i in range(10000)]
    output = singler.annotate_single(
        test,
        test_features=all_features,
        ref_data=ref,
        ref_features=all_features,
        ref_labels=labels,
    )

    assert output.shape[0] == 5
    assert output.column("best") == ["B", "D", "A", "E", "C"]

    # Works correctly with aggregation.
    output = singler.annotate_single(
        test,
        test_features=all_features,
        ref_data=ref,
        ref_features=all_features,
        ref_labels=labels,
        train_args = { "aggregate": True }
    )

    assert output.shape[0] == 5
    assert output.column("best") == ["B", "D", "A", "E", "C"]

    # To mix it up a little, we're going to be taking every 2nd element of the
    # ref and every 3rd element of the test, just to make sure that the slicing
    # works as expected.
    rkeep = list(range(0, ref.shape[0], 2))
    tkeep = list(range(0, ref.shape[0], 3))
    output2 = singler.annotate_single(
        test[tkeep,:],
        test_features=biocutils.subset_sequence(all_features, tkeep),
        ref_data=ref[rkeep,:],
        ref_features=biocutils.subset_sequence(all_features, rkeep),
        ref_labels=labels,
    )

    assert output2.shape[0] == 5
    assert output2.column("best") == ["B", "D", "A", "E", "C"]


def test_annotate_single_intersect():
    ref = numpy.random.rand(10000, 10)
    ref_features = [str(i) for i in range(10000)]
    ref_labels = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]
    test = numpy.random.rand(10000, 50)
    test_features = [str(i + 2000) for i in range(10000)]

    output = singler.annotate_single(
        test,
        test_features=test_features,
        ref_data=ref,
        ref_features=ref_features,
        ref_labels=ref_labels,
    )

    built = singler.train_single(
        ref[2000:, :],
        ref_labels=ref_labels,
        ref_features=ref_features[2000:],
        test_features=test_features[:8000],
    )
    expected = singler.classify_single(test[:8000, :],  built)

    assert output.column("best") == expected.column("best")
    assert (output.column("delta") == expected.column("delta")).all()
    assert (
        output.column("scores").column("B") == expected.column("scores").column("B")
    ).all()

