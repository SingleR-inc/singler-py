import singler
import numpy


def test_train_integrated():
    all_features = [str(i) for i in range(10000)]

    ref1 = numpy.random.rand(8000, 10)
    labels1 = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features1 = [all_features[i] for i in range(8000)]
    built1 = singler.train_single(ref1, labels1, features1)

    ref2 = numpy.random.rand(8000, 6)
    labels2 = ["z", "y", "x", "z", "y", "z"]
    features2 = [all_features[i] for i in range(2000, 10000)]
    built2 = singler.train_single(ref2, labels2, features2)

    test_features = [all_features[i] for i in range(0, 10000, 2)]
    integrated = singler.train_integrated(
        test_features,
        ref_prebuilt=[built1, built2]
    )

    assert list(integrated.reference_labels[0]) == ["A", "B", "C", "D", "E"]
    assert list(integrated.reference_labels[1]) == ["x", "y", "z"]

    # Works in parallel.
    pintegrated = singler.train_integrated(
        test_features,
        ref_prebuilt=[built1, built2],
        num_threads=2,
    )

    assert pintegrated.reference_labels == integrated.reference_labels
