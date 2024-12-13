import singler
import numpy


def test_classify_integrated():
    all_features = [str(i) for i in range(10000)]
    test_features = [all_features[i] for i in range(0, 10000, 2)]
    test_set = set(test_features)

    ref1 = numpy.random.rand(8000, 10)
    labels1 = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features1 = [all_features[i] for i in range(8000)]
    built1 = singler.train_single(
        ref1, labels1, features1, test_features=test_set
    )

    ref2 = numpy.random.rand(8000, 6)
    labels2 = ["z", "y", "x", "z", "y", "z"]
    features2 = [all_features[i] for i in range(2000, 10000)]
    built2 = singler.train_single(
        ref2, labels2, features2, test_features=test_set
    )

    integrated = singler.train_integrated(
        test_features,
        ref_prebuilt=[built1, built2],
        ref_names=["first", "second"],
    )

    # Running the full analysis.
    test = numpy.random.rand(len(test_features), 50)
    results1 = singler.classify_single(test, built1)
    results2 = singler.classify_single(test, built2)

    results = singler.classify_integrated(
        test,
        results=[results1, results2],
        integrated_prebuilt=integrated,
    )

    assert results.shape[0] == 50
    assert set(results.column("best_reference")) == set(["first", "second"])
    assert results.column("scores").has_column("first")

    labels1_set = set(labels1)
    labels2_set = set(labels2)
    for i, b in enumerate(results.column("best_reference")):
        if b == "first":
            assert results1.column("best")[i] in labels1_set
        else:
            assert results2.column("best")[i] in labels2_set

    # Repeating without names.
    integrated_un = singler.train_integrated(
        test_features,
        ref_prebuilt=[built1, built2],
    )

    results_un = singler.classify_integrated(
        test,
        results=[results1, results2],
        integrated_prebuilt=integrated_un,
    )

    assert results_un.shape[0] == 50
    assert set(results_un.column("best_reference")) == set([0, 1])
    assert list(results_un.column("scores").column_names) == ['0', '1']
