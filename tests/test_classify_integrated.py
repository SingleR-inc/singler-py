import singler
import numpy


def test_classify_integrated_basic():
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
        ref_prebuilt=[built1, built2]
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
    assert set(results.column("best_reference")) == set([0, 1])
    assert list(results.column("scores").column_names) == ['0', '1']

    labels1_set = set(labels1)
    labels2_set = set(labels2)
    for i, b in enumerate(results.column("best_reference")):
        if b == "first":
            assert results1.column("best")[i] in labels1_set
        else:
            assert results2.column("best")[i] in labels2_set

    # Same results in parallel.
    presults = singler.classify_integrated(
        test,
        results=[results1, results2],
        integrated_prebuilt=integrated,
        num_threads = 2
    )

    assert presults.column("best_label") == results.column("best_label")
    assert (presults.column("best_reference") == results.column("best_reference")).all()
    assert (presults.column("delta") == results.column("delta")).all()


def test_classify_integrated_sanity():
    numpy.random.seed(42) # ensure we don't get surprised by different results.

    ref1 = numpy.random.rand(1000, 10)
    ref2 = numpy.random.rand(1000, 20)
    all_features = ["GENE_" + str(i) for i in range(ref1.shape[0])]

    ref1[0:100,1:5] = 0
    ref1[200:300,6:10] = 0
    ref2[100:200,1:10] = 0
    ref2[200:300,11:20] = 0

    lab1 = ["A"] * 5 + ["C"] * 5
    lab2 = ["B"] * 10 + ["C"] * 10

    test = numpy.random.rand(1000, 20)
    test[0:100,0:20:2] = 0
    test[100:200,1:20:2] = 0

    train1 = singler.train_single(ref1, lab1, all_features)
    pred1 = singler.classify_single(test, train1)
    train2 = singler.train_single(ref2, lab2, all_features)
    pred2 = singler.classify_single(test, train2)

    integrated = singler.train_integrated(
        all_features,
        ref_prebuilt=[train1, train2],
    )
    results = singler.classify_integrated(
        test,
        results=[pred1, pred2],
        integrated_prebuilt=integrated,
    )

    assert results.column("best_label") == ["A", "B"] * 10
    assert list(results.column("best_reference")) == [0, 1] * 10
