import singler
import numpy
import knncolle


def test_classify_single_simple():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features = [str(i) for i in range(ref.shape[0])]
    built = singler.train_single(ref, labels, features)

    test = numpy.random.rand(10000, 50)
    output = singler.classify_single(test, built)
    assert output.shape[0] == 50
    assert sorted(output.column("scores").column_names) == ["A", "B", "C", "D", "E"]

    all_names = set(labels)
    for x in output.column("best"):
        assert x in all_names

    # Same results in parallel.
    poutput = singler.classify_single(test, built, num_threads = 2)
    assert output.column("best") == poutput.column("best")
    assert (output.column("delta") == poutput.column("delta")).all()


def test_classify_single_sanity():
    numpy.random.seed(69)

    ref = numpy.random.rand(10000, 10) + 1
    ref[:2000, :2] = 0
    ref[2000:4000, 2:4] = 0
    ref[4000:6000, 4:6] = 0
    ref[6000:8000, 6:8] = 0
    ref[8000:, 8:] = 0
    labels = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]

    features = [str(i) for i in range(ref.shape[0])]
    built = singler.train_single(ref, labels, features)

    test = numpy.random.rand(10000, 5) + 1
    test[2000:4000, 0] = 0  # B
    test[6000:8000, 1] = 0  # D
    test[:2000, 2] = 0  # A
    test[8000:, 3] = 0  # E
    test[4000:6000, 4] = 0  # C

    output = singler.classify_single(test, built)
    assert output.shape[0] == 5
    assert output.column("best") == ["B", "D", "A", "E", "C"]


def test_classify_single_features():
    ref = numpy.random.rand(10000, 10)
    labels = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]
    features = [str(i) for i in range(ref.shape[0])]

    # Using an exact NN algorithm so that the ordering doesn't change the results.
    built = singler.train_single(ref, labels, features, nn_parameters=knncolle.VptreeParameters())
    test = numpy.random.rand(10000, 50)
    output = singler.classify_single(test, built)

    # Checking that a different ordering of features in the test is respected.
    revfeatures = features[::-1]
    revbuilt = singler.train_single(ref, labels, features, test_features=revfeatures, nn_parameters=knncolle.VptreeParameters())
    revtest = numpy.array(test[::-1, :])
    revoutput = singler.classify_single(revtest, revbuilt)

    assert output.column("best") == revoutput.column("best")
    assert numpy.isclose(output.column("delta"), revoutput.column("delta")).all()
    assert numpy.isclose(output.column("scores").column("A"), revoutput.column("scores").column("A")).all()
