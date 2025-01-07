import singler
import numpy


def test_aggregate_reference_full():
    ref = numpy.random.rand(10000, 20)
    letters = ["A", "B", "C", "D", "E"]
    labels = letters * 4
    features = ["gene_" + str(i) for i in range(ref.shape[0])]

    aggr = singler.aggregate_reference(ref, labels, features, power=0)
    aglab = aggr.get_column_data().get_column("label")
    assert sorted(aglab) == letters

    out = aggr.assay(0)
    for n in letters:
        keep = [l == n for l in labels]
        subref = ref[:,keep]
        assert numpy.allclose(out[:,aglab.index(n)], subref.mean(axis=1))


def test_aggregate_reference_partial():
    ref = numpy.random.rand(10000, 21)
    labels = ["A"] * 9 + ["B"] * 6 + ["C"] * 4 + ["D"] * 2
    features = ["gene_" + str(i) for i in range(ref.shape[0])]

    aggr = singler.aggregate_reference(ref, labels, features, power=0.5)
    aglab = aggr.get_column_data().get_column("label")
    assert aglab == [ "A", "A", "A", "B", "B", "C", "C", "D" ]
    assert [x.split("_")[0] for x in aggr.get_column_names()] == aglab

    # Contrived example with three elements per label to check the averaging is done correctly.
    labels = ["A", "B", "C", "D", "E", "F", "G"] * 3
    aggr1 = singler.aggregate_reference(ref, labels, features, power=0)
    aggr2 = singler.aggregate_reference(ref, labels, features, power=0.5)
    assert (aggr1.assay(0) == aggr2.assay(0)).all()
    assert aggr1.get_column_data().get_column("label") == aggr2.get_column_data().get_column("label")


def test_aggregate_reference_subset():
    numpy.random.seed(0)
    ref = numpy.random.rand(10000, 20)
    labels = ["A", "B", "C", "D"] * 5
    features = ["gene_" + str(i) for i in range(ref.shape[0])]

    sub = singler.aggregate_reference(ref, labels, features, subset_row=range(10, 50))
    ref = singler.aggregate_reference(ref[10:50,:], labels, features[10:50])

    assert sub.get_column_data().get_column("label") == ref.get_column_data().get_column("label")
    assert sub.get_column_names() == ref.get_column_names()
    assert sub.get_row_names().as_list() == features
    assert (ref.assay(0) == sub.assay(0)[10:50,:]).all()


def test_aggregate_reference_noop():
    ref = numpy.random.rand(10000, 20)
    labels = ["A", "B", "C", "D"] * 5
    features = ["gene_" + str(i) for i in range(ref.shape[0])]
    aggr = singler.aggregate_reference(ref, labels, features, power=1)
    assert aggr.get_column_data().get_column("label") == sorted(labels)
    assert (aggr.assay(0) == ref[:,numpy.argsort(labels, stable=True)]).all()


def test_aggregate_reference_skip():
    ref = numpy.random.rand(10000, 20)
    labels = ["A", "B", "C", "D"] * 5
    features = ["gene_" + str(i) for i in range(ref.shape[0])]
    out = singler.aggregate_reference(ref, labels, features)
    aggr = singler.aggregate_reference(ref, labels, features, rank=1000)
    assert out.shape == aggr.shape


def test_aggregate_reference_empty():
    ref = numpy.random.rand(10000, 0)
    features = ["gene_" + str(i) for i in range(ref.shape[0])]
    aggr = singler.aggregate_reference(ref, [], features)
    assert aggr.shape[0] == ref.shape[0]
    assert aggr.shape[1] == 0
