import singler
import numpy
import biocutils


def test_annotate_integrated_basic():
    all_features = [str(i) for i in range(10000)]

    ref1 = numpy.random.rand(8000, 10)
    labels1 = ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A"]
    features1 = [all_features[i] for i in range(8000)]

    ref2 = numpy.random.rand(8000, 6)
    labels2 = ["z", "y", "x", "z", "y", "z"]
    features2 = [all_features[i] for i in range(2000, 10000)]

    test_features = [all_features[i] for i in range(0, 10000, 2)]
    test = numpy.random.rand(len(test_features), 50)

    single_results, integrated_results = singler.annotate_integrated(
        test,
        test_features=test_features,
        ref_data=[ref1, ref2],
        ref_labels=[labels1, labels2],
        ref_features=[features1, features2],
    )

    assert len(single_results) == 2
    assert set(single_results[0].column("best")) == set(labels1)
    assert set(single_results[1].column("best")) == set(labels2)
    assert set(integrated_results.column("best_reference")) == set([0, 1])


def test_annotate_integrated_sanity():
    numpy.random.seed(6969) # ensure we don't get surprised by different results.

    ref1 = numpy.random.rand(1000, 10)
    ref2 = numpy.random.rand(1000, 20)
    all_features = ["GENE_" + str(i) for i in range(ref1.shape[0])]

    ref1[0:100,1:5] = 0
    ref1[200:300,6:10] = 0
    ref2[100:200,1:10] = 0
    ref2[200:300,11:20] = 0

    labels1 = ["A"] * 5 + ["C"] * 5
    labels2 = ["B"] * 10 + ["C"] * 10

    test = numpy.random.rand(1000, 20)
    test[0:100,0:20:2] = 0
    test[100:200,1:20:2] = 0

    single_results, integrated_results = singler.annotate_integrated(
        test,
        test_features=all_features,
        ref_data=[ref1, ref2],
        ref_labels=[labels1, labels2],
        ref_features=[all_features, all_features],
    )
    assert integrated_results.column("best_label") == ["A", "B"] * 10
    assert list(integrated_results.column("best_reference")) == [0, 1] * 10

    # To mix it up a little, we're going to be taking every 2nd element of the
    # ref1 and every 3rd element of ref2, just to make sure that the slicing
    # works as expected.
    rkeep1 = list(range(0, ref1.shape[0], 2))
    rkeep2 = list(range(0, ref2.shape[0], 3))
    single_results2, integrated_results2 = singler.annotate_integrated(
        test,
        test_features=all_features,
        ref_data=[
            ref1[rkeep1,:],
            ref2[rkeep2,:]
        ],
        ref_features=[
            biocutils.subset_sequence(all_features, rkeep1),
            biocutils.subset_sequence(all_features, rkeep2)
        ],
        ref_labels=[labels1, labels2],
    )
    assert list(integrated_results2.column("best_reference")) == [0, 1] * 10
    assert integrated_results2.column("best_label") == ["A", "B"] * 10
