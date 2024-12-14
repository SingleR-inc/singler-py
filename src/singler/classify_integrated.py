from typing import Any, Sequence, Union

import biocutils
import biocframe 
import mattress
import summarizedexperiment
import numpy

from . import lib_singler as lib
from .train_integrated import TrainedIntegratedReferences


def classify_integrated(
    test_data: Any,
    results: list[biocframe.BiocFrame],
    integrated_prebuilt: TrainedIntegratedReferences,
    assay_type: Union[str, int] = 0,
    quantile: float = 0.8,
    use_fine_tune: bool = True,
    fine_tune_threshold: float = 0.05,
    num_threads: int = 1,
) -> biocframe.BiocFrame:
    """Integrate classification results across multiple references for a single test dataset.

    Args:
        test_data:
            A matrix-like object where each row is a feature and each column
            is a test sample (usually a single cell), containing expression values.
            Normalized and/or transformed expression values are also acceptable as only
            the ranking is used within this function.

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays.

        results:
            List of classification results generated by running
            :py:func:`~singler.classify_single.classify_single` on
            ``test_data`` with each reference. References should be in the
            same order as that used to construct ``integrated_prebuilt``.

        integrated_prebuilt:
            Integrated reference object, constructed with
            :py:func:`~singler.train_integrated.train_integrated`.

        assay_type:
            Assay containing the expression matrix, if ``test_data`` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        quantile:
            Quantile of the correlation distribution for computing the score for each label.
            Larger values increase sensitivity of matches at the expense of
            similarity to the average behavior of each label.

        use_fine_tune:
            Whether fine-tuning should be performed. This improves accuracy for
            distinguishing between references with similar best labels but
            requires more computational work.

        fine_tune_threshold:
            Maximum difference from the maximum correlation to use in
            fine-tuning. All references above this threshold are used for
            another round of fine-tuning.

        num_threads:
            Number of threads to use during classification.

    Returns:
        A :py:class:`~biocframe.BiocFrame.BiocFrame` containing the
        ``best_label`` across all references, defined as the assigned label in
        the best reference; the identity of the ``best_reference``, either as a
        name string or an integer index; the ``scores`` for the best label in
        each reference, as a nested ``BiocFrame``; and the ``delta`` from the
        best to the second-best reference. Each row corresponds to a column of
        ``test_data``.
    """
    if isinstance(test_data, summarizedexperiment.SummarizedExperiment):
        test_data = test_data.assay(assay_type)

    if test_data.shape[0] != integrated_prebuilt._test_num_features: # TODO: move to singlepp.
        raise ValueError("number of rows in 'test_data' is not consistent with 'test_features=' used to create 'integrated_prebuilt'")

    ref_labs = integrated_prebuilt.reference_labels

    # Applying the sanity checks.
    if len(results) != len(ref_labs):
        raise ValueError("length of 'results' should equal the number of references")
    for i, curres in enumerate(results):
        if test_data.shape[1] != curres.shape[0]:
            raise ValueError("numbers of cells in 'results' are not identical")
        available = set(ref_labs[i])
        for l in curres.column("best"):
            if l not in available:
                raise ValueError("not all labels in 'results' are present in the corresponding reference")

    collated = []
    for i, curres in enumerate(results):
        available = set(ref_labs[i])
        collated.append(biocutils.match(curres.column("best"), available, dtype=numpy.uint32))

    test_ptr = mattress.initialize(test_data)
    best_ref, raw_scores, delta = lib.classify_integrated(
        test_ptr.ptr,
        collated,
        integrated_prebuilt._ptr,
        quantile,
        use_fine_tune,
        fine_tune_threshold,
        num_threads
    ) 

    by_ref = {}
    for i, b in enumerate(best_ref):
        if b not in by_ref:
            by_ref[b] = []
        by_ref[b].append(i)
    best_label = [None] * test_data.shape[1]
    for ref, which in by_ref.items():
        curbest = results[ref].column("best")
        for i in which:
            best_label[i] = curbest[i]

    all_refs = [str(i) for i in range(len(raw_scores))]
    scores = {}
    for i, l in enumerate(all_refs):
        scores[l] = biocframe.BiocFrame({ "label": results[i].column("best"), "score": raw_scores[i] })
    scores_df = biocframe.BiocFrame(scores, number_of_rows=test_data.shape[1], column_names=all_refs)

    return biocframe.BiocFrame({
        "best_label": best_label,
        "best_reference": best_ref,
        "scores": scores_df,
        "delta": delta,
    })
