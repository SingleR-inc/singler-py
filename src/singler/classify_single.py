from typing import Any, Sequence, Union

import biocframe
import mattress
import summarizedexperiment

from . import lib_singler as lib
from .train_single import TrainedSingleReference 


def classify_single(
    test_data: Any,
    ref_prebuilt: TrainedSingleReference,
    assay_type: Union[str, int] = 0,
    quantile: float = 0.8,
    use_fine_tune: bool = True,
    fine_tune_threshold: float = 0.05,
    num_threads: int = 1,
) -> biocframe.BiocFrame:
    """Classify a test dataset against a reference by assigning labels from the latter to each column of the former
    using the SingleR algorithm.

    Args:
        test_data:
            A matrix-like object where each row is a feature and each column
            is a test sample (usually a single cell), containing expression values.
            Normalized and transformed expression values are also acceptable as only
            the ranking is used within this function.

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays.

        ref_prebuilt:
            A pre-built reference created with
            :py:func:`~singler.train_single.train_single`.

        assay_type:
            Assay containing the expression matrix, if ``test_data`` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        quantile:
            Quantile of the correlation distribution for computing the score for each label.
            Larger values increase sensitivity of matches at the expense of
            similarity to the average behavior of each label.

        use_fine_tune:
            Whether fine-tuning should be performed. This improves accuracy for distinguishing
            between similar labels but requires more computational work.

        fine_tune_threshold:
            Maximum difference from the maximum correlation to use in fine-tuning.
            All labels above this threshold are used for another round of fine-tuning.

        num_threads:
            Number of threads to use during classification.

    Returns:
        A :py:class:`~BiocFrame.BiocFrame.BiocFrame` containing the ``best``
        label, the ``scores`` for each label as a nested ``BiocFrame``, and the
        ``delta`` from the best to the second-best label. Each row corresponds
        to a column of ``test``. The metadata contains ``markers``, a list of
        the markers from each pairwise comparison between labels; and ``used``,
        a list containing the union of markers from all comparisons.
    """
    if isinstance(test_data, summarizedexperiment.SummarizedExperiment):
        test_data = test_data.assay(assay_type)

    test_ptr = mattress.initialize(test_data)

    best, raw_scores, delta = lib.classify_single(
        test_ptr.ptr,
        ref_prebuilt._ptr,
        quantile,
        use_fine_tune,
        fine_tune_threshold,
        num_threads
    )

    all_labels = ref_prebuilt.labels
    scores = {}
    for i, l in enumerate(all_labels):
        scores[l] = raw_scores[i]
    scores_df = biocframe.BiocFrame(scores, number_of_rows=test_data.shape[1], column_names=all_labels)

    output = biocframe.BiocFrame({
        "best": [all_labels[b] for b in best], 
        "scores": scores_df, 
        "delta": delta
    })
    output = output.set_metadata({
        "used": ref_prebuilt.marker_subset(),
        "markers": ref_prebuilt.markers,
    })

    return output
