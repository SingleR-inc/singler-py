from typing import Any, Sequence, Union

import biocframe
import mattress
import summarizedexperiment

from . import _lib_singler as lib
from ._train_single import TrainedSingleReference 


def classify_single(
    test_data: Any,
    ref_prebuilt: TrainedSingleReference,
    assay_type: Union[str, int] = 0,
    quantile: float = 0.8,
    use_fine_tune: bool = True,
    fine_tune_threshold: float = 0.05,
    num_threads: int = 1,
) -> biocframe.BiocFrame:
    """
    Classify a test dataset against a reference by assigning labels from the latter to each column of the former using the SingleR algorithm.

    Args:
        test_data:
            A matrix-like object where each row is a feature and each column is a test sample (usually a single cell), containing expression values.
            Each entry may be an expression value of any kind; only the ranking within each column is used by this function.

            Alternatively, a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` containing such a matrix in one of its assays.

        ref_prebuilt:
            A pre-built reference created with :py:func:`~singler.train_single`.

        assay_type:
            Assay containing the expression matrix, if ``test_data`` is a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        quantile:
            Quantile of the correlation distribution for computing the score for each label.
            Larger values increase sensitivity of matches at the expense of similarity to the average behavior of each label.

        use_fine_tune:
            Whether fine-tuning should be performed.
            This improves accuracy for distinguishing between similar labels but requires more computational work.

        fine_tune_threshold:
            Maximum difference from the maximum correlation to use in fine-tuning.
            All labels above this threshold are used for another round of fine-tuning.

        num_threads:
            Number of threads to use during classification.

    Returns:
        A :py:class:`~BiocFrame.BiocFrame.BiocFrame` with one row per column of ``test_data``.
        This contains the following columns:

        - ``best``: a list containing the assigned label for each sample in ``test_data``. 
        - ``scores``: a nested ``BiocFrame`` where each column corresponds to a reference label and contains the score for that label across all test samples.
        - ``delta``: a double-precision NumPy array containing the difference in scores between the best and second-best label. 

        The metadata contains ``markers``, a list of the markers from each pairwise comparison between labels;
        and ``used``, a list containing the union of markers from all comparisons.

    Examples:
        >>> # Mocking up data.
        >>> import singler
        >>> ref = singler.mock_reference_data(num_replicates=8)
        >>> test = singler.mock_test_data(ref)
        >>> 
        >>> import scranpy
        >>> ref = scranpy.normalize_rna_counts_se(ref)
        >>> 
        >>> # Training and applying a classifier.
        >>> built = singler.train_single(ref, ref.get_column_data()["label"], ref.get_row_names())
        >>> res = singler.classify_single(test, built)
        >>> print(res)
        >>> import collections
        >>> print(collections.Counter(zip(res["best"], test.get_column_data()["label"])))
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
