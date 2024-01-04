from typing import Any, Sequence, Union

import biocutils as ut
from biocframe import BiocFrame
from mattress import TatamiNumericPointer, tatamize
from numpy import array, float64, int32, ndarray, uintp
from summarizedexperiment import SummarizedExperiment

from . import _cpphelpers as lib
from .build_integrated_references import IntegratedReferences


def classify_integrated_references(
    test_data: Any,
    results: list[Union[BiocFrame, Sequence]],
    integrated_prebuilt: IntegratedReferences,
    assay_type: Union[str, int] = 0,
    quantile: float = 0.8,
    num_threads: int = 1,
) -> BiocFrame:
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
            :py:meth:`~singler.classify_single_reference.classify_single_reference`
            on ``test_data`` with each reference. This may be either the full
            data frame or just the ``"best"`` column. References should be ordered
            as in ``integrated_prebuilt.reference_names``.

        integrated_prebuilt:
            Integrated reference object, constructed with
            :py:meth:`~singler.build_integrated_references.build_integrated_references`.

        assay_type: Assay containing the expression matrix,
            if `test_data` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        quantile:
            Quantile of the correlation distribution for computing the score for each label.
            Larger values increase sensitivity of matches at the expense of
            similarity to the average behavior of each label.

        num_threads:
            Number of threads to use during classification.

    Returns:
        A data frame containing the ``best_label`` across all
        references, defined as the assigned label in the best reference; the
        identity of the ``best_reference``, either as a name string or an
        integer index; the ``scores`` for each reference, as a nested
        BiocFrame; and the ``delta`` from the best to the second-best
        reference. Each row corresponds to a column of ``test``.
    """
    # Don't use _clean_matrix; the features are fixed so no filtering is possible at this point.
    if not isinstance(test_data, TatamiNumericPointer):
        if isinstance(test_data, SummarizedExperiment):
            test_data = test_data.assay(assay_type)

    test_ptr = tatamize(test_data)
    if test_ptr.nrow() != len(integrated_prebuilt.test_features):
        raise ValueError(
            "number of rows in 'test_data' should equal number of features in 'integrated_prebuilt'"
        )
    nc = test_ptr.ncol()

    all_labels = integrated_prebuilt.reference_labels
    nrefs = len(all_labels)
    coerced_labels = []

    all_refs = integrated_prebuilt.reference_names
    has_names = all_refs is not None
    if not has_names:
        all_refs = [str(i) for i in range(nrefs)]

    scores = {}
    score_ptrs = ndarray((nrefs,), dtype=uintp)
    assign_ptrs = ndarray((nrefs,), dtype=uintp)

    if len(all_refs) != len(results):
        raise ValueError(
            "length of 'results' should equal number of references in 'integrated_prebuilt'"
        )

    for i, r in enumerate(all_refs):
        current = ndarray((nc,), dtype=float64)
        scores[r] = current
        score_ptrs[i] = current.ctypes.data

        curlabs = results[i]
        if isinstance(curlabs, BiocFrame):
            curlabs = curlabs.column("best")
        if len(curlabs) != nc:
            raise ValueError(
                "each entry of 'results' should have results for all cells in 'test_data'"
            )

        ind = array(ut.match(curlabs, all_labels[i]), dtype=int32)
        coerced_labels.append(ind)
        assign_ptrs[i] = ind.ctypes.data

    best = ndarray((nc,), dtype=int32)
    delta = ndarray((nc,), dtype=float64)
    lib.classify_integrated_references(
        test_ptr.ptr,
        assign_ptrs.ctypes.data,
        integrated_prebuilt._ptr,
        quantile,
        score_ptrs.ctypes.data,
        best,
        delta,
        num_threads,
    )

    best_label = []
    for i, b in enumerate(best):
        if isinstance(results[b], BiocFrame):
            best_label.append(results[b].column("best")[i])
        else:
            best_label.append(results[b][i])

    if has_names:
        best = [all_refs[b] for b in best]

    scores_df = BiocFrame(scores, number_of_rows=nc)
    return BiocFrame(
        {
            "best_label": best_label,
            "best_reference": best,
            "scores": scores_df,
            "delta": delta,
        }
    )
