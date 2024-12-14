from typing import Any, Optional, Sequence, Tuple, Union

import biocframe
import warnings

from ._utils import _clean_matrix, _restrict_features
from .train_single import train_single
from .train_integrated import train_integrated
from .classify_single import classify_single
from .classify_integrated import classify_integrated


def annotate_integrated(
    test_data: Any,
    ref_data: Sequence,
    ref_labels: list[Sequence],
    test_features: Optional[Sequence] = None,
    ref_features: Optional[list[Optional[Sequence]]] = None,
    test_assay_type: Union[str, int] = 0,
    ref_assay_type: Union[str, int] = "logcounts",
    test_check_missing: bool = False,
    ref_check_missing: bool = True,
    train_single_args: dict = {},
    classify_single_args: dict = {},
    train_integrated_args: dict = {},
    classify_integrated_args: dict = {},
    num_threads: int = 1,
) -> Tuple[list[biocframe.BiocFrame], biocframe.BiocFrame]:
    """Annotate a single-cell expression dataset based on the correlation
    of each cell to profiles in multiple labelled references, where the
    annotation from each reference is then integrated across references.

    Args:
        test_data:
            A matrix-like object representing the test dataset, where rows are
            features and columns are samples (usually cells). Entries should be
            expression values; only the ranking within each column is used.

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays.

        ref_data:
            Sequence consisting of one or more of the following:

            - A matrix-like object representing the reference dataset, where rows
              are features and columns are samples. Entries should be expression values,
              usually log-transformed (see comments for the ``ref_data`` argument in
              :py:func:`~singler.train_single.train_single`).
            - A ``SummarizedExperiment`` object containing such a matrix in its assays.

        ref_labels:
            Sequence of the same length as ``ref_data``. The ``i``-th entry
            should be a sequence of length equal to the number of columns of
            ``ref_data[i]``, containing the label associated with each column.

        test_features:
            Sequence of length equal to the number of rows in ``test_data``,
            containing the feature identifier for each row.  Alternatively
            ``None``, to use the row names of the experiment as features.

        ref_features:
            Sequence of the same length as ``ref_data``. The ``i``-th entry
            should be a sequence of length equal to the number of rows of
            ``ref_data[i]``, containing the feature identifier associated with
            each row. It can also be set to ``None`` to use the row names of
            the experiment as features.

            This can also be ``None`` to indicate that the row names should be
            used for all references, assuming ``ref_data`` only contains
            ``SummarizedExperiment`` objects.

        test_assay_type:
            Assay of ``test_data`` containing the expression matrix, if ``test_data`` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        test_check_missing:
            Whether to check for and remove missing (i.e., NaN) values from the test dataset.

        ref_assay_type:
            Assay containing the expression matrix for any entry of ``ref_data`` that is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        ref_check_missing:
            Whether to check for and remove missing (i.e., NaN) values from the reference datasets.

        train_single_args:
            Further arguments to pass to
            :py:func:`~singler.train_single.train_single`.

        classify_single_args:
            Further arguments to pass to
            :py:func:`~singler.classify_single.classify_single`.

        train_integrated_args:
            Further arguments to pass to
            :py:func:`~singler.train_integrated.train_integrated`.

        classify_integrated_args:
            Further arguments to pass to
            :py:func:`~singler.classify_integrated.classify_integrated`.

        num_threads:
            Number of threads to use for the various steps.

    Returns:
        Tuple where the first element contains per-reference results (i.e. a
        list of BiocFrame outputs, roughly equivalent to running
        :py:func:`~singler.annotate_single.annotate_single` on each reference)
        and the second element contains integrated results across references
        (i.e., a BiocFrame from
        :py:func:`~singler.classify_integrated.classify_integrated`).
    """
    nrefs = len(ref_data)
    if nrefs != len(ref_labels):
        raise ValueError("'ref_data' and 'ref_labels' must be the same length")
    if ref_features is None:
        ref_features = [None] * nrefs
    if nrefs != len(ref_features):
        raise ValueError("'ref_data' and 'ref_features' must be the same length")
    if nrefs != len(ref_labels):
        raise ValueError("'ref_data' and 'ref_labels' must be the same length")

    test_data, test_features = _clean_matrix(
        test_data,
        test_features,
        assay_type=test_assay_type,
        check_missing=test_check_missing,
        num_threads=num_threads,
    )

    all_ref_data = []
    all_ref_labels = []
    all_ref_features = []
    for r in range(nrefs):
        curref_labels = ref_labels[r]
        if isinstance(curref_labels, str):
            warnings.warn(
                "setting 'ref_labels' to a column name of the column data is deprecated",
                category=DeprecationWarning
            )
            curref_labels = ref_data[r].get_column_data().column(curref_labels)
        all_ref_labels.append(curref_labels)

        curref_data, curref_features = _clean_matrix(
            ref_data[r],
            ref_features[r],
            assay_type=ref_assay_type,
            check_missing=ref_check_missing,
            num_threads=num_threads,
        )

        all_ref_data.append(curref_data)
        all_ref_features.append(curref_features)

        # Pre-slicing the test dataset so that we force the downstream analysis
        # to only consider genes that are present in all datasets (test and
        # reference). This avoids the warning in classify_integrated().
        test_data, test_features = _restrict_features(
            test_data,
            test_features,
            curref_features
        )

    if test_data.shape[0] == 0:
        raise ValueError("no genes in common across test and reference datasets")

    all_built = []
    all_results = []

    for r in range(nrefs):
        curbuilt = train_single(
            ref_data=all_ref_data[r],
            ref_labels=all_ref_labels[r],
            ref_features=all_ref_features[r],
            test_features=test_features,
            **train_single_args,
            check_missing=False,
            num_threads=num_threads,
        )

        res = classify_single(
            test_data,
            ref_prebuilt=curbuilt,
            **classify_single_args,
            num_threads=num_threads,
        )

        all_built.append(curbuilt)
        all_results.append(res)

    ibuilt = train_integrated(
        test_features=test_features,
        ref_prebuilt=all_built,
        **train_integrated_args,
        num_threads=num_threads,
    )

    ires = classify_integrated(
        test_data=test_data,
        results=all_results,
        integrated_prebuilt=ibuilt,
        **classify_integrated_args,
        num_threads=num_threads,
    )

    return all_results, ires
