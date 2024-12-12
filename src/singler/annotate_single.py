import warnings
from typing import Any, Optional, Sequence, Union

import biocframe

from .train_single import train_single 
from .classify_single import classify_single
from ._utils import _clean_matrix


def annotate_single(
    test_data: Any,
    ref_data: Any,
    ref_labels: Union[Sequence, str],
    test_features: Optional[Union[Sequence, str]] = None,
    ref_features: Optional[Union[Sequence, str]] = None,
    test_assay_type: Union[str, int] = 0,
    ref_assay_type: Union[str, int] = 0,
    check_missing: bool = True,
    train_args: dict = {},
    classify_args: dict = {},
    num_threads: int = 1,
) -> biocframe.BiocFrame:
    """Annotate a single-cell expression dataset based on the correlation
    of each cell to profiles in a labelled reference.

    Args:
        test_data:
            A matrix-like object representing the test dataset, where rows are
            features and columns are samples (usually cells). Entries should be expression
            values; only the ranking within each column will be used.

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays. Non-default assay
            types can be specified in ``classify_args``.

        ref_data:
            A matrix-like object representing the reference dataset, where rows
            are features and columns are samples. Entries should be expression values,
            usually log-transformed (see comments for the ``ref`` argument in
            :py:meth:`~singler.build_single_reference.build_single_reference`).

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays. Non-default assay
            types can be specified in ``classify_args``.

        ref_labels:
            If ``ref_data`` is a matrix-like object, ``ref_labels`` should be
            a sequence of length equal to the number of columns of ``ref_data``,
            containing the label associated with each column.

            Alternatively, if ``ref_data`` is a ``SummarizedExperiment``, 
            ``ref_labels`` may be a string specifying the label type to use,
            e.g., "main", "fine", "ont". It can also be set to `None`, to use 
            the `row_names` of the experiment as features.

        test_features:
            Sequence of length equal to the number of rows in
            ``test_data``, containing the feature identifier for each row.

            Alternatively, if ``test_data`` is a ``SummarizedExperiment``, ``test_features``
            may be a string speciying the column name in `row_data` that contains the
            features. It can also be set to `None`, to use the `row_names` of
            the experiment as features.

        ref_features:
            If ``ref_data`` is a matrix-like object, ``ref_features`` should be
            a sequence of length equal to the number of rows of ``ref_data``,
            containing the feature identifier associated with each row.

            Alternatively, if ``ref_data`` is a ``SummarizedExperiment``, 
            ``ref_features`` may be a string speciying the column name in `column_data`
            that contains the features. It can also be set to `None`, to use the 
            `row_names` of the experiment as features.

        test_assay_type:
            Assay containing the expression matrix, if `test_data` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        ref_assay_type:
            Assay containing the expression matrix, if `ref_data` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        train_args:
            Further arguments to pass to
            :py:meth:`~singler.train_single.train_single`.

        classify_args:
            Further arguments to pass to
            :py:meth:`~singler.classify_single.classify_single`.

        num_threads:
            Number of threads to use for the various steps.

    Returns:
        A :py:class:`~biocframe.BiocFrame.BiocFrame` of labelling results, see
        :py:meth:`~singler.classify_single.classify_single` for details.
    """
    test_data, test_features = _clean_matrix(
        test_data,
        test_features,
        assay_type=test_assay_type,
        check_missing=check_missing,
        num_threads=num_threads
    )
    if test_features is None:
        raise ValueError("could not determine 'test_features'") 

    ref_data, ref_features = _clean_matrix(
        ref_data,
        ref_features,
        assay_type=ref_assay_type,
        check_missing=check_missing,
        num_threads=num_threads
    )
    if ref_features is None:
        raise ValueError("could not determine 'ref_features'") 

    built = train_single(
        ref_data=ref_data,
        ref_labels=ref_labels,
        ref_features=ref_features,
        test_features=test_features,
        check_missing=False,
        num_threads=num_threads,
        **train_args,
    )

    return classify_single(
        test_data,
        ref_prebuilt=built,
        **classify_args,
        num_threads=num_threads,
    )
