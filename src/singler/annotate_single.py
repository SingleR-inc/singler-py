import warnings
from typing import Any, Optional, Sequence, Union

import biocframe
import summarizedexperiment

from .train_single import train_single 
from .classify_single import classify_single
from ._utils import _clean_matrix, _restrict_features


def annotate_single(
    test_data: Any,
    ref_data: Any,
    ref_labels: Sequence,
    test_features: Optional[Sequence] = None,
    ref_features: Optional[Sequence] = None,
    test_assay_type: Union[str, int] = 0,
    ref_assay_type: Union[str, int] = 0,
    test_check_missing: bool = False,
    ref_check_missing: bool = True,
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
            :py:func:`~singler.train_single.train_single`).

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays. Non-default assay
            types can be specified in ``classify_args``.

        ref_labels:
            Sequence of length equal to the number of columns of ``ref_data``,
            containing the label associated with each column.

        test_features:
            Sequence of length equal to the number of rows in ``test_data``,
            containing the feature identifier for each row. Alternatively
            ``None``, to use the row names of the experiment as features.

        ref_features:
            Sequence of length equal to the number of rows of ``ref_data``,
            containing the feature identifier for each row. Alternatively
            ``None``, to use the row names of the experiment as features.

        test_assay_type:
            Assay containing the expression matrix, if ``test_data`` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        ref_assay_type:
            Assay containing the expression matrix, if ``ref_data`` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        test_assay_type:
            Whether to remove rows with missing values from the test dataset.

        ref_assay_type:
            Whether to remove rows with missing values from the reference dataset.

        train_args:
            Further arguments to pass to
            :py:func:`~singler.train_single.train_single`.

        classify_args:
            Further arguments to pass to
            :py:func:`~singler.classify_single.classify_single`.

        num_threads:
            Number of threads to use for the various steps.

    Returns:
        A :py:class:`~biocframe.BiocFrame.BiocFrame` of labelling results, see
        :py:func:`~singler.classify_single.classify_single` for details.
    """
    if isinstance(ref_labels, str):
        warnings.warn(
            "setting 'ref_labels' to a column name of the column data is deprecated",
            category=DeprecationWarning
        )
        ref_labels = ref_data.get_column_data().column(ref_labels)

    test_data, test_features = _clean_matrix(
        test_data,
        test_features,
        assay_type=test_assay_type,
        check_missing=test_check_missing,
        num_threads=num_threads
    )

    ref_data, ref_features = _clean_matrix(
        ref_data,
        ref_features,
        assay_type=ref_assay_type,
        check_missing=ref_check_missing,
        num_threads=num_threads
    )

    # Pre-slicing the test dataset for consistency with annotate_integrated.
    test_data, test_features = _restrict_features(
        test_data,
        test_features,
        ref_features
    )

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
