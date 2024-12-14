from typing import Any, Optional, Sequence, Union

import delayedarray
import mattress
import numpy
import biocutils

from . import lib_singler as lib
from ._utils import (
    _clean_matrix,
    _create_map,
    _stable_intersect,
    _stable_union,
)


def _get_classic_markers_raw(ref_ptrs: list, ref_labels: list, ref_features: list, num_de=None, num_threads=1):
    nrefs = len(ref_ptrs)

    # We assume that ref_ptrs and ref_features contains the outputs of
    # _clean_matrix, so there's no need to re-check their consistency.
    for i, x in enumerate(ref_ptrs):
        nc = x.ncol()
        if nc != len(ref_labels[i]):
            raise ValueError(
                "number of columns of 'ref' should be equal to the length of the corresponding 'labels'"
            )

    # Defining the intersection of features.
    common_features = _stable_intersect(*ref_features)
    if len(common_features) == 0:
        for r in ref_ptrs:
            if r.nrow():
                raise ValueError("no common feature names across 'features'")

    common_features_map = _create_map(common_features)

    # Computing medians for each group within each median.
    ref2 = []
    ref2_ptrs = []
    tmp_labels = []

    for i, x in enumerate(ref_ptrs):
        survivors = []
        remap = [None] * len(common_features)
        for j, f in enumerate(ref_features[i]):
            if f is not None and f in common_features_map:
                survivors.append(j)
                remap[common_features_map[f]] = len(survivors) - 1

        da = delayedarray.DelayedArray(x)[survivors, :]
        ptr = mattress.initialize(da)
        med, lev = ptr.row_medians_by_group(ref_labels[i], num_threads=num_threads)
        tmp_labels.append(lev)

        finalptr = mattress.initialize(med[remap, :])
        ref2.append(finalptr)
        ref2_ptrs.append(finalptr.ptr)

    ref_labels = tmp_labels

    # Defining the union of labels across all references.
    common_labels = _stable_union(*ref_labels)
    common_labels_map = _create_map(common_labels)

    labels2 = []
    for i, lab in enumerate(ref_labels):
        converted = numpy.ndarray(len(lab), dtype=numpy.uint32)
        for j, x in enumerate(lab):
            converted[j] = common_labels_map[x]
        labels2.append(converted)

    # Finally getting around to calling markers.
    if num_de is None:
        num_de = -1
    elif num_de <= 0:
        raise ValueError("'num_de' should be positive")

    raw_markers = lib.find_classic_markers(
        len(common_labels),
        len(common_features),
        ref2_ptrs,
        labels2,
        num_de,
        num_threads
    )

    return raw_markers, common_labels, common_features


def get_classic_markers(
    ref_data: Union[Any, list[Any]],
    ref_labels: Union[Sequence, list[Sequence]],
    ref_features: Union[Sequence, list[Sequence]],
    assay_type: Union[str, int] = "logcounts",
    check_missing: bool = True,
    num_de: Optional[int] = None,
    num_threads: int = 1,
) -> dict[Any, dict[Any, list]]:
    """Compute markers from a reference using the classic SingleR algorithm. This is typically done for reference
    datasets derived from replicated bulk transcriptomic experiments.

    Args:
        ref_data:
            A matrix-like object containing the log-normalized expression
            values of a reference dataset.  Each column is a sample and each
            row is a feature.
            
            Alternatively, this can be a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing a matrix-like object in one of its assays.
            
            Alternatively, a list of such matrices or ``SummarizedExperiment``
            objects, typically for multiple batches of the same reference; it
            is assumed that different batches exhibit at least some overlap in
            their ``ref_features`` and ``ref_labels``.

        ref_labels:
            If ``ref_data`` is not a list, ``ref_labels`` should be a sequence
            of length equal to the number of columns of ``ref_data``,
            containing a label (usually a string) for each column.

            If ``ref_data`` is a list, ``ref_labels`` should also be a list of
            the same length. Each entry should be a sequence of length equal to
            the number of columns of the corresponding entry of ``ref_data``.

        ref_features:
            If ``ref_data`` is not a list, ``ref_features`` should be a
            sequence of length equal to the number of rows of ``ref_data``,
            containing the feature name (usually a string) for each row.

            If ``ref_data`` is a list, ``ref_features`` should also be a list
            of the same length. Each entry should be a sequence of length
            equal to the number of rows of the corresponding entry of ``ref``.

        assay_type:
            Name or index of the assay of interest, if ``ref`` is or contains
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            objects.

        check_missing:
            Whether to check for and remove rows with missing (NaN) values in
            the reference matrices. This can be set to False if it is known
            that no NaN values exist.

        num_de:
            Number of differentially expressed genes to use as markers for each
            pairwise comparison between labels. If ``None``, an appropriate
            number of genes is automatically determined.

        num_threads:
            Number of threads to use for the calculations.

    Returns:
        A dictionary of dictionary of lists containing the markers for each
        pairwise comparison between labels, i.e., ``markers[a][b]`` contains
        the upregulated markers for label ``a`` over label ``b``.
    """
    if not isinstance(ref_data, list):
        ref_data = [ref_data]
        ref_labels = [ref_labels]
        ref_features = [ref_features]

    nrefs = len(ref_data)
    if nrefs != len(ref_labels):
        raise ValueError("length of 'ref' and 'labels' should be the same")
    if nrefs != len(ref_features):
        raise ValueError("length of 'ref' and 'features' should be the same")

    ref_ptrs = []
    cleaned_features = []
    for i in range(nrefs):
        r, f = _clean_matrix(
            ref_data[i],
            ref_features[i],
            assay_type=assay_type,
            check_missing=check_missing,
            num_threads=num_threads,
        )
        ref_ptrs.append(mattress.initialize(r))
        cleaned_features.append(f)

    raw_markers, common_labels, common_features = _get_classic_markers_raw(
        ref_ptrs=ref_ptrs,
        ref_labels=ref_labels,
        ref_features=cleaned_features,
        num_de=num_de,
        num_threads=num_threads,
    )

    markers = {}
    for i, x in enumerate(common_labels):
        current = {}
        for j, y in enumerate(common_labels):
            current[y] = biocutils.StringList(common_features[k] for k in raw_markers[i][j])
        markers[x] = current
    return markers


def number_of_classic_markers(num_labels: int) -> int:
    """Compute the number of markers to detect for a given number of labels,
    using the classic SingleR marker detection algorithm.

    Args:
        num_labels: Number of labels.

    Returns:
        int: Number of markers.
    """
    return lib.number_of_classic_markers(num_labels)
