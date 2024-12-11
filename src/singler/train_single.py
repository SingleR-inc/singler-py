from typing import Any, Literal, Optional, Sequence, Union

import biocutils
import numpy

from . import lib_singler as lib
from ._utils import _clean_matrix, _factorize
from .get_classic_markers import _get_classic_markers_raw, _markers_to_dict


class TrainedSingleReference:
    """A prebuilt reference object, typically created by
    :py:meth:`~singler.build_single_reference.build_single_reference`. This is intended for advanced users only and
    should not be serialized.
    """

    def __init__(
        self,
        ptr,
        labels: Sequence,
        features: Sequence,
        markers: dict[Any, dict[Any, Sequence]],
    ):
        self._ptr = ptr
        self._features = features
        self._labels = labels
        self._markers = markers

    def num_markers(self) -> int:
        """
        Returns:
            Number of markers to be used for classification. This is the
            same as the size of the array from :py:meth:`~marker_subset`.
        """
        return lib.get_num_markers_from_single_reference(self._ptr)

    def num_labels(self) -> int:
        """
        Returns:
            Number of unique labels in this reference.
        """
        return lib.get_num_labels_from_single_reference(self._ptr)

    @property
    def features(self) -> list:
        """The universe of features known to this reference."""
        return self._features

    @property
    def labels(self) -> Sequence:
        """Unique labels in this reference."""
        return self._labels

    @property
    def markers(self) -> dict[Any, dict[Any, list]]:
        """Markers for every pairwise comparison between labels."""
        return self._markers

    def marker_subset(self, indices_only: bool = False) -> Union[numpy.ndarray, list]:
        """
        Args:
            indices_only:
                Whether to return the markers as indices
                into :py:attr:`~features`, or as a list of feature identifiers.

        Returns:
            If ``indices_only = False``, a list of feature identifiers for the markers.

            If ``indices_only = True``, a NumPy array containing the integer indices of
            features in ``features`` that were chosen as markers.
        """
        buffer = lib.get_markers_from_single_reference(self._ptr)
        if indices_only:
            return buffer
        else:
            return [self._features[i] for i in buffer]


def _markers_from_dict(markers: dict[Any, dict[Any, Sequence]], labels: Sequence, features: Sequence):
    fmapping = {}
    for i, x in enumerate(features):
        fmapping[x] = i

    outer_instance = [None] * len(labels)
    for outer_i, outer_k in enumerate(labels):
        inner_instance = [None] * len(labels)

        for inner_i, inner_k in enumerate(labels):
            current = markers[outer_k][inner_k]
            mapped = []
            for x in current:
                if x in fmapping:  # just skipping features that aren't present.
                    mapped.append(fmapping[x])
            inner_instance[inner_i] = numpy.array(mapped, dtype=numpy.uint32)

        outer_instance[outer_i] = inner_instance

    return outer_instance


def train_single(
    ref_data: Any,
    ref_labels: Sequence,
    ref_features: Sequence,
    test_features: Optional[Sequence] = None,
    assay_type: Union[str, int] = "logcounts",
    check_missing: bool = True,
    markers: Optional[dict[Any, dict[Any, Sequence]]] = None,
    marker_method: Literal["classic"] = "classic",
    marker_args: dict = {},
    approximate: bool = True,
    num_threads: int = 1,
    ) -> TrainedSingleReference:
    """Build a single reference dataset in preparation for classification.

    Args:
        ref_data:
            A matrix-like object where rows are features, columns are
            reference profiles, and each entry is the expression value.
            If `markers` is not provided, expression should be normalized
            and log-transformed in preparation for marker prioritization via
            differential expression analyses. Otherwise, any expression values
            are acceptable as only the ranking within each column is used.

            Alternatively, a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in one of its assays.

        ref_labels:
            Sequence of labels for each reference profile, i.e., column in ``ref_data``.

        ref_features:
            Sequence of identifiers for each feature, i.e., row in ``ref_data``.

        test_features:
            Sequence of identifiers for each feature in the test dataset.

        assay_type:
            Assay containing the expression matrix,
            if `ref_data` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        check_missing:
            Whether to check for and remove rows with missing (NaN) values
            from ``ref_data``.

        markers:
            Upregulated markers for each pairwise comparison between labels.
            Specifically, ``markers[a][b]`` should be a sequence of features
            that are upregulated in ``a`` compared to ``b``. All such features
            should be present in ``features``, and all labels in ``labels``
            should have keys in the inner and outer dictionaries.

        marker_method:
            Method to identify markers from each pairwise comparisons between
            labels in ``ref_data``.  If "classic", we call
            :py:meth:`~singler.get_classic_markers.get_classic_markers`.
            Only used if ``markers`` is not supplied.

        marker_args:
            Further arguments to pass to the chosen marker detection method.
            Only used if ``markers`` is not supplied.

        approximate:
            Whether to use an approximate neighbor search to compute scores
            during classification.

        num_threads:
            Number of threads to use for reference building.

    Returns:
        The pre-built reference, ready for use in downstream methods like
        :py:meth:`~singler.classify_single_reference.classify_single`.
    """

    ref_ptr, ref_features = _clean_matrix(
        ref_data,
        ref_features,
        assay_type=assay_type,
        check_missing=check_missing,
        num_threads=num_threads,
    )

    ref_ptr, ref_features = _restrict_features(ref_ptr, ref_features, restrict_to)

    if markers is None:
        if marker_method == "classic":
            mrk, lablev, ref_features = _get_classic_markers_raw(
                ref_ptrs=[ref_ptr],
                ref_labels=[ref_labels],
                ref_features=[ref_features],
                num_threads=num_threads,
                **marker_args,
            )
            markers = _markers_to_dict(mrk, lablev, ref_features)
            labind = numpy.array(biocutils.match(ref_labels, lablev), dtype=numpy.uint32)
        else:
            raise NotImplementedError("other marker methods are not yet implemented, sorry")
    else:
        lablev, labind = _factorize(ref_labels)
        labind = numpy.array(labind, dtype=numpy.uint32)
        mrk = _markers_from_dict(markers, lablev, ref_features)

    return TrainedSingleReference(
        lib.train_single(
            ref_ptr.ptr,
            labind,
            mrk,
            approximate,
            num_threads,
        ),
        labels=lablev,
        features=ref_features,
        markers=markers,
    )
