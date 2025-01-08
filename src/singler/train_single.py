from typing import Any, Literal, Optional, Sequence, Union

import biocutils
import numpy
import knncolle
import mattress
import delayedarray
import warnings

from . import lib_singler as lib
from ._utils import _clean_matrix, _restrict_features, _stable_intersect
from .get_classic_markers import get_classic_markers
from .aggregate_reference import aggregate_reference


class TrainedSingleReference:
    """A prebuilt reference object, typically created by
    :py:meth:`~singler.train_single.train_single`. This is intended for
    advanced users only and should not be serialized.
    """

    def __init__(
        self,
        ptr: int,
        full_data: Any,
        full_label_codes: numpy.ndarray,
        labels: Sequence,
        features: Sequence,
        markers: dict[Any, dict[Any, Sequence]],
    ):
        self._ptr = ptr
        self._full_data = full_data
        self._full_label_codes = full_label_codes
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
            return biocutils.StringList(self._features[i] for i in buffer)


def _markers_from_dict(markers: dict[Any, dict[Any, Sequence]], labels: Sequence, available_features: Sequence):
    fmapping = {}
    for i, x in enumerate(available_features):
        fmapping[x] = i
    return outer_instance


def train_single(
    ref_data: Any,
    ref_labels: Sequence,
    ref_features: Sequence,
    test_features: Optional[Sequence] = None,
    assay_type: Union[str, int] = "logcounts",
    restrict_to: Optional[Union[set, dict]] = None,
    check_missing: bool = True,
    markers: Optional[dict[Any, dict[Any, Sequence]]] = None,
    marker_method: Literal["classic", "auc", "cohens_d"] = "classic",
    num_de: Optional[int] = None,
    marker_args: dict = {},
    aggregate: bool = False,
    aggregate_args: dict = {},
    nn_parameters: Optional[knncolle.Parameters] = knncolle.VptreeParameters(),
    num_threads: int = 1,
) -> TrainedSingleReference:
    """Build a single reference dataset in preparation for classification.

    Args:
        ref_data:
            A matrix-like object where rows are features, columns are
            reference profiles, and each entry is the expression value.
            If ``markers`` is not provided, expression should be normalized
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
            if ``ref_data`` is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        check_missing:
            Whether to check for and remove rows with missing (NaN) values
            from ``ref_data``.

        restrict_to:
            Subset of available features to restrict to. Only features in
            ``restrict_to`` will be used in the reference building. If
            ``None``, no restriction is performed.

        markers:
            Upregulated markers for each pairwise comparison between labels.
            Specifically, ``markers[a][b]`` should be a sequence of features
            that are upregulated in ``a`` compared to ``b``. All such features
            should be present in ``features``, and all labels in ``labels``
            should have keys in the inner and outer dictionaries.

        marker_method:
            Method to identify markers from each pairwise comparisons between labels in ``ref_data``.
            If ``classic``, we call :py:func:`~singler.get_classic_markers.get_classic_markers`.
            If ``auc`` or ``cohens_d``, we call :py:func:`~scranpy.score_markers.score_markers`.
            Only used if ``markers`` is not supplied.

        num_de:
            Number of differentially expressed genes to use as markers for each pairwise comparison between labels.
            If ``None`` and ``marker_method = "classic"``, an appropriate number of genes is determined by :py:func:`~singler.get_classic_markers.get_classic_markers`.
            Otherwise, it is set to 10.
            Only used if ``markers`` is not supplied.

        marker_args:
            Further arguments to pass to the chosen marker detection method.
            If ``marker_method = "classic"``, this is :py:func:`~singler.get_classic_markers.get_classic_markers`, otherwise it is :py:func:`~scranpy.score_markers.score_markers`.
            Only used if ``markers`` is not supplied.

        aggregate:
            Whether the reference dataset should be aggregated to pseudo-bulk samples for speed, see :py:func:`~singler.aggregate_reference.aggregate_reference` for details.

        aggregate_args:
            Further arguments to pass to :py:func:`~singler.aggregate_reference.aggregate_reference` when ``aggregate = True``.

        nn_parameters:
            Algorithm for constructing the neighbor search index, used to
            compute scores during classification.

        num_threads:
            Number of threads to use for reference building.

    Returns:
        The pre-built reference, ready for use in downstream methods like
        :py:meth:`~singler.classify_single_reference.classify_single`.
    """
    if isinstance(ref_labels, str):
        warnings.warn(
            "setting 'ref_labels' to a column name of the column data is deprecated",
            category=DeprecationWarning
        )
        ref_labels = ref_data.get_column_data().column(ref_labels)

    ref_data, ref_features = _clean_matrix(
        ref_data,
        ref_features,
        assay_type=assay_type,
        check_missing=check_missing,
        num_threads=num_threads,
    )

    if len(set(ref_features)) != len(ref_features):
        encountered = set()
        keep = []
        for i, rg in enumerate(ref_features):
            if rg not in encountered:
                encountered.add(rg)
                keep.append(i)
        if len(keep) < len(ref_features):
            ref_features = biocutils.subset_sequence(ref_features, keep)
            ref_data = delayedarray.DelayedArray(ref_data)[keep,:]

    if None in ref_labels:
        keep = []
        for i, f in enumerate(ref_labels):
            if not f is None:
                keep.append(i)
        ref_data = delayedarray.DelayedArray(ref_data)[:,keep]
        ref_labels = biocutils.subset_sequence(ref_labels, keep)
    unique_labels, label_idx = biocutils.factorize(ref_labels, sort_levels=True, dtype=numpy.uint32, fail_missing=True)

    markers = _identify_genes(
        ref_data=ref_data, 
        ref_features=ref_features,
        ref_labels=ref_labels,
        unique_labels=unique_labels,
        markers=markers,
        marker_method=marker_method,
        num_de=num_de,
        test_features=test_features,
        restrict_to=restrict_to,
        marker_args=marker_args,
        num_threads=num_threads
    )
    markers_idx = [None] * len(unique_labels)
    for outer_i, outer_k in enumerate(unique_labels):
        inner_instance = [None] * len(unique_labels)
        for inner_i, inner_k in enumerate(unique_labels):
            current = markers[outer_k][inner_k]
            inner_instance[inner_i] = biocutils.match(current, ref_features, dtype=numpy.uint32)
        markers_idx[outer_i] = inner_instance

    if test_features is None:
        test_features_idx = numpy.array(range(len(ref_features)), dtype=numpy.uint32)
        ref_features_idx = numpy.array(range(len(ref_features)), dtype=numpy.uint32)
    else:
        common_features = _stable_intersect(test_features, ref_features)
        test_features_idx = biocutils.match(common_features, test_features, dtype=numpy.uint32)
        ref_features_idx = biocutils.match(common_features, ref_features, dtype=numpy.uint32)

    if aggregate:
        aggr = aggregate_reference(ref_data, ref_labels, ref_features, **aggregate_args)
        ref_data = aggr.assay(0)
        label_idx = biocutils.match(aggr.get_column_data().get_column("label"), unique_labels, dtype=numpy.uint32)

    ref_ptr = mattress.initialize(ref_data)
    builder, _ = knncolle.define_builder(nn_parameters)
    return TrainedSingleReference(
        lib.train_single(
            test_features_idx,
            ref_ptr.ptr,
            ref_features_idx,
            label_idx,
            markers_idx,
            builder.ptr,
            num_threads,
        ),
        full_data = ref_data,
        full_label_codes = label_idx,
        labels = unique_labels,
        features = ref_features,
        markers = markers,
    )


def _identify_genes(ref_data, ref_features, ref_labels, unique_labels, markers, marker_method, test_features, restrict_to, num_de, marker_args, num_threads):
    ref_data, ref_features = _restrict_features(ref_data, ref_features, test_features)
    ref_data, ref_features = _restrict_features(ref_data, ref_features, restrict_to)

    # Deriving the markers from expression data.
    if markers is None:
        if marker_method == "classic":
            markers = get_classic_markers(
                ref_data=[ref_data],
                ref_labels=[ref_labels],
                ref_features=[ref_features],
                num_threads=num_threads,
                num_de=num_de,
                **marker_args,
            )
        else:
            if marker_method == "auc":
                compute_auc = True
                compute_cohens_d = False 
                effect_size = "auc"
                boundary = 0.5
            else:
                compute_auc = False 
                compute_cohens_d = True
                effect_size = "cohens_d"
                boundary = 0

            import scranpy
            stats = scranpy.score_markers(
                ref_data,
                groups=ref_labels,
                num_threads=num_threads,
                all_pairwise=True,
                compute_delta_detected=False,
                compute_delta_mean=False,
                compute_auc=compute_auc,
                compute_cohens_d=compute_cohens_d,
                **marker_args
            )
            pairwise = getattr(stats, effect_size)

            if num_de is None:
                num_de = 10

            markers = {}
            for g1, group1 in enumerate(stats.groups):
                group_markers = {}
                for g2, group2 in enumerate(stats.groups):
                    if g1 == g2:
                        group_markers[group2] = biocutils.StringList()
                        continue
                    cureffects = pairwise[g2, g1, :] # remember, second dimension is the first group in the comparison.
                    keep = biocutils.which(cureffects > boundary)
                    o = numpy.argsort(-cureffects[keep], stable=True)
                    group_markers[group2] = biocutils.StringList(biocutils.subset_sequence(ref_features, keep[o[:min(len(o), num_de)]]))
                markers[group1] = group_markers 

            return markers

    # Validating a user-supplied list of markers.
    if not isinstance(markers, dict):
        raise ValueError("'markers' should be a list where the labels are keys")
    if len(unique_labels) != len(markers):
        raise ValueError("'markers' should have length equal to the number of unique labels")

    available_features = set(ref_features)
    new_markers = {}
    for x, y in markers.items():
        if x not in unique_labels:
            raise ValueError("unknown label '" + x + "' in 'markers'")

        if isinstance(y, list):
            collected = []
            for g in y:
                if g in available_features:
                    collected.append(g)
            output = {} 
            for l in unique_labels:
                output[l] = biocutils.StringList()
            output[x] = collected
        elif isinstance(y, dict):
            if len(unique_labels) != len(y):
                raise ValueError("each value of 'markers' should have length equal to the number of unique labels")
            output = {} 
            for x_inner, y_inner in y.items():
                collected = biocutils.StringList()
                for g in y_inner:
                    if g in available_features:
                        collected.append(g)
                output[x_inner] = collected
        else:
            raise ValueError("values of 'markers' should be dictionaries")

        new_markers[x] = output

    return new_markers
