from typing import Sequence, Optional, Union

import numpy 
import biocutils
import warnings
import mattress

from .train_single import TrainedSingleReference
from . import lib_singler as lib
from ._utils import _stable_union, _stable_intersect


class TrainedIntegratedReferences:
    """Object containing integrated references, typically constructed by
    :py:meth:`~singler.train_integrated.train_integrated`."""

    def __init__(self, ptr: int, ref_labels: list):
        self._ptr = ptr
        self._labels = ref_labels

    @property
    def reference_labels(self) -> list:
        """List of lists containing the names of the labels for each reference.

        Each entry corresponds to a reference in :py:attr:`~reference_names`,
        if ``reference_names`` is not ``None``.
        """
        return self._labels


def train_integrated(
    test_features: Sequence,
    ref_prebuilt: list[TrainedSingleReference],
    warn_lost: bool = True,
    num_threads: int = 1,
) -> TrainedIntegratedReferences:
    """Build a set of integrated references for classification of a test dataset.

    Arguments:
        test_features:
            Sequence of features for the test dataset.

        ref_prebuilt:
            List of prebuilt references, typically created by calling
            :py:meth:`~singler.train_single.train_single`.

        warn_lost:
            Whether to emit a warning if the markers for each reference are not
            all present in all references.

        num_threads:
            Number of threads.

    Returns:
        Integrated references for classification with
        :py:meth:`~singler.classify_integrated_references.classify_integrated`.
    """
    # Checking the genes.
    if warn_lost:
        all_refnames = [x.features for x in ref_prebuilt]
        intersected = set(_stable_intersect(*all_refnames))
        for trained in ref_prebuilt:
            for g in trained.marker_subset():
                if g not in intersected:
                    warnings.warn("not all markers in 'ref_prebuilt' are available in each reference")

    all_inter_test = []
    all_inter_ref = []
    for i, trained in enumerate(ref_prebuilt):
        common = _stable_intersect(test_features, trained.features)
        all_inter_test.append(biocutils.match(common, test_features, dtype=numpy.uint32))
        all_inter_ref.append(biocutils.match(common, trained.features, dtype=numpy.uint32))

    all_data = [mattress.initialize(x._full_data) for x in ref_prebuilt]

    # Applying the integration.
    ibuilt = lib.train_integrated(
        all_inter_test,
        [x.ptr for x in all_data],
        all_inter_ref,
        [x._full_label_codes for x in ref_prebuilt],
        [x._ptr for x in ref_prebuilt],
        num_threads
    )

    return TrainedIntegratedReferences(
        ptr=ibuilt,
        ref_labels=[x.labels for x in ref_prebuilt],
    )
