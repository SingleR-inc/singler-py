from typing import Sequence, Optional, Union

import numpy 
import biocutils
import warnings
import mattress

from ._train_single import TrainedSingleReference
from . import _lib_singler as lib
from ._utils import _stable_union, _stable_intersect, _to_NamedList


class TrainedIntegratedReferences:
    """
    Integrated references, typically constructed by :py:meth:`~singler.train_integrated`.
    This is intended for advanced users only and should not be serialized.
    """

    def __init__(self, ptr: int, ref_labels: list, ref_names: Optional[biocutils.Names]):
        self._ptr = ptr
        self._labels = ref_labels
        self._names = ref_names

    @property
    def reference_labels(self) -> list:
        """
        List of lists containing the names of the labels for each reference.
        """
        return self._labels

    @property
    def reference_names(self) -> Optional[biocutils.Names]:
        """
        Names of the references, or ``None`` if they were unnamed.
        """
        return self._names


def train_integrated(
    test_features: Sequence,
    ref_prebuilt: Union[dict, Sequence, biocutils.NamedList],
    warn_lost: bool = True,
    num_threads: int = 1,
) -> TrainedIntegratedReferences:
    """
    Build a set of integrated references for classification of a test dataset.

    Arguments:
        test_features:
            Sequence of features for the test dataset.

        ref_prebuilt:
            List of prebuilt references, typically created by calling :py:meth:`~singler.train_single`.

        warn_lost:
            Whether to emit a warning if the markers for each reference are not all present in all references.

        num_threads:
            Number of threads.

    Returns:
        An integrated reference object, for classification with :py:meth:`~singler.classify_integrated`.

    Examples:
        >>> # Mocking up data.
        >>> import singler
        >>> ref = singler.mock_reference_data(num_replicates=8)
        >>> ref1 = ref[:,[True, False] * int(ref.shape[1]/2)]
        >>> ref2 = ref[:,[False, True] * int(ref.shape[1]/2)]
        >>> 
        >>> cd2 = ref2.get_column_data()
        >>> label2 = [l.lower() for l in cd2["label"]] # converting to lower case for some variety.
        >>> cd2.set_column("label", label2, in_place=True)
        >>> ref2.set_column_data(cd2, in_place=True)
        >>> 
        >>> import scranpy
        >>> ref1 = scranpy.normalize_rna_counts_se(ref1)
        >>> ref2 = scranpy.normalize_rna_counts_se(ref2)
        >>> 
        >>> # Building a classifier for each reference.
        >>> test = singler.mock_test_data(ref)
        >>> built1 = singler.train_single(ref1, ref1.get_column_data()["label"], ref1.get_row_names())
        >>> built2 = singler.train_single(ref2, ref2.get_column_data()["label"], ref2.get_row_names())
        >>> 
        >>> # Creating an integrated classifier across references.
        >>> in_built = singler.train_integrated(test.get_row_names(), {"first": built1, "second": built2})
        >>> in_built.reference_labels
        >>> in_built.reference_names
    """

    ref_prebuilt = _to_NamedList(ref_prebuilt)

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
        ref_names=ref_prebuilt.get_names()
    )
