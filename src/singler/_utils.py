from typing import Sequence, Tuple

import biocutils
import numpy
import delayedarray
import summarizedexperiment
import mattress


def _factorize(x: Sequence) -> Tuple[list, numpy.ndarray]:
    f = biocutils.Factor.from_sequence(x, sort_levels=False)
    return f.levels, numpy.array(f.codes, dtype=numpy.uint32)


def _create_map(x: Sequence) -> dict:
    mapping = {}
    for i, val in enumerate(x):
        if val is not None:
            # Again, favor the first occurrence.
            if val not in mapping:
                mapping[val] = i
    return mapping


def _stable_intersect(*args) -> list:
    nargs = len(args)
    if nargs == 0:
        return []

    occurrences = {}
    for f in args[0]:
        if f is not None and f not in occurrences:
            occurrences[f] = [1, 0]

    for i in range(1, len(args)):
        for f in args[i]:
            if f is not None and f in occurrences:
                state = occurrences[f]
                if state[1] < i:
                    state[0] += 1
                    state[1] = i

    output = []
    for f in args[0]:
        if f is not None and f in occurrences:
            state = occurrences[f]
            if state[0] == nargs and state[1] >= 0:
                output.append(f)
                state[1] = -1  # avoid duplicates

    return output


def _stable_union(*args) -> list:
    if len(args) == 0:
        return []

    output = []
    present = set()
    for a in args:
        for f in a:
            if f is not None and f not in present:
                output.append(f)
                present.add(f)

    return output


def _clean_matrix(x, features, assay_type, check_missing, num_threads):
    if isinstance(x, summarizedexperiment.SummarizedExperiment):
        if features is None:
            features = x.get_row_names()
        elif isinstance(features, str):
            features = x.get_row_data().column(features)
        features = list(features)
        x = x.assay(assay_type)

    curshape = x.shape
    if len(curshape) != 2:
        raise ValueError("each entry of 'ref' should be a 2-dimensional array")

    if curshape[0] != len(features):
        raise ValueError(
            "number of rows of 'x' should be equal to the length of 'features'"
        )

    if not check_missing:
        return x, features

    ptr = mattress.initialize(x)
    retain = ptr.row_nan_counts(num_threads=num_threads) == 0
    if retain.all():
        return x, features

    new_features = []
    for i, k in enumerate(retain):
        if k:
            new_features.append(features[i])

    sub = delayedarray.DelayedArray(x)[retain, :]
    return sub, new_features


def _restrict_features(data, features, restrict_to):
    if restrict_to is not None:
        if not isinstance(restrict_to, set):
            restrict_to = set(restrict_to)
        keep = []
        new_features = []
        for i, x in enumerate(features):
            if x in restrict_to:
                keep.append(i)
                new_features.append(x)
        return delayedarray.DelayedArray(data)[keep, :], new_features
    return data, features
