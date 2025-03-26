import singler
import celldex
import scrnaseq
from biocframe import BiocFrame

def value_counts(data):
    counts = {}
    for item in data:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts

def test_with_minimal_args():
    sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)
    immgen_ref = celldex.fetch_reference("immgen", "2024-02-26", realize_assays=True)

    matches = singler.annotate_single(
        test_data=sce,
        ref_data=immgen_ref,
        ref_labels=immgen_ref.get_column_data().column("label.main"),
    )
    assert isinstance(matches, BiocFrame)

    counts = value_counts(matches["best"])
    assert counts is not None


def test_with_all_supplied():
    sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)
    immgen_ref = celldex.fetch_reference("immgen", "2024-02-26", realize_assays=True)

    matches = singler.annotate_single(
        test_data=sce.assays["counts"],
        test_features=sce.get_row_names(),
        ref_data=immgen_ref,
        ref_labels=immgen_ref.get_column_data().column("label.main"),
        ref_features=immgen_ref.get_row_names(),
    )
    assert isinstance(matches, BiocFrame)

    counts = value_counts(matches["best"])
    assert counts is not None


def test_with_colname():
    sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)
    immgen_ref = celldex.fetch_reference("immgen", "2024-02-26", realize_assays=True)

    matches = singler.annotate_single(
        test_data=sce,
        ref_data=immgen_ref,
        ref_labels="label.main",
    )
    assert isinstance(matches, BiocFrame)

    counts = value_counts(matches["best"])
    assert counts is not None
