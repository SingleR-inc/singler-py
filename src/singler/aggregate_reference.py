from typing import Any, Sequence, Optional, Union

import delayedarray
import summarizedexperiment
import numpy
import biocframe

from ._utils import _clean_matrix


def aggregate_reference(
    ref_data: Any,
    ref_labels: Sequence,
    ref_features: Sequence,
    num_centers: Optional[int] = None,
    power: float = 0.5,
    num_top: int = 1000,
    rank: int = 20,
    assay_type: Union[int, str] = "logcounts",
    subset_row: Optional[Sequence] = None,
    check_missing: bool = True,
    num_threads: int = 1
) -> summarizedexperiment.SummarizedExperiment:
    """Aggregate reference samples for a given label by using vector quantization to average their count profiles.
    The idea is to reduce the size of single-cell reference datasets so as to reduce the computation time of :py:func:`~singler.train_single.train_single`.
    We perform k-means clustering for all cells in each label and aggregate all cells within each k-means cluster.
    (More specifically, the clustering is done on the principal components generated from the highly variable genes to better capture the structure within each label.)
    This yields one or more profiles per label, reducing the number of separate observations while preserving some level of intra-label heterogeneity.

    Args:
        ref_data:
            Floating-point matrix of reference expression values, usually containing log-expression values.
            Alternatively, a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object containing such a matrix.

        ref_labels:
            Array of length equal to the number of columns in ``ref_data``, containing the labels for each cell.

        ref_features:
            Sequence of identifiers for each feature, i.e., row in ``ref_data``.

        num_centers:
            Maximum number of aggregated profiles to produce for each label with :py:func:`~scranpy.cluster_kmeans.cluster_kmeans`.
            If ``None``, a suitable number of profiles is automatically chosen. 

        power:
            Number between 0 and 1 indicating how much aggregation should be performed.
            Specifically, we set the number of clusters to ``X**power`` where ``X`` is the number of cells assigned to that label.
            Ignored if ``num_centers`` is not ``None``.

        num_top:
            Number of highly variable genes to use for PCA prior to clustering, see :py:func:`~scranpy.choose_highly_variable_genes.choose_highly_variable_genes`.

        rank:
            Number of principal components to use during clustering, see :py:func:`~scranpy.run_pca.run_pca`.

        assay_type:
            Integer or string specifying the assay of ``ref_data`` containing the relevant expression matrix,
            if ``ref`` is a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object.

        subset_row:
            Array of row indices specifying the rows of ``ref_data`` to use for clustering.
            If ``None``, no additional filtering is performed.
            Note that even if ``subset_row`` is provided, aggregation is still performed on all genes.

        check_missing:
            Whether to check for and remove rows with missing (NaN) values from ``ref_data``.

        num_threads:
            Number of threads to use.

    Returns:
        A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` containing the aggregated values in its first assay.
        The label for each aggregated profile is stored in the column data.
    """
    ref_data, ref_features = _clean_matrix(
        ref_data,
        ref_features,
        assay_type=assay_type,
        check_missing=check_missing,
        num_threads=num_threads,
    )

    by_label = {}
    for i, lab in enumerate(ref_labels):
        if lab in by_label:
            by_label[lab].append(i)
        else:
            by_label[lab] = [i]

    output_vals = []
    output_labels = []
    output_names = []
    for lab, chosen in by_label.items():
        current = ref_data[:,chosen]

        cur_num_centers = num_centers
        if cur_num_centers is None:
            cur_num_centers = int(current.shape[1]**power)

        if cur_num_centers <= 1:
            output = numpy.reshape(current.mean(axis=1), (current.shape[0], 1))
        else:
            subcurrent = current
            if subset_row is not None:
                subcurrent = subcurrent[subset_row,:]

            # Doing a mini-analysis here: PCA on HVGs followed by k-means.
            import scranpy
            stats = scranpy.model_gene_variances(subcurrent, num_threads=num_threads)
            keep = scranpy.choose_highly_variable_genes(stats.residual, top=num_top)
            subcurrent = subcurrent[keep,:]

            if rank <= min(subcurrent.shape)-1:
                pcs = scranpy.run_pca(subcurrent, number=rank, num_threads=num_threads).components
            else:
                pcs = subcurrent

            clustered = scranpy.cluster_kmeans(pcs, k=cur_num_centers, num_threads=num_threads)
            agg = scranpy.aggregate_across_cells(current, [clustered.clusters], num_threads=num_threads)
            output = agg.sum / agg.counts

        output_vals.append(output)
        output_labels += [lab] * output.shape[1]
        for i in range(output.shape[1]):
            output_names.append(lab + "_" + str(i))

    if len(output_vals) == 0:
        output_vals.append(numpy.zeros((ref_data.shape[0], 0)))

    output = summarizedexperiment.SummarizedExperiment(
        { "logcounts": numpy.concatenate(output_vals, axis=1) },
        column_data = biocframe.BiocFrame({ "label": output_labels })
    )
    output.set_column_names(output_names, in_place=True)
    output.set_row_names(ref_features, in_place=True)
    return output
