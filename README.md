<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/singler.svg?branch=main)](https://cirrus-ci.com/github/<USER>/singler)
[![ReadTheDocs](https://readthedocs.org/projects/singler/badge/?version=latest)](https://singler.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/singler/main.svg)](https://coveralls.io/r/<USER>/singler)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/singler.svg)](https://anaconda.org/conda-forge/singler)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/singler)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/singler.svg)](https://pypi.org/project/singler/)
[![Monthly Downloads](https://static.pepy.tech/badge/singler/month)](https://pepy.tech/project/singler)
![Unit tests](https://github.com/SingleR-inc/singler-py/actions/workflows/run-tests.yml/badge.svg)

# Tinder for single-cell data

## Overview

This package provides Python bindings to the [C++ implementation](https://github.com/SingleR-inc/singlepp) of the [**SingleR** method](https://github.com/SingleR-inc/SingleR),
originally developed by [Aran et al. (2019)](https://www.nature.com/articles/s41590-018-0276-y).
It is designed to annotate cell types by matching cells to known references based on their expression profiles.
So kind of like Tinder, but for cells.

## Quick start

Firstly, let's load in the famous PBMC 4k dataset from 10X Genomics.
Any [`SummarizedExperiment`](https://github.com/biocpy/SummarizedExperiment) can be used here.

```python
import singlecellexperiment
sce = singlecellexperiment.read_tenx_h5("pbmc4k-tenx.h5", realize_assays=True)
## class: SingleCellExperiment
## dimensions: (33694, 4340)
## assays(1): ['counts']
## row_data columns(2): ['id', 'name']
## row_names(0):
## column_data columns(0): []
## column_names(0):
## main_experiment_name:
## reduced_dims(0): []
## alternative_experiments(0): []
## row_pairs(0): []
## column_pairs(0): []
## metadata(0):
```

Now, we fetch the Blueprint/ENCODE reference from the [**celldex**](https://pypi.org/project/celldex) package:

```python
import celldex
ref_data = celldex.fetch_reference("blueprint_encode", "2024-02-26", realize_assays=True)
## class: SummarizedExperiment
## dimensions: (19859, 259)
## assays(1): ['logcounts']
## row_data columns(0): []
## row_names(19859): ['TSPAN6', 'TNMD', 'DPM1', ..., 'MIR522', 'LINC00550', 'GIMAP1-GIMAP5']
## column_data columns(3): ['label.main', 'label.fine', 'label.ont']
## column_names(259): ['mature.neutrophil', 'CD14.positive..CD16.negative.classical.monocyte', 'mature.neutrophil.1', ..., 'fibroblast.of.dermis.1', 'epithelial.cell.of.umbilical.artery.1', 'dermis.lymphatic.vessel.endothelial.cell.1']
## metadata(0): 
```

We annotate each cell in `sce` against the reference.
This yields a data frame that contains all of the assignments and the scores for each label:

```python
import singler
results = singler.annotate_single(
    test_data = sce,
    test_features = sce.get_row_data()["name"],
    ref_data = ref_data,
    ref_labels = ref_data.get_column_data().column("label.main"),
)
print(results)
## BiocFrame with 4340 rows and 3 columns
##             best                                   scores               delta
##           <list>                              <BiocFrame>  <ndarray[float64]>
##    [0] Monocytes 0.2562168476981947:0.1254343439610945...  0.4378177347327983
##    [1] Monocytes 0.2834593285584352:0.1350551446328624... 0.06708042619997218
##    [2] Monocytes 0.27001789110872965:0.149733483922888... 0.29630159290612557
##              ...                                      ...                 ...
## [4337]  NK cells 0.22504679944584366:0.128832705528845... 0.09253938940916262
## [4338]   B-cells 0.21466213533061748:0.143717963254983... 0.06727011631382662
## [4339] Monocytes 0.2880677943712168:0.1327331541412791... 0.06576621116161818
## ------
## metadata(2): used markers

print(results["scores"]["Macrophages"])
## [0.3553803  0.40346796 0.3680465  ... 0.32339334 0.29082273 0.39644526]
```

## Calling low-level functions

The `annotate_single()` function is a convenient wrapper around a number of lower-level functions in **singler**.
Advanced users may prefer to build the reference and run the classification separately.
This allows us to re-use the same reference for multiple datasets without repeating the build step.

```python
built = singler.train_single(
    ref_data = ref_data,
    ref_labels = ref_data.get_column_data().column("label.main"),
    ref_features = ref_data.get_row_names(),
    test_features = sce.get_row_data()["name"]
)
```

Then, we apply the pre-built reference to the test dataset to obtain our label assignments.
This can be repeated with different datasets that have the same features as `test_features=`.

```python
output = singler.classify_single(mat, ref_prebuilt=built)
print(output)
## BiocFrame with 4340 rows and 3 columns
##             best                                   scores               delta
##           <list>                              <BiocFrame>  <ndarray[float64]>
##    [0] Monocytes 0.2562168476981947:0.1254343439610945...  0.4378177347327983
##    [1] Monocytes 0.2834593285584352:0.1350551446328624... 0.06708042619997218
##    [2] Monocytes 0.27001789110872965:0.149733483922888... 0.29630159290612557
##              ...                                      ...                 ...
## [4337]  NK cells 0.22504679944584366:0.128832705528845... 0.09253938940916262
## [4338]   B-cells 0.21466213533061748:0.143717963254983... 0.06727011631382662
## [4339] Monocytes 0.2880677943712168:0.1327331541412791... 0.06576621116161818
## ------
## metadata(2): used markers
```

## Integrating labels across references

We can use annotations from multiple references through the `annotate_integrated()` function.
This annotates the test dataset against each reference individually to obtain the best per-reference label,
and then it compares across references to find the best label from all references.

```python
import singler
import celldex
blueprint_ref = celldex.fetch_reference("blueprint_encode", "2024-02-26", realize_assays=True)
immune_cell_ref = celldex.fetch_reference("dice", "2024-02-26", realize_assays=True)

integrated_res = singler.annotate_integrated(
    mat,
    ref_data = [
        blueprint_ref,
        immune_cell_ref
    ],
    ref_labels = [
        blueprint_ref.get_column_data().column("label.main"),
        immune_cell_ref.get_column_data().column("label.main")
    ],
    test_features = features
)

print(integrated_res["integrated"])
## BiocFrame with 4340 rows and 4 columns
##        best_label    best_reference                                   scores               delta
##            <list> <ndarray[uint32]>                              <BiocFrame>  <ndarray[float64]>
##    [0]  Monocytes                 0 Monocytes:0.4601040318745395:Monocyte... 0.07172619402931646
##    [1]  Monocytes                 0 Monocytes:0.5569436588644365:Monocyte... 0.10337145230321299
##    [2]  Monocytes                 0 Monocytes:0.460675384672641:Monocytes... 0.06302300967458618
##               ...               ...                                      ...                 ...
## [4337]   NK cells                 0 NK cells:0.5639386082584756:NK cells:... 0.02453897370863012
## [4338]    B-cells                 0 B-cells:0.49462921210156113:B cells:0...  0.0259893105975339
## [4339]  Monocytes                 0 Monocytes:0.49997247809330014:Monocyt... 0.08744894116986357
```

The ``best_label`` columns contains the best label for each cell across all references,
while the ``best_reference`` specifies the reference (in the same order as ``ref_data``) that contains the best label.
