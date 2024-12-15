# Changelog

## Version 0.4.0

- Switch to using **pybind11** for the Python/C++ interface.
- Update to the latest **singlepp** C++ library.
- Store marker lists as lists of `StringList` objects in the metadata of the `BiocFrame` objects.
- Automatically remove `None` labels in the reference. 
- Automatically remove duplicated feature names in the reference. 
- Remove the `ref_names=` option in `train_integrated()` for simplicity.
- Remove support for auto-loading of references based on their names; users should extract references from **celldex** instead.
- Deprecate options to extract features/labels from a `SummarizedExperiment` based on the column name; users should explicitly pass the column contents to each function.

## Version 0.3.0

Compatibility with NumPy 2.0

## Version 0.2.0

Migrate references to the `celldex` package.

## Version 0.1.1

- Support `SummarizedExperiment` inputs consistently in all functions.

## Version 0.1.0

- Added the `annotate_integrated()` function, which performs annotation across multiple references.
