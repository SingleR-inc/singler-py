import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    _dist_name = __name__
    __version__ = version(_dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from ._get_classic_markers import get_classic_markers, number_of_classic_markers
from ._train_single import train_single, TrainedSingleReference
from ._classify_single import classify_single
from ._annotate_single import annotate_single
from ._train_integrated import train_integrated, TrainedIntegratedReferences
from ._classify_integrated import classify_integrated
from ._annotate_integrated import annotate_integrated
from ._aggregate_reference import aggregate_reference
from ._mock_data import mock_reference_data, mock_test_data


__all__ = []
for _name in dir():
    if _name.startswith("_") or _name == "sys":
        continue
    __all__.append(_name)
