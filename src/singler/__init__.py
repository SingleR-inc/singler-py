import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from .get_classic_markers import get_classic_markers, number_of_classic_markers
from .train_single import train_single, TrainedSingleReference
from .classify_single import classify_single
from .annotate_single import annotate_single
from .train_integrated import train_integrated, TrainedIntegratedReferences
from .classify_integrated import classify_integrated
from .annotate_integrated import annotate_integrated
from .aggregate_reference import aggregate_reference
