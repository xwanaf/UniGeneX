__version__ = "0.1.0"
import importlib
import logging
import sys

logger = logging.getLogger("UniGeneX")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from .mapping import (
    AtlasMapper,
    align_atlas_to_mapping,
    load_mapping_result,
    project_query_to_atlas,
    validate_mapping_result,
)
from .plotting import plot_atlas_projection


def __getattr__(name):
    """Load the training subpackages only when they are requested."""
    if name in {"model", "tokenizer", "utils"}:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
