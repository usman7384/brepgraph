"""brepgraph: B-rep CAD files in, ML-ready face-adjacency graphs out."""

__version__ = "0.3.0.dev0"

from . import features as _features  # noqa: F401  (registers the built-in features)
from . import presets, viz
from .batch import LoadResult, load_many
from .errors import (
    BrepGraphError,
    FeatureError,
    FileReadError,
    InvalidSolidError,
    MissingExtraError,
    SchemaVersionError,
    UnknownFeatureError,
    UnsupportedShapeError,
)
from .graph import SCHEMA_VERSION, BrepGraph
from .pipeline import build_graph, load, load_all
from .registry import (
    EDGE_FEATURES,
    NODE_FEATURES,
    PAIR_FEATURES,
    FeatureSet,
    edge_feature,
    node_feature,
    pair_feature,
)

__all__ = [
    "EDGE_FEATURES",
    "NODE_FEATURES",
    "PAIR_FEATURES",
    "SCHEMA_VERSION",
    "BrepGraph",
    "BrepGraphError",
    "FeatureError",
    "FeatureSet",
    "FileReadError",
    "InvalidSolidError",
    "LoadResult",
    "MissingExtraError",
    "SchemaVersionError",
    "UnknownFeatureError",
    "UnsupportedShapeError",
    "__version__",
    "build_graph",
    "edge_feature",
    "load",
    "load_all",
    "load_many",
    "node_feature",
    "pair_feature",
    "presets",
    "viz",
]
