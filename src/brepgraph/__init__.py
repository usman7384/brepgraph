"""brepgraph: B-rep CAD files in, ML-ready face-adjacency graphs out."""

__version__ = "0.1.0.dev0"

from . import features as _features  # noqa: F401  (registers the built-in features)
from . import presets
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
from .pipeline import build_graph, load
from .registry import (
    EDGE_FEATURES,
    NODE_FEATURES,
    FeatureSet,
    edge_feature,
    node_feature,
)

__all__ = [
    "EDGE_FEATURES",
    "NODE_FEATURES",
    "SCHEMA_VERSION",
    "BrepGraph",
    "BrepGraphError",
    "FeatureError",
    "FeatureSet",
    "FileReadError",
    "InvalidSolidError",
    "MissingExtraError",
    "SchemaVersionError",
    "UnknownFeatureError",
    "UnsupportedShapeError",
    "__version__",
    "build_graph",
    "edge_feature",
    "load",
    "node_feature",
    "presets",
]
