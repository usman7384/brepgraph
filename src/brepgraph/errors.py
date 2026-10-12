"""Exception hierarchy. Catch :class:`BrepGraphError` to handle everything the library raises."""

from __future__ import annotations


class BrepGraphError(Exception):
    """Base class for all library errors."""


class FileReadError(BrepGraphError):
    """The file is missing, unreadable, not a recognised format, or the reader reported failure."""


class UnsupportedShapeError(BrepGraphError):
    """No solid, more than one solid, an assembly, or a non-manifold edge."""


class InvalidSolidError(BrepGraphError):
    """The solid fails the validity check and healing did not fix it."""


class FeatureError(BrepGraphError):
    """A feature function failed or returned something that violates its declaration."""

    def __init__(self, feature: str, index: int | None, message: str, kind: str = "face"):
        self.feature = feature
        self.index = index
        self.kind = kind
        where = f" ({kind} {index})" if index is not None else ""
        super().__init__(f"feature '{feature}'{where}: {message}")


class UnknownFeatureError(BrepGraphError, ValueError):
    """A feature name is not in the registry."""


class SchemaVersionError(BrepGraphError):
    """A saved graph uses an incompatible major schema version."""


class MissingExtraError(BrepGraphError, ImportError):
    """An optional dependency is missing; the message names the extra to install."""
