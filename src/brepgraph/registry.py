"""Feature registry: decorators that add a function to a dict, plus :class:`FeatureSet`.

A feature function takes ``(ctx, index, **params)`` and returns a NumPy array of the declared
shape for ONE face (``@node_feature``) or ONE retained edge (``@edge_feature``). The pipeline
stacks the results into ``[F, ...]`` / ``[E, ...]`` arrays and checks shape, dtype and finiteness.

``@pair_feature`` functions are different: they take ``(ctx, **params)`` and return the whole
``[F, F, ...]`` array at once (pairwise quantities are far cheaper to compute in bulk). The declared
``shape`` is the trailing shape after the two face axes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import UnknownFeatureError

Shape = tuple[int, ...] | Callable[[Mapping[str, Any]], tuple[int, ...]]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str  # "node" | "edge"
    dtype: str
    shape: Shape
    version: str
    defaults: Mapping[str, Any]
    fn: Callable = field(repr=False, compare=False)
    doc: str = ""

    def expected_shape(self, params: Mapping[str, Any]) -> tuple[int, ...]:
        return tuple(self.shape(params)) if callable(self.shape) else tuple(self.shape)

    def shape_label(self) -> str:
        if callable(self.shape):
            return "(" + ", ".join(str(d) for d in self.shape(self.defaults)) + ") *"
        return "(" + ", ".join(str(d) for d in self.shape) + ")"


NODE_FEATURES: dict[str, FeatureSpec] = {}
EDGE_FEATURES: dict[str, FeatureSpec] = {}
PAIR_FEATURES: dict[str, FeatureSpec] = {}
_TABLES = {"node": NODE_FEATURES, "edge": EDGE_FEATURES, "pair": PAIR_FEATURES}


def _register(kind: str, table: dict, name, dtype, shape, version, defaults):
    def deco(fn):
        if name in table:
            raise ValueError(f"{kind} feature '{name}' is already registered")
        table[name] = FeatureSpec(
            name=name,
            kind=kind,
            dtype=dtype,
            shape=shape,
            version=str(version),
            defaults=dict(defaults or {}),
            fn=fn,
            doc=(fn.__doc__ or "").strip().split("\n\n")[0].replace("\n", " "),
        )
        return fn

    return deco


def node_feature(name: str, *, dtype: str, shape: Shape, version: str = "1", defaults=None):
    """Register a per-face feature."""
    return _register("node", NODE_FEATURES, name, dtype, shape, version, defaults)


def edge_feature(name: str, *, dtype: str, shape: Shape, version: str = "1", defaults=None):
    """Register a per-edge feature."""
    return _register("edge", EDGE_FEATURES, name, dtype, shape, version, defaults)


def pair_feature(name: str, *, dtype: str, shape: Shape, version: str = "1", defaults=None):
    """Register a face-pair feature: ``fn(ctx, **params) -> [F, F, *shape]``."""
    return _register("pair", PAIR_FEATURES, name, dtype, shape, version, defaults)


def get_spec(kind: str, name: str) -> FeatureSpec:
    table = _TABLES[kind]
    try:
        return table[name]
    except KeyError:
        raise UnknownFeatureError(
            f"unknown {kind} feature '{name}'; registered {kind} features: {sorted(table)}"
        ) from None


Entry = str | tuple[str, Mapping[str, Any]]


def _norm(entries: Iterable[Entry]) -> tuple[tuple[str, dict[str, Any]], ...]:
    out: list[tuple[str, dict[str, Any]]] = []
    for e in entries:
        if isinstance(e, str):
            out.append((e, {}))
        else:
            n, p = e
            out.append((n, dict(p)))
    return tuple(out)


@dataclass(frozen=True)
class FeatureSet:
    """Ordered feature names with optional per-feature parameters.

    ``FeatureSet(nodes=["area", ("uv_grid", {"size": 16})], edges=["length"])``
    """

    nodes: Sequence[Any] = ()
    edges: Sequence[Any] = ()
    pairs: Sequence[Any] = ()

    def __post_init__(self):
        object.__setattr__(self, "nodes", _norm(self.nodes))
        object.__setattr__(self, "edges", _norm(self.edges))
        object.__setattr__(self, "pairs", _norm(self.pairs))
        for kind, entries in (("node", self.nodes), ("edge", self.edges), ("pair", self.pairs)):
            names = [n for n, _ in entries]
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate {kind} feature in feature set: {names}")
            for n, _ in entries:
                get_spec(kind, n)  # validate eagerly

    def resolve(self, grid_size: int | None = None):
        """Return ``(nodes, edges, pairs)``, each ``[(spec, params), ...]``, defaults merged in.

        ``grid_size`` overrides the ``size`` parameter of every feature that has one, unless that
        feature set it explicitly.
        """
        resolved = []
        for kind, entries in (("node", self.nodes), ("edge", self.edges), ("pair", self.pairs)):
            rows = []
            for name, explicit in entries:
                spec = get_spec(kind, name)
                params = dict(spec.defaults)
                if grid_size is not None and "size" in params:
                    params["size"] = int(grid_size)
                unknown = set(explicit) - set(spec.defaults)
                if unknown:
                    raise UnknownFeatureError(
                        f"feature '{name}' has no parameter(s) {sorted(unknown)}; "
                        f"valid: {sorted(spec.defaults)}"
                    )
                params.update(explicit)
                rows.append((spec, params))
            resolved.append(rows)
        return resolved[0], resolved[1], resolved[2]


def describe() -> str:
    """Markdown reference of every registered feature (used to generate docs/features.md)."""
    lines = ["# Feature reference", "", "Generated from the registry; do not edit by hand.", ""]
    for title, table in (
        ("Face (node) features", NODE_FEATURES),
        ("Edge features", EDGE_FEATURES),
        ("Face-pair features (stored as [F, F, ...])", PAIR_FEATURES),
    ):
        lines += [
            f"## {title}",
            "",
            "| Name | Shape | dtype | Params | Version | Meaning |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for s in table.values():
            params = ", ".join(f"{k}={v}" for k, v in s.defaults.items()) or "-"
            lines.append(f"| `{s.name}` | {s.shape_label()} | {s.dtype} | {params} | {s.version} | {s.doc} |")
        lines.append("")
    lines.append("`*` shape shown for default parameters.")
    return "\n".join(lines) + "\n"
