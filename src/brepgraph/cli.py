"""``brepgraph convert`` and ``brepgraph info`` (argparse only)."""

from __future__ import annotations

import argparse
import sys
import traceback

from . import __version__, presets
from .errors import BrepGraphError


def _convert(args) -> None:
    from .graph import BrepGraph  # noqa: F401
    from .pipeline import load

    g = load(
        args.input,
        presets.get(args.preset),
        grid_size=args.grid_size,
        heal=not args.no_heal,
        unify_faces=args.unify_faces,
        normalize=args.normalize,
    )
    g.save(args.output)
    print(
        f"{args.input} -> {args.output}: {g.num_faces} faces, {g.num_edges} edges, "
        f"{len(g.node_feats)} node + {len(g.edge_feats)} edge features"
    )


def _info(args) -> None:
    from .graph import BrepGraph

    g = BrepGraph.load(args.input)
    print(f"{args.input}: schema {g.schema_version}, {g.num_faces} faces, {g.num_edges} edges")
    for title, feats in (("node", g.node_feats), ("edge", g.edge_feats)):
        for name, arr in feats.items():
            print(f"  {title:4s} {name:15s} {str(tuple(arr.shape)):18s} {arr.dtype}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brepgraph", description="B-rep CAD -> ML graph")
    parser.add_argument("--version", action="version", version=f"brepgraph {__version__}")
    parser.add_argument("--debug", action="store_true", help="show tracebacks")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="convert a STEP/IGES/BREP file to a .npz graph")
    c.add_argument("input")
    c.add_argument("output")
    c.add_argument("--preset", default="uvnet", choices=presets.names())
    c.add_argument("--grid-size", type=int, default=None)
    c.add_argument("--normalize", action="store_true", help="centre and scale longest side to 1")
    c.add_argument("--unify-faces", action="store_true", help="merge same-domain faces")
    c.add_argument("--no-heal", action="store_true", help="do not attempt ShapeFix on invalid input")
    c.set_defaults(func=_convert)

    i = sub.add_parser("info", help="print the contents of a .npz graph")
    i.add_argument("input")
    i.set_defaults(func=_info)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except BrepGraphError as exc:
        if args.debug:
            traceback.print_exc()
        print(f"brepgraph: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
