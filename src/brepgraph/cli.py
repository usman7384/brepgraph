"""``brepgraph convert`` and ``brepgraph info`` (argparse only)."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

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
    for title, feats in (("node", g.node_feats), ("edge", g.edge_feats), ("pair", g.pair_feats)):
        for name, arr in feats.items():
            print(f"  {title:4s} {name:15s} {str(tuple(arr.shape)):18s} {arr.dtype}")


def _view(args) -> None:
    from .pipeline import load
    from .viz import show, show_graph

    g = load(args.input, presets.get(args.preset), grid_size=args.grid_size)
    color_by = args.color_by
    if ":" in color_by:  # "name:index" picks one column of a 2-D feature
        name, idx = color_by.split(":", 1)
        color_by = (name, int(idx) if idx.isdigit() else idx)
    if args.kind == "graph":
        fig = show_graph(g, color_by, layout=args.layout)
    else:
        fig = show(g, color_by, edges=not args.no_edges, graph=args.graph)
    out = args.out or str(Path(args.input).with_suffix(".html"))
    fig.write_html(out)
    print(f"{args.input} -> {out} (coloured by {args.color_by})")


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

    v = sub.add_parser("view", help="write an interactive HTML view (needs the viz extra)")
    v.add_argument("input")
    v.add_argument("--color-by", default="surface_type", help="node feature, or name:index")
    v.add_argument("--out", default=None, help="default: next to the input")
    v.add_argument("--preset", default="uvnet", choices=presets.names())
    v.add_argument("--grid-size", type=int, default=None)
    v.add_argument("--kind", choices=["part", "graph"], default="part", help="3-D part or face graph")
    v.add_argument(
        "--layout", choices=["spring", "spring3d", "centroid"], default="spring", help="for --kind graph"
    )
    v.add_argument("--no-edges", action="store_true")
    v.add_argument("--graph", action="store_true", help="overlay the face-adjacency graph")
    v.set_defaults(func=_view)

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
