"""Load or convert many files in parallel worker processes.

OpenCASCADE keeps global state and is not safe to drive from several threads, so parallelism uses
processes. A native crash (for example a segmentation fault on a corrupt file) kills one worker and
breaks the whole pool; the runner then retries the files that were in flight, isolates a file that
crashes twice in its own process, and reports it as failed instead of losing the batch.
"""

from __future__ import annotations

import functools
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import FileReadError
from .graph import BrepGraph

CRASH_MESSAGE = "worker process crashed (native error inside OpenCASCADE?)"
CAD_SUFFIXES = (".step", ".stp", ".iges", ".igs", ".brep", ".brp")


@dataclass
class LoadResult:
    """Outcome for one input file of :func:`load_many`."""

    source: str
    graphs: list[BrepGraph] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _tracked(fn: Callable, marker_dir: str, index: int, item: Any):
    """Run ``fn(item)`` with a marker file present while it runs, so a crash can be attributed."""
    marker = Path(marker_dir) / str(index)
    marker.touch()
    try:
        return fn(item)
    finally:
        marker.unlink(missing_ok=True)


def run_parallel(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    error_result: Callable[[Any, str], Any],
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[Any]:
    """Call ``fn(item)`` for every item in worker processes and return the results in input order.

    ``fn`` must be a picklable top-level function (``functools.partial`` of one is fine) that
    handles its own errors. ``error_result(item, message)`` builds the result for an item whose
    worker crashed. ``workers=1`` runs everything in this process (no crash protection).
    """
    n = len(items)
    results: list[Any] = [None] * n
    done = 0

    def finish(i: int, result: Any) -> None:
        nonlocal done
        results[i] = result
        done += 1
        if progress is not None:
            progress(done, n)

    workers = workers or os.cpu_count() or 1
    if workers <= 1:
        for i, item in enumerate(items):
            finish(i, fn(item))
        return results

    context = multiprocessing.get_context("spawn")  # same behaviour on every OS; no forked OCCT state
    crashes = [0] * n
    marker_dir = tempfile.mkdtemp(prefix="brepgraph-batch-")
    try:
        todo = list(range(n))
        while todo:
            pooled = [i for i in todo if crashes[i] < 2]
            alone = [i for i in todo if crashes[i] >= 2]
            todo = []
            if pooled:
                crashed = False
                with ProcessPoolExecutor(min(workers, len(pooled)), mp_context=context) as pool:
                    futures = {pool.submit(_tracked, fn, marker_dir, i, items[i]): i for i in pooled}
                    for future in as_completed(futures):
                        i = futures[future]
                        try:
                            finish(i, future.result())
                        except BrokenProcessPool:
                            crashed = True
                            todo.append(i)
                if crashed:  # only the files that were running when the pool broke are suspects
                    for name in os.listdir(marker_dir):
                        crashes[int(name)] += 1
                        os.remove(os.path.join(marker_dir, name))
            for i in alone:
                with ProcessPoolExecutor(1, mp_context=context) as pool:
                    try:
                        finish(i, pool.submit(fn, items[i]).result())
                    except BrokenProcessPool:
                        finish(i, error_result(items[i], CRASH_MESSAGE))
    finally:
        shutil.rmtree(marker_dir, ignore_errors=True)
    return results


def _load_one(source, options: dict[str, Any]) -> LoadResult:
    from .pipeline import load, load_all

    opts = dict(options)
    all_solids = opts.pop("all_solids")
    try:
        graphs = load_all(source, **opts) if all_solids else [load(source, **opts)]
    except Exception as exc:  # a bad file must not stop the batch; report it instead
        return LoadResult(str(source), error=f"{type(exc).__name__}: {exc}")
    return LoadResult(str(source), graphs)


def load_many(
    sources: Iterable[str | os.PathLike[str]],
    features=None,
    *,
    workers: int | None = None,
    all_solids: bool = False,
    grid_size: int | None = None,
    heal: bool = True,
    unify_faces: bool = False,
    normalize: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> list[LoadResult]:
    """Load many files in parallel; one :class:`LoadResult` per source, in input order.

    A file that fails does not raise: its result has ``ok == False`` and an ``error`` message.
    ``workers`` defaults to the number of CPUs; ``workers=1`` loads in this process. With
    ``all_solids=True`` every solid of a file becomes a graph (see :func:`brepgraph.load_all`).

    Workers are fresh processes, so custom features must be registered in an importable module (or
    at the top level of the main script). Features defined in a notebook cell are unknown to the
    workers; use ``workers=1`` there.
    """
    options = dict(
        features=features,
        grid_size=grid_size,
        heal=heal,
        unify_faces=unify_faces,
        normalize=normalize,
        all_solids=all_solids,
    )
    items = [str(s) for s in sources]
    return run_parallel(
        functools.partial(_load_one, options=options),
        items,
        error_result=lambda source, message: LoadResult(str(source), error=message),
        workers=workers,
        progress=progress,
    )


def solid_output_path(output: str | Path, solid_index: int, num_solids: int) -> Path:
    """``part.npz`` for a single-solid file, ``part.solid<k>.npz`` for solid ``k`` of several."""
    output = Path(output)
    if num_solids == 1:
        return output
    return output.with_name(f"{output.stem}.solid{solid_index}{output.suffix}")


def convert_one(task: tuple[str, str, dict[str, Any], bool]) -> dict[str, Any]:
    """Worker for ``brepgraph batch``: load one file and save its graph(s); returns a report row."""
    source, output, options, include_source_path = task
    result = _load_one(source, options)
    row: dict[str, Any] = {"source": source, "ok": result.ok, "outputs": [], "error": result.error}
    for g in result.graphs:
        out = solid_output_path(output, g.meta.get("solid_index", 0), g.meta.get("num_solids", 1))
        out.parent.mkdir(parents=True, exist_ok=True)
        g.save(out, include_source_path=include_source_path)
        row["outputs"].append({"path": str(out), "faces": g.num_faces, "edges": g.num_edges})
    return row


def find_cad_files(inputs: Iterable[str | os.PathLike[str]]) -> list[tuple[Path, Path]]:
    """``(file, path relative to its input root)`` for every CAD file under the given files/folders."""
    found = []
    for raw in inputs:
        root = Path(raw)
        if not root.exists():
            raise FileReadError(f"{root}: no such file or folder")
        if root.is_dir():
            for f in sorted(root.rglob("*")):
                if f.is_file() and f.suffix.lower() in CAD_SUFFIXES:
                    found.append((f, f.relative_to(root)))
        else:
            found.append((root, Path(root.name)))
    return found
