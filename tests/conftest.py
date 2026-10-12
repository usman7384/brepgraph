import pytest

import brepgraph as bg
import fixtures


@pytest.fixture
def step(tmp_path):
    """Write an OCP shape to a STEP file and return its path."""
    counter = iter(range(10_000))

    def _step(shape):
        return fixtures.write_step(shape, tmp_path / f"shape{next(counter)}.step")

    return _step


@pytest.fixture
def graph(step):
    """Build a graph from an OCP shape through the real STEP loader."""

    def _graph(shape, **kw):
        return bg.load(step(shape), **kw)

    return _graph
