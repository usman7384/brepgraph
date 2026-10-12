# Contributing

Thanks for helping. The package turns B-rep CAD into graphs; it deliberately does not contain models or
training code.

## Setup

```bash
git clone https://github.com/<your-username>/brepgraph.git   # your fork of https://github.com/usman7384/brepgraph
cd brepgraph
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install                                    # optional
```

## Before opening a pull request

```bash
ruff check src tests scripts
ruff format --check src tests scripts
mypy
pytest                  # fast suite; `pytest -m slow` adds the 500+ face part
```

CI runs the same checks on Linux, macOS and Windows for Python 3.10 to 3.13.

## Guidelines

- **Tests first for features.** Every geometric feature needs a test against a shape with a known answer
  (box, cylinder, sphere, pocketed box, ...) and must hold the invariants in `tests/test_invariants.py`
  (unit normals where the mask is 1, rigid-motion invariance, scale behaviour, determinism).
- **New features go through the registry** (`docs/extending.md`); the core should not need to change.
  If a feature comes from a paper, cite the paper and section in its docstring and say where the
  implementation had to choose something the paper leaves open.
- **OCP differences.** The code supports OCP 7.9 (Python 3.10) and 8.x (Python 3.11+). Version-specific
  imports live in `src/brepgraph/_compat.py`; run the tests on both lines when touching OCP calls.
- **Keep the base install light.** The core depends on NumPy and OCP only; ML frameworks and Plotly are
  imported lazily inside the functions that need them.
- **Schema changes.** Anything that changes the stored layout bumps `SCHEMA_VERSION` (minor for additive,
  major for breaking) and is described in `docs/schema.md`.
- Regenerate the feature reference with `python scripts/gen_feature_docs.py`; a test fails if it drifts.
- Add a line to `CHANGELOG.md`.

## Test data

Tests build their shapes in code. To run the invariants on real parts, put STEP files in `tests/data`
(not committed; check each file's license) and run `pytest -m data`.
