"""Top-level functions for the batch tests (worker processes must be able to import them)."""

import os


def crash_on_marked(item):
    """Simulate a native segfault: kill the worker process for items containing 'CRASH'."""
    if "CRASH" in item:
        os._exit(1)
    return f"ok:{item}"


def square(x):
    return x * x


def thread_env(_item):
    import os

    return os.environ.get("OPENBLAS_NUM_THREADS"), os.environ.get("OMP_NUM_THREADS")
