"""pytest configuration: sibling imports and defaults isolation.

The gates are **imported** here rather than spawned, which is faster (numba
compiles once for the session instead of once per gate) but takes back two
things a subprocess gave for free.

1. SIBLING IMPORTS. A gate is a script run by path, so its directory is
   ``sys.path[0]`` and a ``_common.py`` beside it just imports. Under pytest
   there is no such entry, so each gate directory goes on the path here, once.
   This is the one sanctioned ``sys.path`` insertion in the tree: gate support
   files are deliberately not packaged (see ``pyproject.toml``'s explicit
   ``packages`` list), so there is no installed name to import them by. Do not
   "fix" it by packaging ``tests/``.

2. DEFAULTS ISOLATION, the one that would otherwise bite silently. ``mhs``
   cannot test its byte ceiling without rebinding ``defaults.MATRIX_MAX_BYTES``
   down around a clause -- the alternative is allocating 20 GiB to watch it
   succeed. In a shared interpreter a clause that failed between the set and the
   restore would leak the small value into every later gate, and every later
   matrix call would refuse for a reason invisible in its own source. The autouse
   fixture snapshots every uppercase name, restores it in ``finally``, and
   deletes any a test invented, so the leak is impossible rather than unlikely.

Filename collisions are handled on the other side, in ``test_gates._import_gate``:
``verify_pde_residual`` is expected to exist in more than one suite, and under the
bare stem the second import would return the first suite's module and the test
would silently check the wrong file.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
GATE_DIRS = {
    "mhs": HERE / "gates" / "mhs",
    "parity": HERE / "gates" / "parity",
    "oracle": HERE / "gates" / "oracle",
}

# tests/ for `import run_all`, then each gate directory for the sibling imports.
for _p in [HERE, *GATE_DIRS.values()]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture(autouse=True)
def defaults_isolated():
    """Every test sees ``mhs.defaults`` as it shipped, and leaves it that way.

    Restores rather than merely checking, so one gate's deliberate rebind -- or
    an exception thrown between a set and its restore -- cannot change what a
    later gate measures. Yields the module so a test may rebind freely.
    """
    from mhs import defaults

    held = {k: v for k, v in vars(defaults).items() if k.isupper()}
    try:
        yield defaults
    finally:
        for k, v in held.items():
            setattr(defaults, k, v)
        for k in [k for k in vars(defaults) if k.isupper() and k not in held]:
            delattr(defaults, k)          # a default a test invented
