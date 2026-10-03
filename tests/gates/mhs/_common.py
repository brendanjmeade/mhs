"""Shared fixtures and the report harness for the ported derivation gates.

Named with a leading underscore so ``run_all.discover``'s ``verify_*`` glob skips
it, and reachable as a sibling import because ``tests/conftest.py`` puts each
gate directory on ``sys.path`` (a gate is a script run by path, so under
``run_all`` its own directory is already ``sys.path[0]``).

**Why these gates keep their upstream shape.** The ported gates use the ``Report``
harness below rather than the inline ``check``/``check_true`` the hand-written
mhs gates use, and that inconsistency is deliberate. Their value is entirely in
their tolerances -- 1e-13 on a two-route identity, 1e-7 on an on-plane
quadrature comparison -- and translating a tolerance from one harness into
another is exactly where one gets quietly loosened. Port faithfully, then leave
it alone.

**What these gates are for.** The parity gate proves the shipped full-space
engine and the frozen oracle agree, which today is only that a copy is a copy.
These check the MATHEMATICS, against Gauss quadrature computed in the gate
itself -- a reference that shares no code with the thing it checks. That is what
turns "the copies agree" into "and they are correct".
"""
from __future__ import annotations

import pathlib

import numpy as np

#: A generic tilted, non-right, non-unit triangle: no axis is special, no edge is
#: horizontal or vertical, and no vertex sits above another. A right or
#: axis-aligned triangle hides a transposed frame.
TRI = np.array([[0.37, -0.81, 0.44],
                [1.92, 0.11, -0.63],
                [-0.25, 1.57, 1.22]])

#: nu = 0.30, deliberately not 1/4, where lam == mu and a lam/mu swap is
#: invisible -- which is how a swapped kernel shipped for months upstream.
MU, NU_DEFAULT = 1.0, 0.3


def shipped(module: str):
    """Import a module from the SHIPPED engine, asserting it is the shipped one.

    These gates must check ``mhs.fullspace`` and not ``mhs_oracle.clq``. The two
    are byte-identical today, so a gate pointed at the wrong one would pass
    identically and nobody would learn anything -- and after the numba rewrite it
    is the shipped copy that changes, so it is the shipped copy that needs the
    mathematics pinned. Asserting the resolved parent directory makes the
    mistake loud rather than invisible.
    """
    import importlib
    mod = importlib.import_module(f"mhs.fullspace.{module}")
    parent = pathlib.Path(mod.__file__).parent.name
    assert parent == "fullspace", (
        f"mhs.fullspace.{module} resolved to .../{parent}/, wanted "
        f".../fullspace/ -- this gate must check the SHIPPED engine, not the "
        f"frozen oracle")
    return mod


def relmax(a, b) -> float:
    """max|a - b| / max|b|, falling back to absolute when b is ~0."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ref = np.max(np.abs(b))
    diff = np.max(np.abs(a - b))
    return diff / ref if ref > 1e-300 else diff


def random_rotation(seed: int = 0) -> np.ndarray:
    """A proper rotation, for covariance checks."""
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


class Report:
    """Collect named checks and print the one column-0 verdict at the end."""

    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, bool]] = []
        print("=" * 76)
        print(title)
        print("=" * 76)

    def check(self, name: str, value: float, tol: float, extra: str = "") -> bool:
        # NaN compares False against everything, so a NaN fails rather than
        # sliding through -- which is the behaviour we want from a tolerance.
        ok = bool(value < tol)
        self.rows.append((name, ok))
        print(f"  [{'ok' if ok else 'XX'}] {name:52s} {value:10.3e}  "
              f"(tol {tol:.0e}) {extra}")
        return ok

    def check_bool(self, name: str, ok: bool, extra: str = "") -> bool:
        ok = bool(ok)
        self.rows.append((name, ok))
        print(f"  [{'ok' if ok else 'XX'}] {name:52s} {extra}")
        return ok

    def finish(self) -> bool:
        ok = all(r[1] for r in self.rows)
        n_fail = sum(1 for r in self.rows if not r[1])
        print("-" * 76)
        if ok:
            print(f"PASS: {self.title} ({len(self.rows)} checks)")
        else:
            print(f"FAIL: {self.title} "
                  f"({n_fail} of {len(self.rows)} checks failed)")
        return ok
