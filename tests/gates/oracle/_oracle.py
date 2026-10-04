"""Shared harness and Mindlin stress machinery for the oracle-suite gates.

**Why this is not called ``_common.py``.** ``tests/gates/mhs/_common.py``
already exists, and the two runners resolve sibling imports differently: under
``run_all`` a gate is a script, so ``sys.path[0]`` is its OWN directory and
``from _common import ...`` finds the right file; under pytest, ``conftest``
puts EVERY gate directory on the path, so the same statement in an oracle gate
could resolve to the mhs copy by path order. Both files define ``Report`` and
``relmax``, so it would not even fail -- it would quietly read the wrong file.
A distinct name makes that impossible under either runner. Per-directory
support files are the architecture here (a gate can only import its own
siblings), so the small duplication of the report harness is the price of that,
paid deliberately.

The Mindlin machinery below is shared by the free-surface and vertical-fault
gates, and is written once here for the usual reason: a convention stated twice
is one that can disagree with itself.
"""
from __future__ import annotations

import numpy as np

MU = 1.0

#: One-sided FD step, as a fraction of the geometric scale. The stress is a
#: FIRST derivative of G computed to second order, so the truncation error is
#: O(h^2) ~ 1e-10 at h = 1e-5 -- two orders below the smallest residual being
#: measured, and far below the O(eps^2) signal.
FD_STEP = 1.0e-5


def relmax(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ref = np.max(np.abs(b))
    diff = np.max(np.abs(a - b))
    return diff / ref if ref > 1e-300 else diff


def lam_of(mu: float, nu: float) -> float:
    """Only for talking to the vendored engine, which takes ``(mu, nu)``."""
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


def dG_obs(mindlin_G, obs, src, mu, nu, eps, h=FD_STEP):
    """``dG[i,j,m] = dG_ij / dx_m`` of the Mindlin kernel at ``obs``.

    Central differences in x and y, and a ONE-SIDED second-order backward
    stencil in z:

        f'(0) ~ [3 f(0) - 4 f(-h) + f(-2h)] / (2h)

    because ``mindlin_G`` refuses ``obs[2] > 0`` and a central difference at the
    free surface would step out of the body. That refusal is correct -- the
    kernel is a half-space kernel -- so the stencil accommodates it rather than
    the evaluation being moved off the surface, which would measure something
    else: the free-surface condition is a statement AT z = 0.
    """
    dG = np.zeros((3, 3, 3))
    for m in (0, 1):
        e = np.zeros(3)
        e[m] = h
        dG[:, :, m] = (mindlin_G(obs + e, src, mu, nu, eps)
                       - mindlin_G(obs - e, src, mu, nu, eps)) / (2.0 * h)
    z = np.array([0.0, 0.0, h])
    dG[:, :, 2] = (3.0 * mindlin_G(obs, src, mu, nu, eps)
                   - 4.0 * mindlin_G(obs - z, src, mu, nu, eps)
                   + mindlin_G(obs - 2.0 * z, src, mu, nu, eps)) / (2.0 * h)
    return dG


def sigma_of(mindlin_G, obs, src, mu, nu, eps, h=FD_STEP):
    """Stress ``(3, 3, 3)`` at ``obs``: ``sigma[i, j, k]`` per unit force ``k``.

    Hooke applied to the symmetric gradient of the Green's function, which is
    the definition rather than a second closed form -- so a disagreement here is
    a statement about the kernel, not about two implementations of a stress.
    """
    lam = lam_of(mu, nu)
    dG = dG_obs(mindlin_G, obs, src, mu, nu, eps, h)
    out = np.zeros((3, 3, 3))
    eye = np.eye(3)
    for k in range(3):
        g = dG[:, k, :]                         # g[i, m] = dG_ik / dx_m
        e = 0.5 * (g + g.T)
        out[:, :, k] = lam * np.trace(e) * eye + 2.0 * mu * e
    return out


def order_in_eps(values, epss) -> list[float]:
    """Observed convergence order between consecutive rungs."""
    return [float(np.log(values[i] / values[i + 1])
                  / np.log(epss[i] / epss[i + 1]))
            for i in range(len(values) - 1)]


class Report:
    """Collect named checks and print the one column-0 verdict at the end."""

    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, bool]] = []
        print("=" * 76)
        print(title)
        print("=" * 76)

    def check(self, name: str, value: float, tol: float, extra: str = "") -> bool:
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
