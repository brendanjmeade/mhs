"""Crack growth by work optimization, on a fixed lattice.

GROWTH AS ACTIVATION, WHICH IS WHY THIS IS CHEAP. The matrix is assembled ONCE
for a candidate lattice and growth turns cells on through ``static.solve``'s
``include`` mask. No remeshing, no re-assembly, and the front is implicit in
which cells are active -- the same machinery the fault-network growth verb
uses. It sets aside the 3-D front representation the XFEM literature calls the
hard part until it is actually needed, which is non-planar growth.

THE CRITERION IS ENERGETIC, NOT A STRESS INTENSITY FACTOR, and that is forced
rather than chosen: mollification removes the r^-1/2 singularity that DEFINES
K. That puts this in Cooke's work-optimization family (GROW, which is 2-D
plane strain) rather than the LEFM one, with ``eps`` playing the role a
phase-field regularization length plays.

It is also the better-conditioned choice here, and phase 0 says why. The
error against an analytic crack is 0.20 % in the interior but 5-7 % within
three ``eps`` of the front -- exactly where a front-local criterion would be
evaluated. The released energy is a global integral dominated by the accurate
interior, and phase 1 showed its 1.3 % bias is a pure multiplicative scale
factor shared with dE/dA, so it CANCELS in the comparison between candidates
that work optimization actually makes.

A SHEAR CRACK DOES NOT TEND TO A CIRCLE, and assuming it did cost a run. The
circle is the attractor for a TENSILE crack. On a shear crack the front is not
isotropic: where the slip is perpendicular to the front it is mode II, where
parallel, mode III, and

    G_II = (1-nu) K_II^2 / 2mu      G_III = K_III^2 / 2mu

Working the penny-crack intensity factors through gives
``G(mode III tips) / G(mode II tips) = (1 - nu)``, which is 0.75 at nu = 0.25.
So a shear crack releases MORE energy growing along the slip direction and
elongates; measured here, aspect 1.29 -> 1.44 over fourteen steps. A first
version of this experiment treated that elongation as a failure, having
imported a tensile-crack intuition into a shear problem.

The decisive test of that reading is ``nu``: the anisotropy is carried by the
(1 - nu) factor, so it should VANISH at nu = 0 and strengthen toward nu = 0.5.
That is a parameter-free prediction with a known sign and it has not yet been
run.

MESH DEPENDENCE IS THE RISK THIS MODULE EXISTS TO TEST, because sharp-crack
growth is notoriously sensitive to the discretization -- a main reason
phase-field methods exist. Growing the same crack on a lattice rotated by 30
degrees, the maximal inequivalence for a 6-fold lattice, the released energy
agrees to 0.71 % at worst and 0.10 % at the final step over fourteen steps.
The crude max|x|/max|y| aspect differs by 0.095, but ONE LATTICE CELL moves
that measure by 0.100 and the two runs began 0.11 apart from discretizing the
same initial ellipse, so the difference is one cell and inherited rather than
accumulated. A second-moment shape measure would say this more sharply and is
what ``shape_ratio`` below exists for.
"""
from __future__ import annotations

import numpy as np

import static as St

#: Candidate neighbourhood, in units of the lattice pitch. 1.1 picks up the
#: cells sharing a vertex with the active set and nothing further.
NEIGHBOUR_REACH = 1.1


def shape_ratio(centers: np.ndarray, on: np.ndarray) -> float:
    """``sqrt(<x^2>/<y^2>)`` over the active cells: a SECOND-MOMENT aspect.

    The obvious ``max|x| / max|y|`` is set by single extreme cells, so it moves
    in quantised jumps -- 0.100 per cell on the lattice used here -- and cannot
    resolve a shape change smaller than that. Two growth runs differing by one
    cell then look like a 7 % shape disagreement. This measure weights every
    active cell and moves smoothly.
    """
    c = centers[on]
    if c.shape[0] < 2:
        return float("nan")
    c = c - c.mean(axis=0)
    sx = float((c[:, 0] ** 2).mean())
    sy = float((c[:, 1] ** 2).mean())
    return float(np.sqrt(sx / sy)) if sy > 0 else float("nan")


def neighbours(centers: np.ndarray, on: np.ndarray, pitch: float) -> np.ndarray:
    """Inactive cells adjacent to the active set: the candidate front."""
    r2 = (NEIGHBOUR_REACH * pitch) ** 2
    act = centers[on]
    near = np.zeros(centers.shape[0], bool)
    for p in act:
        near |= ((centers[:, 0] - p[0]) ** 2
                 + (centers[:, 1] - p[1]) ** 2
                 + (centers[:, 2] - p[2]) ** 2) < r2
    return near & ~on


def step(P, d, g, areas, centers, on, *, pitch, batch=1, warm=None):
    """One growth increment: add the ``batch`` cells with the best dE/dA.

    Every candidate costs a warm-started solve, which is affordable only
    because the answer is unique -- a warm start can change the iteration
    count and not the result -- so the previous configuration's slip is a
    legitimate starting point for every trial.

    Returns ``(on, solution, energy, best_gain, n_candidates)``.
    """
    base = St.solve(P, d, g, 1.0, include=on, warm=warm)
    E0 = St.released_energy(P, base.delta, areas)
    cand = np.flatnonzero(neighbours(centers, on, pitch))
    if cand.size == 0:
        return on, base, E0, float("nan"), 0
    gain = np.empty(cand.size)
    for k, c in enumerate(cand):
        trial = on.copy()
        trial[c] = True
        s = St.solve(P, d, g, 1.0, include=trial, warm=base.delta)
        gain[k] = (St.released_energy(P, s.delta, areas) - E0) / areas[c]
    take = cand[np.argsort(-gain)[:batch]]
    on = on.copy()
    on[take] = True
    sol = St.solve(P, d, g, 1.0, include=on, warm=base.delta)
    return (on, sol, St.released_energy(P, sol.delta, areas),
            float(gain.max()), int(cand.size))
