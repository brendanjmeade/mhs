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
factor shared with dE/dA, so THAT bias cancels in the comparison between
candidates that work optimization actually makes.

A SECOND BIAS DOES NOT CANCEL, and it was found by the nu sweep below rather
than by phase 1, which could not have seen it: phase 1 compared whole cracks
of different radii, where the bias is common to both. Comparing candidates at
different AZIMUTHS on one front is a different question, and there the answer
is anisotropic by about 20 % -- see ``front_anisotropy``.

A SHEAR CRACK DOES NOT TEND TO A CIRCLE, and assuming it did cost a run. The
circle is the attractor for a TENSILE crack. On a shear crack the front is not
isotropic: where the slip is perpendicular to the front it is mode II, where
parallel, mode III, and :func:`penny` gives the consequence,
``G(phi)/G(0) = cos^2 phi + (1 - nu) sin^2 phi``. More energy is released
growing ALONG the slip direction, so the crack elongates. A first version of
this experiment treated that elongation as a failure, having imported a
tensile-crack intuition into a shear problem.

nu IS THE DECISIVE TEST OF THAT READING, because the whole anisotropy is
carried by the (1 - nu) factor: it VANISHES at nu = 0, where the front becomes
isotropic and the circle is restored as the attractor. A parameter-free
prediction containing a SIGN CHANGE, and it holds. Fourteen increments from an
aspect-1.35 ellipse, 2346 cells, h = 0.075, measured with ``shape_ratio``:

    nu = 0.00    1.3501 -> 1.1523    -0.198   rounds up, monotonically
    nu = 0.25    1.3501 -> 1.4044    +0.054
    nu = 0.40    1.3501 -> 1.6236    +0.273

Monotone in nu with the sign change between 0 and 0.25. A directional bug in
the criterion could not do this, because a bug does not know about nu. The
energy confirms nu moved as intended rather than something else changing:
``E/analytic`` is 1.0417, 1.0429, 1.0443 -- constant to 0.25 % -- while
``penny(nu).energy`` itself moves 33 %, from 4/3 to 1.

THE MAGNITUDES CARRY A KNOWN 20 % BIAS and the signs do not, which is worth
keeping straight. ``front_anisotropy`` tracks it down: the criterion ranks
SINGLE-cell gains, and a one-cell bump's own self-interaction does not vanish
under refinement because the lattice fixes its aspect ratio. Spreading the
perturbation along the front recovers the closed form to 4-9 %. So the crack
elongates for the predicted reason and in the predicted direction, but it
elongates about 20 % too eagerly.

MESH DEPENDENCE IS THE RISK THIS MODULE EXISTS TO TEST, because sharp-crack
growth is notoriously sensitive to the discretization -- a main reason
phase-field methods exist. The ENERGY passes: growing the same crack on a
lattice rotated by 30 degrees, the maximal inequivalence for a 6-fold lattice,
it agrees to 0.71 % at worst and 0.10 % at the final step over fourteen steps.

THE SHAPE PATH DOES NOT, and the calibrated measure is what revealed it. In
``shape_ratio`` the two lattices' initial ellipses agree to 0.0058, yet the
paths end 0.0952 apart -- so the divergence is ACCUMULATED, not inherited.
A first reading of this called it inherited, on the crude max|x|/max|y|
aspect, whose initial offset is 0.0733 and which therefore hid the growth of
the discrepancy inside its own discretization error. Why nu = 0.25 in
particular: +0.054 over fourteen steps is near the balance point between
elliptical rounding and (1 - nu) elongation, and a path near a balance point
is sensitive -- the rotated lattice gives +0.1437, a factor 2.6. The nu TREND
survives because 0.09 of lattice scatter is small against the 0.47 swing from
nu = 0 to nu = 0.40, but THE nu = 0.25 MAGNITUDE ALONE IS NOT RESOLVED. Both
lattices do agree on its sign.
"""
from __future__ import annotations

import math
import typing

import numpy as np

import static as St

#: Candidate neighbourhood, in units of the lattice pitch. 1.1 picks up the
#: cells sharing a vertex with the active set and nothing further.
NEIGHBOUR_REACH = 1.1


class Penny(typing.NamedTuple):
    """Closed-form coefficients of the penny-shaped shear crack at one ``nu``."""

    slip: float      # delta_0     = slip   * d_tau a^1 / mu
    energy: float    # E           = energy * d_tau^2 a^3 / mu
    g: float         # G = dE/dA   = g      * d_tau^2 a^1 / mu
    front: float     # G(phi)/G(0) = cos^2 phi + front * sin^2 phi


def penny(nu: float) -> Penny:
    """The penny-shaped shear crack at general ``nu``: the analytic reference.

    Eshelby's crack is nearly always quoted at ``nu = 1/4``, where the
    coefficients are ``24/(7 pi)``, ``8/7`` and ``12/(7 pi)``, and phases 0 and
    1 were checked against exactly those numbers. The ``nu`` dependence had to
    be written down for the ``nu`` sweep, and it lives here because three call
    sites were otherwise about to carry a copy of the ``nu = 1/4`` values with
    the dependence left implicit:

        delta_0     = 8(1-nu)/(pi(2-nu))   d_tau a^1   / mu
        E           = (8/3)(1-nu)/(2-nu)   d_tau^2 a^3 / mu
        G = dE/dA   = (4/pi)(1-nu)/(2-nu)  d_tau^2 a^1 / mu
        G(phi)/G(0) = cos^2 phi + (1-nu) sin^2 phi

    The first three share the factor ``(1-nu)/(2-nu)`` and so say nothing the
    others do not. The FOURTH is the one that matters: ``phi`` is measured from
    the slip direction, the ratio is below one for every ``nu > 0``, and it is
    therefore why a shear crack releases more energy growing ALONG slip. It is
    also the only one of the four that a directional bug in the growth
    criterion could not fake, since the other three are azimuthal integrals.

    THE FOURTH IS CONSISTENT WITH THE THIRD, which is worth stating because it
    was derived by a different route and a slip there would have been invisible.
    Taking the near-tip asymptotics, ``K_III/K_II = (1-nu) tan phi``, so with
    ``G = ((1-nu) K_II^2 + K_III^2) / 2mu`` the front profile above follows;
    averaging that over ``phi`` with the penny-crack amplitude
    ``K_II(0) = 4 d_tau sqrt(pi a) / (pi (2-nu))`` returns ``g`` exactly,
    checked numerically to 1e-6 (quadrature) at nu = 0, 0.1, 0.25, 0.4, 0.49.
    So the azimuthal shape and the global rate are one statement, not two.
    """
    if not 0.0 <= nu < 0.5:
        raise ValueError(f"nu must lie in [0, 1/2), got {nu!r}")
    f = (1.0 - nu) / (2.0 - nu)
    return Penny(slip=8.0 * f / math.pi, energy=8.0 * f / 3.0,
                 g=4.0 * f / math.pi, front=1.0 - nu)


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


def _converged(P, d, g, mask, warm, max_sweeps) -> St.Solution:
    """A converged solve, or a refusal -- never a quietly truncated one.

    An unconverged PGS has not finished growing the slip, so its energy is too
    LOW, and the growth criterion is a DIFFERENCE of energies between
    candidates: the error does not cancel, the step goes to the wrong cell,
    and every number printed stays entirely plausible.
    """
    sol = St.solve(P, d, g, 1.0, include=mask, warm=warm,
                   max_sweeps=max_sweeps)
    if not sol.converged:
        raise RuntimeError(
            f"PGS stopped at {max_sweeps} sweeps with residual "
            f"{sol.residual:.3e} on {int(mask.sum())} active cells; the "
            f"energy is too low by an unknown amount, so the growth "
            f"decision it feeds is not usable")
    return sol


def front_gains(P, d, g, areas, centers, on, *, pitch, warm=None,
                max_sweeps=St.MAX_SWEEPS):
    """``dE/dA`` for every candidate cell on the front, and the base state.

    Every candidate costs a warm-started solve, which is affordable only
    because the answer is unique -- a warm start can change the iteration
    count and not the result -- so the previous configuration's slip is a
    legitimate starting point for every trial.

    Separate from :func:`step` because the AZIMUTHAL STRUCTURE of ``gain`` is
    itself the measurement that justifies the criterion. Around a circular
    shear crack it must follow ``cos^2 phi + (1 - nu) sin^2 phi``, ``phi``
    from the slip direction -- a closed form with no free parameter, which is
    what separates "mode mixity elongates the crack" from "my criterion has a
    directional bug". Reading that off a growth path cannot distinguish the
    two; reading it off one front can.

    Returns ``(cand, gain, base, E0)``.
    """
    base = _converged(P, d, g, on, warm, max_sweeps)
    E0 = St.released_energy(P, base.delta, areas)
    cand = np.flatnonzero(neighbours(centers, on, pitch))
    gain = np.empty(cand.size)
    for k, c in enumerate(cand):
        trial = on.copy()
        trial[c] = True
        s = _converged(P, d, g, trial, base.delta, max_sweeps)
        gain[k] = (St.released_energy(P, s.delta, areas) - E0) / areas[c]
    return cand, gain, base, E0


def front_anisotropy(centers, on, cand, gain, slip_dir):
    """``c`` in ``dE/dA ~ cos^2 phi + c sin^2 phi`` around a crack front.

    ``phi`` is measured from ``slip_dir`` in the crack plane, so ``c`` is to be
    read against :func:`penny`'s ``front = 1 - nu``. This is the measurement
    that justifies the growth criterion rather than merely exercising it: the
    energy, the slip and the global ``dE/dA`` are azimuthal integrals and a
    criterion with a directional bug would reproduce all three.

    THE RADIUS COVARIATE IS NOT COSMETIC. :func:`neighbours` returns two to
    three rows of cells, so how far a candidate sits beyond the front varies
    by most of a cell width, and ``dE/dA`` depends strongly on that -- it is
    most of the variance in ``gain`` and none of it is azimuthal. Dropping it
    recovers the same ``c`` to under one percent but with an R2 of 0.001-0.17,
    which reads as a broken measurement; with it, 0.78-0.79. The model is
    MULTIPLICATIVE, ``(alpha + beta (r - rbar))`` scaling the azimuthal
    factor, because the radial falloff attenuates the whole gain rather than
    adding to it. Nonlinear in ``c`` through one column only, so a grid beats
    a solver.

    WHAT THIS RETURNS IS NOT QUITE G, AND READING IT AS G COSTS 20 %. Against
    ``penny(nu).front`` the measurement is linear in ``nu`` with the right
    intercept and a slope 35 % too steep. The reference is not the suspect
    (``1 - nu`` re-derives symbolically from the Eshelby slip coefficient
    alone) and neither is resolution: ``c`` plateaus under refinement in
    ``h``, and falling ``eps/h`` moves it AWAY from the closed form.

    It is the SHAPE of the perturbation. Rice's weight function makes the
    first-order energy change ``integral G(s) da(s) ds``, so a perturbation
    spread ALONG the front recovers ``G`` -- but a single triangle is a bump
    whose along-front length and outward height are both ``h``, and its own
    self-interaction is second order and survives ``h -> 0`` because the
    lattice fixes the bump's ASPECT RATIO. Spreading it over a 24-51 degree
    arc recovers the closed form to 4-9 %, which is the measurement that
    identifies the bump term as the cause.

    SO :func:`step`, WHICH RANKS SINGLE-CELL GAINS, INHERITS IT and
    over-weights slip-parallel growth by about 20 %. Signs and orderings in
    ``nu`` are safe; magnitudes carry the bias.

    Returns ``(c, r2)``.
    """
    rel = centers[cand] - centers[on].mean(axis=0)
    along = rel[:, 0] * slip_dir[0] + rel[:, 1] * slip_dir[1]
    across = -rel[:, 0] * slip_dir[1] + rel[:, 1] * slip_dir[0]
    phi = np.arctan2(across, along)
    rb = np.hypot(along, across)
    rb = rb - rb.mean()
    cos2, sin2 = np.cos(phi) ** 2, np.sin(phi) ** 2
    best_c, best_r = float("nan"), np.inf
    for c in np.linspace(0.2, 1.8, 1601):
        u = cos2 + c * sin2
        M = np.stack([u, u * rb], axis=1)
        coef, *_ = np.linalg.lstsq(M, gain, rcond=None)
        r = float(((gain - M @ coef) ** 2).sum())
        if r < best_r:
            best_c, best_r = float(c), r
    ss = float(((gain - gain.mean()) ** 2).sum())
    return best_c, (1.0 - best_r / ss if ss > 0 else float("nan"))


def step(P, d, g, areas, centers, on, *, pitch, batch=1, warm=None,
         max_sweeps=St.MAX_SWEEPS):
    """One growth increment: add the ``batch`` cells with the best ``dE/dA``.

    Returns ``(on, solution, energy, best_gain, n_candidates)``.
    """
    cand, gain, base, E0 = front_gains(P, d, g, areas, centers, on,
                                       pitch=pitch, warm=warm,
                                       max_sweeps=max_sweeps)
    if cand.size == 0:
        return on, base, E0, float("nan"), 0
    take = cand[np.argsort(-gain)[:batch]]
    on = on.copy()
    on[take] = True
    sol = _converged(P, d, g, on, base.delta, max_sweeps)
    return (on, sol, St.released_energy(P, sol.delta, areas),
            float(gain.max()), int(cand.size))
