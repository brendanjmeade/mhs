"""Crack growth by work optimization, on a fixed lattice.

GROWTH AS ACTIVATION, WHICH IS WHY THIS IS CHEAP. The matrix is assembled ONCE
for a candidate lattice and growth turns cells on through ``static.solve``'s
``include`` mask. No remeshing, no re-assembly, and the front is implicit in
which cells are active -- the same machinery the fault-network growth verb
uses.

OUT OF PLANE IT CANNOT BE ACTIVATION, because a kinked cell is in no lattice,
so :func:`kink_candidates` BUILDS one on each front edge. That is the 3-D
front representation the XFEM literature calls the hard part, and it stays
cheap here for two reasons: the edge IS the exact local tangent, so no tangent
has to be estimated, and the candidate blocks are rectangular, so the parent
block is assembled once and only the border is new (verified bit-identical to
assembling the whole thing).

WHAT IT SAYS SO FAR IS THAT THE FRONT STAYS PUT. Under pure remote shear the
preferred kink is theta = 0 at both the mode-II and mode-III tips, with gains
symmetric in theta to six digits as the loading's symmetry demands -- a shear
crack grows in its own plane, which is why faults are planar surfaces. That is
not merely the trigonometry of a kinked plane resolving less shear: holding
the candidate's drive at full ``d_tau`` to isolate the elastic term still
costs 38.0 % at the mode-III tip and 36.9 % at the mode-II tip by 30 degrees.
The two tips resist nearly EQUALLY; an earlier reading had the mode-III tip
four times weaker, which was two coincident candidates (see
:func:`front_edges`).

The honest limit on that is arc length against mode purity. An arc must be
long to suppress the one-cell bump term and short to sample one mode, and at
a/h = 12 an arc of nine spans about 33 degrees and mixes the two -- visible in
the drive at the "mode-III" tip reading 0.7533 at 30 degrees where pure mode
III would give cos(30) = 0.8660. Mode purity within 10 degrees needs a/h = 48,
about 16,700 cells.

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
isotropic and the circle is restored. A parameter-free prediction containing a
SIGN CHANGE, which is much harder to reproduce by accident than a magnitude,
and it holds. Fourteen increments from an aspect-1.35 ellipse, 2346 cells,
h = 0.075, ranked by ``arc = 9`` gains and measured with ``shape_ratio``:

    nu = 0.00    1.3501 -> 1.1399    -0.210   rounds up, monotonically
    nu = 0.25    1.3501 -> 1.3526    +0.002   holds
    nu = 0.40    1.3501 -> 1.5735    +0.223   still rising at the last step

A directional bug in the criterion could not do this, because a bug does not
know about nu. And nu moved by the intended amount rather than something else
changing with it: ``E/analytic`` is 1.0417, 1.0429, 1.0443 -- constant to
0.25 % -- while ``penny(nu).energy`` itself moves 33 %, from 4/3 to 1.

SO THE SHAPE HAS AN ATTRACTOR, not a preferred direction of change, and the
nu = 0.25 row above is a FIXED POINT rather than a null result. The test of a
fixed point is not the path but whether different starts converge on it, which
they do, with OPPOSITE SIGNS from one parameter:

    nu = 0.25   from 1.10 -> 1.2545  (+0.152)   from 1.60 -> 1.4303  (-0.183)
    nu = 0.40   from 1.35 -> 1.5735  (+0.223)   from 1.90 -> 1.7046  (-0.189)

bracketing a*(0.25) in [1.25, 1.43] -- with the holding run at 1.3526, near
the midpoint -- and a*(0.40) in [1.57, 1.70]. The brackets do not overlap, so
the preferred aspect demonstrably RISES with nu, from a*(0) = 1. Starts are
matched in AREA, not in semi-axis, or a smaller start would simply grow faster
and the comparison would be of sizes. The nu = 0.40 approach is non-monotone
(1.90 falls to 1.6025 by step 12, then returns to 1.7046), so that bracket is
loose; nu = 0.25 is the clean one.

MESH DEPENDENCE IS THE RISK THIS MODULE EXISTS TO TEST, because sharp-crack
growth is notoriously sensitive to the discretization -- a main reason
phase-field methods exist. The ENERGY passes: growing the same crack on a
lattice rotated by 30 degrees, the maximal inequivalence for a 6-fold lattice,
it agrees to 0.71 % at worst and 0.10 % at the final step over fourteen steps.

THE SHAPE PATH AT nu = 1/4 DOES NOT, AND THE FIXED POINT IS WHY. In
``shape_ratio`` the two lattices' initial ellipses agree to 0.0058 yet the
one-cell-ranked paths end 0.0952 apart, +0.054 against +0.1437, a factor of
2.6 that reads as a defect. It is not: nu = 1/4 starts ON its own attractor,
where the signal is zero, so that is the ratio of two near-zero numbers. The
sensitivity is a signature and the convergence table above is the right
diagnostic. (A first reading instead called the divergence inherited rather
than accumulated, on the crude max|x|/max|y| aspect, whose own initial offset
of 0.0733 hid it.) Away from the fixed point there is no such problem: the nu
trend spans 0.43 against 0.09 of lattice scatter.
"""
from __future__ import annotations

import math
import typing

import numpy as np

import static as St

#: Candidate neighbourhood, in units of the lattice pitch. 1.1 picks up the
#: cells sharing a vertex with the active set and nothing further.
NEIGHBOUR_REACH = 1.1

#: Reach that takes EDGE-SHARING neighbours only. The tiling's centroid
#: separations are 0.577, 1.000, 1.155, ... pitch, so 0.6 takes the first
#: shell alone where ``NEIGHBOUR_REACH`` takes the first two -- which is why
#: the default candidate set is two to three cells thick.
EDGE_REACH = 0.6


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


def neighbours(centers: np.ndarray, on: np.ndarray, pitch: float, *,
               reach: float = NEIGHBOUR_REACH) -> np.ndarray:
    """Inactive cells adjacent to the active set: the candidate front."""
    r2 = (reach * pitch) ** 2
    act = centers[on]
    near = np.zeros(centers.shape[0], bool)
    for p in act:
        near |= ((centers[:, 0] - p[0]) ** 2
                 + (centers[:, 1] - p[1]) ** 2
                 + (centers[:, 2] - p[2]) ** 2) < r2
    return near & ~on


def front_row(centers: np.ndarray, on: np.ndarray, pitch: float) -> np.ndarray:
    """Candidates sharing an EDGE with the active set: the front, one cell thick.

    :func:`neighbours` at the default reach returns two to three rows, and an
    arc patch must be a single row or its "width" conflates along-front extent
    with outward thickness -- which is the very distinction the one-cell bias
    turns on. ``EDGE_REACH`` takes the first separation shell and no more.
    """
    return neighbours(centers, on, pitch, reach=EDGE_REACH)


def arc_patch(centers: np.ndarray, row: np.ndarray, cell: int,
              width: int) -> np.ndarray:
    """The ``width`` front-row cells nearest ``cell``, itself included: an ARC.

    Nearest by Euclidean distance among the row rather than by azimuth about
    the crack's centroid, because azimuth assumes a convex planar front and
    this has to survive one that is neither.
    """
    d2 = ((centers[row] - centers[cell]) ** 2).sum(axis=1)
    return row[np.argsort(d2, kind="stable")[:width]]


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
                max_sweeps=St.MAX_SWEEPS, arc=1):
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

    ``arc`` IS THE FIX FOR THE 20 % BIAS :func:`front_anisotropy` MEASURED.
    At ``arc = 1`` each trial adds one triangle, a bump whose along-front
    length and outward height are both ``pitch``, so its own self-interaction
    is second order in a ratio the lattice holds FIXED -- the error survives
    refinement. At ``arc = w`` the trial instead adds the ``w`` front-row
    cells nearest the candidate and divides by their total area, which is
    Rice's ``integral G ds`` over a longer arc and drives the self term down
    like ``pitch / arc length``. Measured against the closed form: 9 to 15
    cells recovers it to 4-9 %, where one cell is 18 % low.

    The gain is attributed to the candidate at the arc's centre, so the
    RANKING is smoothed over the arc while the increment :func:`step` takes
    is unchanged. Direction is then resolved to about the arc's angular
    width -- 36 degrees for 9 cells on the production front, 60 for 15 --
    which is ample for a two-fold signal, and is not the axis along which a
    kink angle is measured. A side effect worth knowing: because neighbouring
    centres share most of their arc, the best ``batch`` centres come out as a
    contiguous stretch of front rather than scattered cells, so the increment
    is a genuine front advance.

    ``arc > 1`` restricts candidates to :func:`front_row`, since an arc that
    is two cells thick is not an arc. ``arc = 1`` keeps the full
    :func:`neighbours` set, so it reproduces earlier runs exactly.

    Returns ``(cand, gain, base, E0)``.
    """
    if arc < 1:
        raise ValueError(f"arc must be at least 1, got {arc!r}")
    base = _converged(P, d, g, on, warm, max_sweeps)
    E0 = St.released_energy(P, base.delta, areas)
    cand = np.flatnonzero(front_row(centers, on, pitch) if arc > 1
                          else neighbours(centers, on, pitch))
    gain = np.empty(cand.size)
    for k, c in enumerate(cand):
        patch = arc_patch(centers, cand, c, arc) if arc > 1 else [c]
        trial = on.copy()
        trial[patch] = True
        s = _converged(P, d, g, trial, base.delta, max_sweeps)
        gain[k] = ((St.released_energy(P, s.delta, areas) - E0)
                   / areas[patch].sum())
    return cand, gain, base, E0


def front_anisotropy(centers, on, cand, gain, slip_dir, *, arc=1, areas=None):
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
    ``nu`` are safe; magnitudes carry the bias. ``front_gains(arc=w)`` is the
    fix and ``arc`` here must MATCH the ``arc`` the gains were measured with.

    THE ARC'S OWN AVERAGING IS THE OPPOSITE-SIGNED ERROR, and it has to be
    handled or it replaces the one it cures. A gain measured over an arc is
    that arc's AVERAGE of the azimuthal factor, not its value at the centre,
    and averaging compresses any ratio toward one: fitting arc-9 gains as if
    they were centre values reads c = 0.62 where the truth is 0.60, and arc-15
    gains read 0.66. Because the model is linear in ``c``, the remedy is exact
    rather than a correction factor -- the two design columns become each
    candidate's area-weighted ``<cos^2 phi>`` and ``<sin^2 phi>`` over ITS OWN
    patch, which reduces to the centre values at ``arc = 1``.

    Returns ``(c, r2)``.
    """
    ctr = centers[on].mean(axis=0)

    def frame(idx):
        rel = centers[idx] - ctr
        al = rel[:, 0] * slip_dir[0] + rel[:, 1] * slip_dir[1]
        ac = -rel[:, 0] * slip_dir[1] + rel[:, 1] * slip_dir[0]
        return np.arctan2(ac, al), np.hypot(al, ac)

    if arc > 1:
        cos2 = np.empty(cand.size)
        sin2 = np.empty(cand.size)
        rb = np.empty(cand.size)
        for k, c in enumerate(cand):
            patch = arc_patch(centers, cand, c, arc)
            ph, rr = frame(patch)
            w = (np.ones(patch.size) if areas is None
                 else np.asarray(areas, float)[patch])
            w = w / w.sum()
            cos2[k] = float((w * np.cos(ph) ** 2).sum())
            sin2[k] = float((w * np.sin(ph) ** 2).sum())
            rb[k] = float((w * rr).sum())
    else:
        phi, rb = frame(cand)
        cos2, sin2 = np.cos(phi) ** 2, np.sin(phi) ** 2
    rb = rb - rb.mean()
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


class Front(typing.NamedTuple):
    """The active set's boundary, ONE ENTRY PER DISTINCT CANDIDATE POSITION."""

    cell: np.ndarray      # (m,)      parent cell index
    edge: np.ndarray      # (m, 2, 3) the shared edge, in the parent's order
    apex: np.ndarray      # (m, 3)    the parent's third vertex
    mid: np.ndarray       # (m, 3)    the planar candidate's centroid


def front_edges(tris: np.ndarray, on: np.ndarray, *, tol=1e-3) -> Front:
    """Edges of the active set whose across-edge neighbour is not active.

    EDGE-BASED, NOT CELL-BASED, because the edge IS the exact local front
    tangent -- so out-of-plane growth needs no tangent estimate. (Estimating
    one from a cell's active neighbours is 43 degrees wrong at worst here,
    since a front triangle has only 1.14 active neighbours on average.)

    The planar continuation across an edge is the parallelogram completion
    ``q = v0 + v1 - p``, which on an equilateral TILING is exactly the
    neighbouring cell: measured, 162 hits out of 162 with centroid agreement
    8e-17. That identity is what makes ``theta = 0`` in
    :func:`kink_candidates` reproduce planar growth rather than approximate it.

    ONE ENTRY PER POSITION IS A CORRECTNESS REQUIREMENT, not tidiness. A
    triangle just outside a ragged front can share edges with TWO different
    active cells, so two edges generate the SAME candidate. Admitting both
    puts coincident elements in a trial: the added area counts twice while the
    pair's mutual interaction equals its self-interaction, so the gain per unit
    area is wrong. It cost 31 % in the one quantity the kink scan cross-checks
    -- two candidates 0.000 h apart turned a tip-to-tip gain ratio of 0.677
    into 0.515 against a closed form of 0.750 -- and it also invented a
    four-fold difference between the mode-II and mode-III tips that is not
    there.
    """
    cen = tris.mean(axis=1)
    h = float(np.sqrt(4.0 / np.sqrt(3.0) * 0.5 * np.linalg.norm(
        np.cross(tris[0, 1] - tris[0, 0], tris[0, 2] - tris[0, 0]))))
    cell, edge, apex, mid = [], [], [], []
    for i in np.flatnonzero(on):
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            v0, v1, p = tris[i, a], tris[i, b], tris[i, c]
            m = (v0 + v1 + (v0 + v1 - p)) / 3.0
            d2 = ((cen - m) ** 2).sum(axis=1)
            j = int(np.argmin(d2))
            if d2[j] <= (tol * h) ** 2 and on[j]:
                continue                       # the neighbour is active
            if mid and np.min(((np.array(mid) - m) ** 2).sum(axis=1)) < (
                    tol * h) ** 2:
                continue                       # this position already taken
            cell.append(i)
            edge.append([v0, v1])
            apex.append(p)
            mid.append(m)
    return Front(np.array(cell, np.int64), np.array(edge).reshape(-1, 2, 3),
                 np.array(apex).reshape(-1, 3), np.array(mid).reshape(-1, 3))


def kink_candidates(front: Front, theta: float,
                    which=None) -> np.ndarray:
    """``(m, 3, 3)`` candidate triangles, rotated out of plane by ``theta``.

    Growth out of plane CANNOT be activation, which is what separates this
    from :func:`step`: a kinked cell is in no lattice, so it is built. Each
    candidate keeps its parent's front edge and rotates the planar
    continuation apex about that edge, which is rigid -- edge length and area
    are preserved to 1.000000 at every ``theta``.

    THE VERTEX ORDER IS (v1, v0, q), AND THE LATTICE DECIDES THAT, not taste:
    adjacent triangles in a consistent tiling traverse their shared edge in
    OPPOSITE directions, and 288 of 288 interior edges match (v1, v0, q)
    against 0 of 288 for (v0, v1, q). The wrong order flips the slip frame's
    handedness and so the SIGN of the self-interaction -- measured, +618.75
    where the parent diagonal is -618.75 -- which destroys the negative
    definiteness the LCP rests on while ``tdcs.unit_normals`` still reports a
    perfectly innocent ``n_z = 1``. Callers must still check the projected
    diagonal; this only avoids guaranteeing it wrong.

    ``theta``'s sense is tied to the original ``v0 -> v1`` direction so its
    sign keeps one geometric meaning despite that swap. Beyond about 60
    degrees the candidate normal approaches horizontal, where ``tdcs``'s
    hemisphere flip is float noise, so callers cap it.
    """
    e = front.edge if which is None else front.edge[which]
    p = front.apex if which is None else front.apex[which]
    v0, v1 = e[:, 0], e[:, 1]
    q = v0 + v1 - p
    ax = v1 - v0
    ax = ax / np.linalg.norm(ax, axis=1, keepdims=True)
    m = 0.5 * (v0 + v1)
    r = q - m
    rot = (r * np.cos(theta)
           + np.cross(ax, r) * np.sin(theta)
           + ax * (ax * r).sum(axis=1, keepdims=True) * (1.0 - np.cos(theta)))
    return np.stack([v1, v0, m + rot], axis=1)


def step(P, d, g, areas, centers, on, *, pitch, batch=1, warm=None,
         max_sweeps=St.MAX_SWEEPS, arc=1):
    """One growth increment: add the ``batch`` cells with the best ``dE/dA``.

    ``arc`` changes how the candidates are RANKED, never how many cells are
    added; see :func:`front_gains` for why one cell is the wrong probe.

    Returns ``(on, solution, energy, best_gain, n_candidates)``.
    """
    cand, gain, base, E0 = front_gains(P, d, g, areas, centers, on,
                                       pitch=pitch, warm=warm,
                                       max_sweeps=max_sweeps, arc=arc)
    if cand.size == 0:
        return on, base, E0, float("nan"), 0
    take = cand[np.argsort(-gain)[:batch]]
    on = on.copy()
    on[take] = True
    sol = _converged(P, d, g, on, base.delta, max_sweeps)
    return (on, sol, St.released_energy(P, sol.delta, areas),
            float(gain.max()), int(cand.size))
