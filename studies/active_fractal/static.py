"""The static frictional solution: which faults slip under a given stress.

THE PRIMARY OBJECT OF THIS STUDY, and the reason is that it is UNIQUE. At one
load the quasi-static Coulomb problem is

    w = A delta + q,   A = -P,   q = g - lam d
    delta >= 0,   w >= 0,   delta . w = 0

-- a Linear Complementarity Problem, ``w`` being each element's margin to
failure. Every element is either locked and below strength, or slipping and
exactly at strength, and nothing else is admissible. When ``A`` has positive
definite symmetric part it is a P-matrix, and an LCP with a P-matrix has
EXACTLY ONE solution for every ``q`` (Cottle, Pang & Stone, Thm 3.3.7).

So "which faults slip under this stress field" has one answer, and it depends
on no loading path, no update rule, and no frictional weakening whatsoever.
``cascade.py`` exists for the DIFFERENT question of how slip arrives -- in what
sized jumps -- which does need a strength drop, because the stiffness criterion
forbids a finite jump without one. Measured on the published network, the two
static solutions BRACKET the whole loading history exactly:

    A(mu_s) 10822  subset of  A(history) 10977  subset of  A(mu_d) 11299

with zero violations either way. The strength drop moves the active set by at
most 2.3 % of the network; it does not decide it.

MOLLIFICATION IS WHAT MAKES THIS WELL POSED EVERYWHERE ON THE FAULT, rather
than being a numerical convenience. The uniqueness theorem needs a finite,
correctly signed diagonal ``P_ii`` -- how much an element's own slip relieves
its own shear. For a classical dislocation element that number exists at ONE
point, the centroid, which sits a fixed 0.29 h from the element's own edges
and so a fixed distance from the 1/d singularity of its bounding loop:
measured (commit 1ef89ab, and the on-fault page), it is finite, negative, and
moves by under a per cent as the centroid is approached along the normal, so
a centroid-collocated classical matrix is a perfectly usable one. What the
classical kernel does NOT have is a value anywhere else on the element -- it
is hypersingular on the surface and 1/d at every edge -- so its ``P_ii`` is a
property of where it was sampled rather than of the fault. Mollification
smears the slip over a width ``eps`` and the self-interaction becomes a
convergent integral at every point -- and ``eps`` is physical rather than
arbitrary, a fault being a gouge zone of finite thickness, so ``P_ii`` is the
self-stiffness of a slip patch of that width, the same number at the centroid
and between centroids.

TWO THINGS THAT ARE MEASURED RATHER THAN GUARANTEED, and both matter before
this is trusted on a new network.

* POSITIVE DEFINITENESS. The continuum hypersingular operator is coercive for
  a single crack (Costabel, Stephan), but a COLLOCATION discretization on a
  network of arbitrary orientations and near-contacts does not inherit that.
  Ours did not, the first time: exactly one negative eigenvalue out of 2826,
  with 99.4 % of its weight on a single nearly perpendicular pair whose
  collocation point sat on the other's face. The face-clearance floor is how
  it was earned back, and ``min eig sym(-P)`` is checked per run.
* ``P`` IS STRONGLY NON-SYMMETRIC (measured |P - P^T|max/|P|max = 0.72), as
  collocation is even for a self-adjoint operator where Galerkin would not be.
  Uniqueness survives, needing only positive definiteness and not symmetry,
  but the problem is a genuine LCP and not a convex QP -- so the projected
  Gauss-Seidel below converging in ~141 sweeps is an empirical fact here, not
  a theorem, and it is reported rather than assumed.

WHAT THE CHEAPNESS BUYS. A static solve is two to three orders of magnitude
cheaper than a loading history -- ~141 sweeps against 165,659 events -- and
the critically-stressed null is free. That is what makes the incremental
question affordable: build one large network, then ask what each ADDED fault
does to the rest of it, by re-solving on nested subsets of a matrix assembled
once.
"""
from __future__ import annotations

import dataclasses

import numpy as np

#: Natural-residual tolerance, GPa. Matches ``cascade.F_TOL``: 1e-9 GPa is 1 Pa
#: against strengths of order 0.1 GPa.
TOL = 1.0e-9

#: Projected Gauss-Seidel sweeps before a solve is declared non-convergent.
#: Cold starts take ~141 on the published network; a warm start takes a
#: handful. Hitting this is reported, never silently accepted.
MAX_SWEEPS = 5_000


@dataclasses.dataclass
class Solution:
    """One static solution. ``active`` is the thing the study measures."""

    delta: np.ndarray           # (n,) slip, zero outside `include`
    lam: float
    sweeps: int
    residual: float             # natural residual, max|min(delta, w)|
    converged: bool
    n_included: int

    @property
    def active(self) -> np.ndarray:
        return self.delta > 0.0

    @property
    def n_active(self) -> int:
        return int((self.delta > 0.0).sum())


def solve(P: np.ndarray, d: np.ndarray, g: np.ndarray, lam: float, *,
          include: np.ndarray | None = None,
          warm: np.ndarray | None = None,
          tol: float = TOL, max_sweeps: int = MAX_SWEEPS,
          verbose: bool = False) -> Solution:
    """Solve the LCP at load ``lam`` by projected Gauss-Seidel.

    ``include`` restricts the problem to a SUBNETWORK without copying a
    submatrix: excluded elements are simply held at zero slip, which is
    exactly what it means for them not to be there, and costs one skipped
    index rather than a 3 GiB slice. That is what makes the growth experiment
    affordable -- every subnetwork reads the same assembled matrix.

    ``warm`` starts from a previous solution. The LCP's answer is unique, so a
    warm start can only change the ITERATION COUNT and never the answer; it is
    how a leave-one-out sweep costs a few sweeps instead of a hundred.

    Convergence is the NATURAL RESIDUAL ``max|min(delta, w)|``, which is zero
    exactly at the solution -- rather than a step-size test, which reports
    success on a solver that has merely slowed down.

    WHILE NOTHING LOCKS, THIS IS A LINEAR SOLVE, and the consequence is worth
    stating because it rules out a whole class of experiment. On a uniformly
    driven, uniformly strong crack every element slips, so the complementarity
    never binds: slip is exactly proportional to the driving and energy to its
    square. Measured on a disc, ``moment / driving`` is constant to six figures
    across a THOUSANDFOLD change in the Coulomb margin -- 1.0 down to 0.001 --
    with no element locked at any level. EVERY DIMENSIONLESS RATIO IS THEREFORE
    BLIND TO THE LOADING AMPLITUDE, so no uniform knob can change a competition:
    not the load, not the confining pressure, not the system stiffness of
    :func:`load_feedback`, not proximity to failure. Each was measured
    separately before the common cause was noticed.

    What survives is geometric. The energy available at a crack's front against
    the same patch in the far field is 4.67, 9.34 and 13.42 at radii 3, 6 and
    9 h -- linear in length, since ``G ~ d_tau^2 L / mu`` grows while an
    isolated patch is fixed -- and a FRICTIONLESS weak site still reaches only
    0.669 of the front at 6 h and 0.465 at 9 h. So one crack outgrows a new one
    by a widening margin, and only spatial heterogeneity can change that.
    """
    n = P.shape[0]
    idx = np.arange(n)
    inv_diag = -1.0 / P[idx, idx]              # P_ii < 0, so positive
    order = (np.flatnonzero(include) if include is not None
             else np.arange(n, dtype=np.int64))
    delta = np.zeros(n) if warm is None else np.array(warm, float)
    if include is not None:
        delta[~include] = 0.0                  # excluded means absent
    q = P.T @ delta if delta.any() else np.zeros(n)

    res = np.inf
    for sweep in range(1, max_sweeps + 1):
        for i in order:
            f = lam * d[i] + q[i] - g[i]
            new = delta[i] + f * inv_diag[i]
            if new < 0.0:
                new = 0.0
            change = new - delta[i]
            if change != 0.0:
                delta[i] = new
                q += P[i] * change             # the contiguous column read
        w = g - (lam * d + q)
        res = float(np.abs(np.minimum(delta[order], w[order])).max())
        if verbose and (sweep <= 3 or sweep % 25 == 0):
            print(f"      sweep {sweep:4d}  residual {res:.3e}  "
                  f"slipping {int((delta > 0).sum())}")
        if res <= tol:
            return Solution(delta, lam, sweep, res, True, order.size)
    return Solution(delta, lam, max_sweeps, res, False, order.size)


@dataclasses.dataclass
class ConeSolution:
    """One free-rake static solution: the slip VECTOR per element."""
    slip: np.ndarray            # (n, 2) in each element's (strike, dip) frame
    shear: np.ndarray           # (n, 2) total shear carried at the solution
    lam: float
    sweeps: int
    residual: float             # cone natural residual, max |t - Proj_D(t + k s)|
    converged: bool

    @property
    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.slip, axis=1)

    @property
    def active(self) -> np.ndarray:
        return self.magnitude > 0.0

    @property
    def rake_deg(self) -> np.ndarray:
        """Aki-Richards rake of the slip, NaN where locked."""
        out = np.full(self.slip.shape[0], np.nan)
        on = self.active
        out[on] = np.degrees(np.arctan2(self.slip[on, 1], self.slip[on, 0]))
        return out


def full_shear(Ksh: np.ndarray, tau_load: np.ndarray, lam: float,
               slip: np.ndarray) -> np.ndarray:
    """``(n, 2)`` total shear: the regional ``lam tau_load`` plus what every
    element's slip vector puts on every other, from the shear rows of ``K``."""
    return lam * tau_load + np.einsum("jbai,jb->ia", Ksh, slip)


def cone_residual(shear: np.ndarray, slip: np.ndarray, g: np.ndarray,
                  k: np.ndarray) -> float:
    """``max_i |t_i - Proj_{D_i}(t_i + k_i s_i)|``, GPa: zero exactly at a
    solution of the cone problem, for any positive ``k``."""
    r = shear + k[:, None] * slip
    nr = np.linalg.norm(r, axis=1)
    pr = r * np.minimum(1.0, g / np.maximum(nr, 1e-300))[:, None]
    return float(np.linalg.norm(shear - pr, axis=1).max())


def solve_cone(Ksh: np.ndarray, tau_load: np.ndarray, g: np.ndarray,
               lam: float, *, warm: np.ndarray | None = None,
               tol: float = TOL, max_sweeps: int = MAX_SWEEPS,
               verbose: bool = False) -> ConeSolution:
    """The FREE-RAKE static problem: each element's slip is a vector in its
    own plane, and Coulomb friction is a disc rather than a threshold.

    Per element ``i`` with total shear ``t_i`` (``full_shear``) and disc
    radius ``g_i = mu_i sigma'_i`` (``Friction.strength_at``):

        |t_i| <= g_i,   s_i = alpha_i t_i / |t_i|,   alpha_i >= 0,
        alpha_i (g_i - |t_i|) = 0

    -- locked strictly inside the disc, or slipping on its rim PARALLEL to
    the shear it carries. Restricting ``s_i`` to the ray along a fixed
    direction collapses this to :func:`solve`'s scalar LCP, so the fixed rake
    is a CONSTRAINT on this problem, and the two differ exactly where it
    binds: where the neighbours' slip turns an element's shear off the
    regional rake, and where the scalar problem's ``delta >= 0`` forbids an
    element from slipping against it.

    THE METHOD is the contact-dynamics projected Gauss-Seidel (Moreau; Jean
    1999; Jourdan, Alart & Jean 1998): with everything else frozen,
    ``r = t_i + k_i s_i`` and ``s_i <- (r - Proj_D(r)) / k_i``, whose fixed
    point is the local condition exactly for ANY ``k_i > 0``; ``k_i`` is the
    scalar self-stiffness ``max(-K_ii^ss, -K_ii^dd)`` and only sets the rate.
    One contiguous source-major column ``Ksh[i]`` per update, as
    :func:`solve` reads ``P[i]``. ``Ksh`` is ``K[:, :, :2, :]``, the shear
    rows of ``kernel.assemble``'s matrix, ``(n, 2, 2, n)``.

    MEASURED, NOT GUARANTEED, as for :func:`solve`. Uniqueness of the cone
    problem holds when the symmetric part of ``-Ksh`` is positive definite;
    on the Longitudinal Valley Fault it is not (seven negative eigenvalues of
    4996, the supplied mesh's slivers) and yet warm and cold starts agree to
    1e-11 km on all 70 solves of the sweep, which is what ``selfcheck -k
    rake`` and the study report. Genuine Coulomb non-uniqueness (Klarbring
    1990) lives in the normal-stress coupling, which this solver does not
    carry: the radius is the load's, as in :func:`solve` without
    :func:`coulomb_coupled`. A negative radius (an element the load has
    driven into tension) is clamped at zero, where the element slips freely,
    the same thing the scalar problem does with a negative threshold.

    Convergence is the cone's natural residual ``max |t - Proj_D(t + k s)|``,
    zero exactly at a solution, at the scalar solver's tolerance.
    """
    n = Ksh.shape[0]
    idx = np.arange(n)
    k = np.maximum(-Ksh[idx, 0, 0, idx], -Ksh[idx, 1, 1, idx])
    if not np.all(k > 0.0):
        raise ValueError("a non-positive self-stiffness: the matrix is not "
                         "source-major shear rows with a negative diagonal")
    g = np.maximum(np.asarray(g, float), 0.0)
    s = np.zeros((n, 2)) if warm is None else np.array(warm, float)
    T = full_shear(Ksh, tau_load, lam, s)
    res = np.inf
    for sweep in range(1, max_sweeps + 1):
        for i in range(n):
            r0 = T[i, 0] + k[i] * s[i, 0]
            r1 = T[i, 1] + k[i] * s[i, 1]
            nr = (r0 * r0 + r1 * r1) ** 0.5
            if nr <= g[i]:
                n0 = n1 = 0.0                       # locked, inside the disc
            else:
                f = (nr - g[i]) / (k[i] * nr)       # on the rim, along r
                n0, n1 = f * r0, f * r1
            d0, d1 = n0 - s[i, 0], n1 - s[i, 1]
            if d0 != 0.0 or d1 != 0.0:
                s[i, 0], s[i, 1] = n0, n1
                col = Ksh[i]                        # (2, 2, n), one contiguous read
                T[:, 0] += d0 * col[0, 0] + d1 * col[1, 0]
                T[:, 1] += d0 * col[0, 1] + d1 * col[1, 1]
        res = cone_residual(T, s, g, k)
        if verbose and (sweep <= 3 or sweep % 25 == 0):
            print(f"      sweep {sweep:4d}  cone residual {res:.3e}  "
                  f"slipping {int((np.abs(s).sum(axis=1) > 0).sum())}")
        if res <= tol:
            return ConeSolution(s, T, lam, sweep, res, True)
    return ConeSolution(s, T, lam, max_sweeps, res, False)


#: Active-set Newton iterations before a solve is declared non-convergent.
#: Measured: a few on a penny crack, tens on the on-fault page's system.
MAX_NEWTON = 200


def solve_newton(P: np.ndarray, d: np.ndarray, g: np.ndarray, lam: float, *,
                 include: np.ndarray | None = None,
                 warm: np.ndarray | None = None,
                 tol: float = TOL, max_iter: int = MAX_NEWTON,
                 verbose: bool = False) -> Solution:
    """The same LCP as :func:`solve`, by an ACTIVE-SET NEWTON method.

    WHY A SECOND SOLVER. Projected Gauss-Seidel is guaranteed to converge
    when ``sym(-P)`` is positive definite and nothing promises it otherwise;
    on the continuous-P2 operator of the on-fault page's system (14k nodes,
    a/h = 24, eps = 0.1 h) that symmetric part has eigenvalues down to -52
    -- the nodes within 2 eps of the free surface, and the body modes of a
    large rough fault at that eps/h (``onfault.EPS_CONTINUOUS``) -- and PGS
    overflows within a few hundred sweeps. The LCP's
    uniqueness needs ``-P^T`` to be a P-matrix, a weaker thing than a
    definite symmetric part, and this method needs no definiteness at all:
    it guesses the active set, solves the LINEAR system on it exactly, moves
    every node that violates a complementarity condition to the other set,
    and repeats. On a P-matrix each guess is corrected toward the unique
    answer; measured, it takes a few iterations on a penny crack and agrees
    with PGS to the solver tolerance wherever PGS converges (selfcheck
    [nodal]).

    Each iteration is one dense solve on the active block, O(n_active^3):
    seconds at 10k unknowns with a threaded BLAS, which PGS's hundreds of
    sweeps of O(n^2) also cost. ``include`` and ``warm`` mean what they mean
    in :func:`solve`; the starting active set is the warm start's, else the
    nodes the load alone drives past strength.

    CYCLING is the one failure mode of block pivoting: a swap that undoes
    the previous one. It is detected (the active set repeats) and the
    method then moves only the single worst violator per iteration, which
    is Murty's least-index rule in spirit and cannot cycle on a P-matrix.
    """
    n = P.shape[0]
    free = np.ones(n, bool) if include is None else np.asarray(include, bool)
    q = g - lam * d                            # w = q - P^T delta
    if warm is not None:
        S = (np.asarray(warm, float) > 0.0) & free
    else:
        S = (q < 0.0) & free
    delta = np.zeros(n)
    seen: set[bytes] = set()
    single = False
    scale = float(np.abs(q[free]).max()) if free.any() else 1.0
    for it in range(1, max_iter + 1):
        idx = np.flatnonzero(S)
        delta[:] = 0.0
        if idx.size:
            M = -P[np.ix_(idx, idx)].T          # (-P^T) on the active block
            delta[idx] = np.linalg.solve(M, -q[idx])
        w = q - P.T @ delta
        viol_d = S & (delta < -tol)             # slipping backwards
        viol_w = free & ~S & (w < -tol)         # locked but past strength
        res = float(np.abs(np.minimum(delta[free], w[free])).max()) \
            if free.any() else 0.0
        if verbose and (it <= 3 or it % 10 == 0):
            print(f"      newton {it:4d}  residual {res:.3e}  active "
                  f"{int(S.sum())}  swaps {int(viol_d.sum())}+"
                  f"{int(viol_w.sum())}")
        if res <= tol or not (viol_d.any() or viol_w.any()):
            np.maximum(delta, 0.0, out=delta)
            return Solution(delta, lam, it, res, res <= tol, int(free.sum()))
        key = S.tobytes()
        if key in seen:
            single = True
        seen.add(key)
        if single:
            # the single worst violator, in units of the problem's scale
            cand_d = np.where(viol_d, -delta, -np.inf)
            cand_w = np.where(viol_w, -w / scale * float(np.abs(delta).max()
                                                        or 1.0), -np.inf)
            if cand_d.max() >= cand_w.max():
                S[int(np.argmax(cand_d))] = False
            else:
                S[int(np.argmax(cand_w))] = True
        else:
            S = (S & ~viol_d) | viol_w
    np.maximum(delta, 0.0, out=delta)
    return Solution(delta, lam, max_iter, res, False, int(free.sum()))


def released_energy(P: np.ndarray, delta: np.ndarray,
                    areas: np.ndarray) -> float:
    """Energy released from the elastic field, net of frictional dissipation.

        E = (1/2) sum_i A_i (tau_0,i - g_i) delta_i

    The traction falls from its initial value to the residual strength AS slip
    accumulates, so the work released is the AVERAGE traction through the
    slip -- hence the half. On the slipping set ``tau_0 - g`` is exactly
    ``-(P^T delta)``, and locked elements contribute nothing because their
    slip is zero, so the expression below is general without needing the
    active set.

    THE FACTOR OF TWO IS THE WHOLE RISK IN THIS FUNCTION. Getting it wrong
    would corrupt every growth decision while leaving every printed number
    plausible, so it is pinned against a closed form rather than argued: for a
    penny-shaped crack of radius ``a`` under uniform drop ``d_tau``,

        E = (8/7) d_tau^2 a^3 / mu

    which ``selfcheck``'s crack clause checks at three radii, and the
    derivative against Irwin by the compliance method.

    MEASURED BIAS, AND WHY IT DOES NOT MATTER FOR GROWTH. The value comes out
    1.3 % high and -- the part that counts -- 1.3 % high at EVERY radius, and
    the compliance-derived dE/dA carries the same 1.3 %. The bias is a pure
    multiplicative scale factor shared between the energy and its derivative,
    so it cancels in the comparison between candidate growth configurations
    that work optimization actually makes. It would matter only against an
    absolute measured fracture energy.

    NOT PROVIDED: a full W_ext / W_internal / W_friction split. That
    bookkeeping has not been validated against anything, and shipping an
    unchecked energy budget is how a factor of two survives.
    """
    d = np.asarray(delta, float)
    a = np.asarray(areas, float)
    return -0.5 * float((a * (P.T @ d) * d).sum())


def energy_split(P: np.ndarray, Nrm: np.ndarray, mu: np.ndarray,
                 delta: np.ndarray, areas: np.ndarray):
    """``(E_shear, E_clamp)`` per element, summing to :func:`released_energy`.

    THE SPLIT IS ALONG THE OPERATOR, since ``P_eff = P + mu N`` makes
    ``E = -1/2 sum A_i (P_eff^T d)_i d_i`` separate exactly into a shear part
    and a clamping part. The two answer different questions: the clamping
    part is the coupling entering the functional directly, the shear part is
    the coupling changing the SOLUTION and that feeding back.

    ``P`` MUST BE THE BARE SHEAR OPERATOR, not ``P_eff``. ``P_eff`` already
    contains ``mu N``, so passing it counts the clamping twice and the halves
    then cancel to almost nothing -- a reconstructed asymmetry came out
    0.00 % where the measurement it had to reproduce was -11.9 %. That was
    caught only because it disagreed with a number already trusted, which is
    why the gate pins the SUM identity rather than the pieces.

    A PER-ELEMENT TERM IS BOOKKEEPING AND NOT A LOCAL ENERGY: the quadratic
    form is not diagonal, so one entry means nothing by itself. A partition
    of the sum is exact, and that is all attribution needs.

    WHAT IT SHOWED on the stepping asymmetry. The clamping term is purely
    ANTISYMMETRIC in the stepping sense -- ``Nrm`` is odd there while the slip
    is even -- and it OPPOSES the net asymmetry one for one, while the shear
    term carries the answer at twice the size:

        odd(E_clamp)/odd(E) = -1.0051    odd(E_shear)/odd(E) = +2.0052

    each to a spread of 0.016 across a sixfold range of offset AND through
    the reversal, where only ``shear + clamp = total`` is forced. So the
    asymmetry is a GLOBAL SLIP REDISTRIBUTION rather than a tip being
    clamped, which is why the tip's own ``sigma_n`` fails to predict it; by
    region it splits 50/32/18 between the far crack, the near crack and the
    advancing front, constant to 0.010 in the far crack's share.
    """
    d = np.asarray(delta, float)
    a = np.asarray(areas, float)
    m = np.asarray(mu, float)
    return (-0.5 * a * (P.T @ d) * d,
            -0.5 * a * m * (Nrm.T @ d) * d)


def coulomb_coupled(P: np.ndarray, Nrm: np.ndarray, fric):
    """``(P_eff, d, g)`` for the LCP with ``sigma_n`` COUPLED to slip.

    EVERYTHING ELSE IN THIS MODULE IS UNCHANGED, which is the point: letting
    slip change its own clamping keeps the problem an LCP rather than making
    it nonlinear. With Coulomb strength ``mu (sigma_eff_0 + lam sigma_n_load
    - N^T delta)`` against shear ``lam tau_load_mag + P^T delta``, the margin
    is

        w = mu sigma_eff_0 + lam (mu sigma_n_load - tau_load_mag)
            - (P + mu N)^T delta

    so :func:`solve`, :func:`released_energy` and the growth machinery all run
    on ``P_eff`` with no change at all. ``released_energy`` in particular
    stays correct because complementarity makes the effective stress drop on
    the slipping set exactly ``-(P_eff^T delta)``, the form it already
    assumes.

    ``mu`` MULTIPLIES ALONG THE RECEIVER AXIS, not the source axis: it is the
    receiver's own friction that converts a normal-stress change into a
    strength change, and ``P`` is source-major, so the broadcast is over
    axis 1. With a uniform ``mu`` the distinction is invisible, which is
    exactly why it is written down here.

    ``Nrm`` is ``kernel.project_normal``'s output and TENSION POSITIVE, so it
    enters with a PLUS sign; see that function for how the convention is
    pinned. Getting it backwards swaps which side of a step is clamped.

    WHAT THIS BUYS, measured on two offset cracks: the stepping asymmetry in
    the energy available for linkage, which is EXACTLY ZERO without the
    coupling. The shear transferred between the two is even in the stepping
    sense and the normal traction odd -- a mirror symmetry of the elastic
    problem, holding to 5e-07 and 6e-06 relative -- so a frictionless
    criterion cannot tell left- from right-stepping at all, and the measured
    asymmetry is linear in ``mu`` to 0.11 % over a six-fold range.

    UNIQUENESS IS NOT AUTOMATIC ANY MORE. ``solve``'s guarantee needs
    ``sym(-P_eff)`` positive definite, and the coupling can in principle
    destroy it -- that is the bifurcation this study's own notes flag as the
    hazard of letting the operator depend on the solution. On coplanar
    geometry it cannot: in-plane slip induces no normal traction on a
    coplanar element, so the self-term is 8e-24 and the coupling lives
    entirely between non-parallel elements. Measured on the two offset cracks
    the gate builds, ``min eig sym(-P_eff)`` moves 106.15 -> 103.04 -> 99.77
    across ``mu`` = 0, 0.3, 0.6 -- it falls, so it is not free, but it stays
    far from zero. On a network of mixed orientations this is an OPEN
    question, so callers check it rather than assume it.
    """
    mu = np.asarray(fric.mu, float)
    return (P + mu[None, :] * Nrm,
            fric.tau_load_mag - mu * fric.sigma_n_load,
            fric.strength())


def load_feedback(slip_xyz: np.ndarray, normals: np.ndarray,
                  areas: np.ndarray, *, mu: float,
                  volume: float) -> tuple[np.ndarray, np.ndarray]:
    """``(dP, dN)``: the remote load relaxing as slip accumulates.

    EVERY RESULT OF THIS STUDY SO FAR ASSUMES AN INFINITE RESERVOIR. The
    remote stress is a fixed tensor, so a crack's own slip never reduces what
    drives it, and the energy release rate grows without bound with length:
    measured, the gain at a disc's front beats the same patch in the far field
    by 4.67, 9.34 and 13.42 at radii 3, 6 and 9 h -- linear in length, because
    ``G ~ d_tau^2 L / mu`` while an isolated patch is fixed forever. The
    largest crack therefore wins by a widening margin and NOTHING ELSE CAN
    EVER START. That is a statement about the boundary condition, not about
    rock, and this function is the boundary condition made finite.

    Hold the far-field STRAIN fixed over a volume ``V`` instead of the stress.
    The slip's average inelastic strain is ``eps_p = (1/V) sum_i A_i d_i m_i``
    with ``m_i = sym(s_i (x) n_i)``, the stress falls by ``C : eps_p``, and
    resolving that back onto each element gives a correction that is LINEAR in
    slip -- so the problem stays an LCP and every caller is unchanged.

    ``C : m = 2 mu m`` EXACTLY, with no stiffness tensor anywhere, because
    slip is tangential and so ``tr(m) = s . n = 0``. The lambda term of an
    isotropic ``C`` multiplies that trace and vanishes. The caller's slip
    directions must therefore be tangential, which the gate checks rather than
    trusts: a normal component would silently make the 2 mu wrong.

        dP[i, j] = -(2 mu A_i / V) (m_i : m_j)
        dN[i, j] = -(2 mu A_i / V) (m_i : n_j (x) n_j)

    with ``m_i : m_j = ((s_i.s_j)(n_i.n_j) + (s_i.n_j)(n_i.s_j)) / 2`` and
    ``m_i : n_j (x) n_j = (n_j.s_i)(n_j.n_i)``, so the whole thing is four
    Gram matrices. ``dN`` is TENSION POSITIVE to match
    :func:`kernel.project_normal`, so ``coulomb_coupled`` composes it with the
    same plus sign and nothing about the convention changes.

    THE RANK IS AT MOST 5, not 6: the ``m_i`` span the traceless symmetric
    tensors. So this is a small correction to a dense operator however many
    elements there are, and the single-element case has a closed form the gate
    uses -- ``dP = -mu A / V``, since ``m : m = 1/2`` for a unit tangential
    slip.

    ``V -> inf`` RECOVERS THE INFINITE RESERVOIR EXACTLY, which is the free
    gate: it is the model every earlier commit measured, so the correction
    must vanish and not merely become small.
    """
    s = np.asarray(slip_xyz, float)
    n = np.asarray(normals, float)
    a = np.asarray(areas, float)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError(f"volume must be finite and positive, got {volume!r}")
    ss, nn = s @ s.T, n @ n.T
    sn = s @ n.T                       # sn[i, j] = s_i . n_j
    gm = 0.5 * (ss * nn + sn * sn.T)   # m_i : m_j
    hm = sn * nn                       # m_i : n_j (x) n_j = (n_j.s_i)(n_j.n_i)
    k = -2.0 * mu / volume * a[:, None]
    return k * gm, k * hm


def critically_stressed(fric, lam: float,
                        include: np.ndarray | None = None) -> np.ndarray:
    """The no-interaction null: elements failing under the FAR FIELD alone.

    This is the standard screening of reservoir geomechanics -- resolve
    Coulomb on each fracture independently against the regional stress and
    call the ones past it critically stressed (Barton, Zoback & Moos 1995) --
    and it is what the static solve has to beat. It costs nothing: no matrix,
    no solve. The study's central measurement is the DIFFERENCE between this
    and ``solve``, which is what elastic interaction adds over the screening
    that is already routine practice.
    """
    out = lam * (fric.tau_load_mag - fric.mu * fric.sigma_n_load) >= \
        fric.strength()
    return out if include is None else (out & include)


def elements_of(net, fault_ids) -> np.ndarray:
    """Boolean mask of the elements belonging to ``fault_ids``."""
    want = np.zeros(int(net.fault_id.max()) + 1, bool)
    want[np.asarray(fault_ids, dtype=np.int64)] = True
    return want[net.fault_id]


def influence(base: Solution, perturbed: Solution, shared: np.ndarray, *,
              rel: float = 1.0e-6) -> dict:
    """What changed among elements present in BOTH solutions.

    The sign matters and is reported separately, because a fault does not only
    switch neighbours on: its own slip relieves stress on planes in its
    shadow, so removing it can ACTIVATE them. Lumping the two into one "number
    changed" would hide exactly the asymmetry worth measuring.

    ``rel`` THRESHOLDS THE SLIP COUNTS, and it is not cosmetic. Two solves of
    the same problem differ everywhere at float level, and counting the sign
    of those differences reported 6815 "up" and 4005 "down" for eight
    different faults -- the identical pair of numbers each time, which is what
    exposed it. A change is counted only when it exceeds ``rel`` times the
    largest slip in the base solution, so the counts mean "moved" rather than
    "was recomputed".
    """
    a0 = base.active & shared
    a1 = perturbed.active & shared
    both = a0 & a1
    ds = perturbed.delta - base.delta
    scale = float(np.abs(base.delta).max())
    cut = rel * scale
    big = np.abs(ds) > cut
    return {"n_shared": int(shared.sum()),
            "n_base": int(a0.sum()), "n_perturbed": int(a1.sum()),
            "switched_on": int((a1 & ~a0).sum()),
            "switched_off": int((a0 & ~a1).sum()),
            "slip_up": int((big & both & (ds > 0)).sum()),
            "slip_down": int((big & both & (ds < 0)).sum()),
            "max_abs_dslip_km": float(np.abs(ds[shared]).max())
            if shared.any() else 0.0,
            "cut_km": cut,
            "moment_ratio": (float(perturbed.delta[shared].sum()
                                   / base.delta[shared].sum())
                             if base.delta[shared].sum() > 0 else float("nan"))}
