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

MOLLIFICATION IS WHAT MAKES THIS WELL POSED, rather than being a numerical
convenience. The uniqueness theorem needs a finite, correctly signed diagonal
``P_ii`` -- how much an element's own slip relieves its own shear. For a
classical dislocation element that number DOES NOT EXIST: the traction kernel
is hypersingular on the element's own surface, so one gets a Hadamard finite
part, a collocation-dependent value, or a divergence. Mollification smears the
slip over a width ``eps`` and the self-interaction becomes a convergent
integral -- and ``eps`` is physical rather than arbitrary, a fault being a
gouge zone of finite thickness, so ``P_ii`` is the self-stiffness of a slip
patch of that width. Without it the theorem has no matrix to act on.

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
