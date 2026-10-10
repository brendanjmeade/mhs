"""Quasi-static Coulomb loading of a fault network. No time, no dynamics.

``lambda`` is a LOAD PARAMETER, not time. Between events the state is an
equilibrium: every element is locked, with its shear stress strictly below its
static strength. Nothing propagates, nothing has a velocity, and the only
ordering in the output is by ``lambda``.

THE MODEL IS STICK-SLIP WITH RE-LOCKING, and the two alternatives were both
tried and both fail for stated reasons.

* ONE friction coefficient and no weakening gives slip CONTINUOUS in the load.
  An element reaching ``F = 0`` starts slipping at zero rate, so no event has a
  finite size and there are no avalanches to count. The first version of this
  module did exactly that: it advanced the load to the first failure and then
  relaxed zero slip, forever.
* TWO coefficients but with failed elements left at yield means those elements
  must CREEP as the load rises to stay there. Slip is then not constant between
  events, so the load at which the next element fails is no longer available in
  closed form -- it needs a parametric complementarity solve,
  ``ds_A/dlam = -(P_AA^T)^-1 d_A``, at ``O(|A|^3)`` per event.

So: an element is locked unless it is failing. It fails when its shear reaches
the STATIC strength ``mu_s * sigma_eff``, slips by exactly the amount that
brings its shear down to the DYNAMIC strength ``mu_d * sigma_eff``, and locks
again. Slip is therefore piecewise constant in the load, the next failure is a
closed-form ``min``, events are finite, and an element can fail many times.
Cumulative slip only grows.

EVENT-DRIVEN LOADING, WITH NO LOAD STEP. A fixed ``d_lam`` would merge distinct
events and the size distribution would become a measurement of ``d_lam``.
Because slip is constant between events, the load at which the next element
reaches static strength is one ``min`` over elements. There is no step size to
choose and none to report.

THE END STATE OF AN EVENT IS PATH DEPENDENT, AND THE MEASUREMENT SAYS BY HOW
MUCH. A triggered element slips to exactly its dynamic strength, but a later
element in the same sweep can push it back up, so elements finish an event
somewhere in ``[g_dynamic, g_static)`` depending on the order they were
visited. There is therefore NO unique answer to check a cascade solver against,
and a direct complementarity solve would be checking a different model. What can
be done is to vary the update rule and measure the spread. Over a 254-element
network to four times the first-failure load:

    rule              events   mean size   mean M0    total slip   active set
    sequential          3634       2.475   8.57e15        1.9497          198
    random order        3656       2.469   -0.6 %        +0.07 %          198
    synchronous         3886       2.367   -6.4 %        +0.10 %          198
      (Jacobi)         +6.9 %     -4.3 %

So: EVENT PARTITIONING carries a ~5-7 % solver dependence between Gauss-Seidel
and Jacobi, and under 1 % between element orderings. CUMULATIVE quantities do
not: total slip is invariant to 0.1 %, and the ACTIVE SET -- the thing the
dimension is measured on -- came out BIT IDENTICAL under all four rules
(Jaccard 1.0000, symmetric difference 0, slip correlation 0.9999).

That is the useful division. ``D_active`` is rule independent and can be quoted
without a solver caveat; an avalanche-size exponent cannot, and carries the
5-7 % as part of its uncertainty.

AVALANCHE STATISTICS COME FROM DIFFERENCING TWO EQUILIBRIA, NEVER FROM THE
ITERATION PATH. The relaxation below visits elements in sweeps, and it is
tempting to call a sweep a "generation" and the sequence a cascade structure.
That would report a solver artefact as physics: the sweep count and ordering
depend on whether the relaxation is Jacobi or Gauss-Seidel and on the order
elements are visited, none of which is in the problem. What IS physical is the
difference between the equilibrium before a load increment and the one after:
which elements slipped, by how much, and what moment that released. ``Event``
carries exactly that, and the sweep count sits beside it labelled as a solver
diagnostic.

THE RELAXATION IS GAUSS-SEIDEL on single columns, which is the cheap solver
here for a structural reason: returning one element to its dynamic level is a
single read of that element's own column, and ``kernel`` stores the matrix
source-major precisely so the read is contiguous. One relaxation step is
``O(n)`` and no matrix-vector product appears anywhere.

RUNAWAY IS POSSIBLE AND IS REPORTED. If the stress a failing element transfers
exceeds what it shed, the cascade does not terminate. That is a property of the
model at strong coupling, not a bug, so the sweep cap ends the run with a
statement rather than a tuned answer.
"""
from __future__ import annotations

import dataclasses

import numpy as np

#: Residual on the yield function, GPa, at which equilibrium is accepted.
#: 1e-9 GPa is 1 Pa against strengths of order 0.1 GPa -- 1e-8 relative, far
#: below any physical meaning and far above float64 noise on these sums.
F_TOL = 1.0e-9

#: Relaxation sweeps before an event is declared non-convergent. A terminating
#: cascade takes a handful; hitting this means the cascade is runaway and the
#: right response is to say so.
MAX_SWEEPS = 20_000

#: Outer iterations for the fully coupled variant, where the slip direction and
#: the normal stress both move. Not guaranteed to converge -- quasi-static
#: Coulomb friction with slip-dependent normal stress can fail to have a unique
#: solution at all -- so failures are reported, not tuned away.
MAX_OUTER = 100

#: Relative slip change below which the coupled outer loop is converged.
OUTER_TOL = 1.0e-7

#: GPa * km^3 -> N m. 1 GPa = 1e9 Pa, 1 km^3 = 1e9 m^3.
MOMENT_TO_NM = 1.0e18


@dataclasses.dataclass(frozen=True)
class Event:
    """One load increment, as the difference between two equilibria."""

    index: int
    lam: float                  # load at which this event occurred
    lam_increment: float        # how far the load advanced to trigger it
    n_slipped: int              # distinct elements that slipped
    n_faults: int               # distinct faults involved
    moment_Nm: float            # sum G A ds over the event
    max_slip_km: float
    total_slip_km: float
    n_ever: int                 # cumulative elements that have ever slipped
    sweeps: int                 # SOLVER DIAGNOSTIC, not a physical stage count
    converged: bool


@dataclasses.dataclass
class State:
    """Mutable solver state. ``q`` is maintained incrementally, never rebuilt."""

    s: np.ndarray               # (n,) cumulative slip along each slip_dir
    q: np.ndarray               # (n,) the interaction term, = P^T s
    lam: float

    def shear(self, d: np.ndarray) -> np.ndarray:
        """Driving shear in the yield function's units."""
        return self.lam * d + self.q


def coefficients(fric) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(d, g_static, g_dynamic)``.

    ``d`` is the loading rate of the yield function and carries the normal
    stress the REGIONAL LOAD resolves onto each plane -- orientation dependent,
    therefore geometric, therefore part of the reference variant. What the
    reference variant leaves out is the normal stress change caused by SLIP,
    which is the part that can cost uniqueness.
    """
    d = fric.tau_load_mag - fric.mu * fric.sigma_n_load
    return d, fric.strength(), fric.strength_dynamic()


def next_failure(state: State, d: np.ndarray,
                 g_s: np.ndarray) -> tuple[float, int]:
    """``(lam_next, element)`` for the next element to reach static strength.

    Closed form, because slip is constant between events. Elements with
    ``d <= 0`` are never driven to failure by this load and are excluded, so a
    network where nothing further can fail is reported as finished rather than
    looping.
    """
    cand = d > 0.0
    if not np.any(cand):
        return np.inf, -1
    lam_i = np.full(d.shape, np.inf)
    lam_i[cand] = (g_s[cand] - state.q[cand]) / d[cand]
    # Only forward loading counts: an element already above its static strength
    # would give a lam below the current one, and that is a state the previous
    # event should have relaxed.
    lam_i[lam_i < state.lam] = np.inf
    if not np.any(np.isfinite(lam_i)):
        return np.inf, -1
    j = int(np.argmin(lam_i))
    return float(lam_i[j]), j


def relax(state: State, P: np.ndarray, d: np.ndarray, g_s: np.ndarray,
          g_d: np.ndarray, *, tol: float = F_TOL,
          max_sweeps: int = MAX_SWEEPS, order: str = "sequential",
          rng=None) -> tuple[int, bool]:
    """Run the cascade to the next equilibrium. Returns ``(sweeps, ok)``.

    Any element whose shear has reached its STATIC strength slips by exactly
    the amount that returns it to its DYNAMIC strength, and every other
    element's stress is updated by one contiguous read of that element's
    column. Slip is only ever added, so cumulative slip is monotone by
    construction.

    ``order`` selects the update rule, which exists so the path dependence in
    the module docstring can be measured rather than assumed away:
    ``sequential`` and ``random`` are Gauss-Seidel (each element sees the
    others' updates immediately), ``sync`` is Jacobi (all triggered elements
    slip on the same stress state). ``sync`` changes event counts by ~7 % and
    leaves the active set identical.
    """
    n = state.s.shape[0]
    inv_diag = -1.0 / P[np.arange(n), np.arange(n)]      # P_ii < 0, so > 0
    if order not in ("sequential", "random", "sync"):
        raise ValueError(f"order must be 'sequential', 'random' or 'sync', "
                         f"got {order!r}")
    if order == "random" and rng is None:
        rng = np.random.default_rng(0)
    for sweep in range(1, max_sweeps + 1):
        trig = np.flatnonzero(state.shear(d) >= g_s - tol)
        if trig.size == 0:
            return sweep - 1, True
        if order == "sync":
            f = state.lam * d[trig] + state.q[trig] - g_d[trig]
            delta = f * inv_diag[trig]
            state.s[trig] += delta
            state.q += (P[trig] * delta[:, None]).sum(axis=0)
            continue
        seq = trig if order == "sequential" else rng.permutation(trig)
        for i in seq:
            # Re-read: an earlier element in this sweep may have changed it.
            f = state.lam * d[i] + state.q[i] - g_d[i]
            if f <= tol:
                continue
            delta = f * inv_diag[i]
            state.s[i] += delta
            state.q += P[i] * delta             # the contiguous column read
    return max_sweeps, False


def equilibrium_residual(state: State, d: np.ndarray,
                         g_s: np.ndarray) -> float:
    """``max(tau - g_static)``: must be <= 0 at an equilibrium."""
    return float((state.shear(d) - g_s).max())




def run_history(net, fric, P, *, shear_modulus: float = 30.0,
                lam_max: float | None = None, max_events: int = 200_000,
                order: str = "sequential", seed: int = 0,
                verbose: bool = False) -> tuple[State, list[Event]]:
    """Load the network quasi-statically, one ``Event`` per load increment.

    Monotonic loading, cumulative slip, no healing of the slip itself: every
    element that has ever slipped keeps that slip, and may slip again when the
    load rebuilds its shear to static strength. That is the experiment as
    posed -- all faults locked at the start, patches activating and interacting
    as the load rises -- so ``D_active`` is a function of ``lambda`` and not one
    number.
    """
    n = net.n_elements
    d, g_s, g_d = coefficients(fric)
    areas = net.areas()
    state = State(s=np.zeros(n), q=np.zeros(n), lam=0.0)
    events: list[Event] = []
    ever = np.zeros(n, bool)

    for k in range(1, max_events + 1):
        lam_next, j = next_failure(state, d, g_s)
        if not np.isfinite(lam_next):
            break
        if lam_max is not None and lam_next > lam_max:
            break
        inc = lam_next - state.lam
        state.lam = lam_next
        before = state.s.copy()
        sweeps, ok = relax(state, P, d, g_s, g_d, order=order,
                           rng=np.random.default_rng(seed))
        ds = state.s - before
        moved = ds > 0.0
        ever |= moved
        ev = Event(index=k, lam=lam_next, lam_increment=inc,
                   n_slipped=int(moved.sum()),
                   n_faults=int(np.unique(net.fault_id[moved]).size)
                   if moved.any() else 0,
                   moment_Nm=float(shear_modulus
                                   * np.sum(areas[moved] * ds[moved])
                                   * MOMENT_TO_NM),
                   max_slip_km=float(ds.max() if moved.any() else 0.0),
                   total_slip_km=float(ds[moved].sum()),
                   n_ever=int(ever.sum()), sweeps=sweeps, converged=bool(ok))
        events.append(ev)
        if verbose and (k <= 5 or k % 500 == 0):
            print(f"    event {k:6d}  lam {lam_next:.6f}  "
                  f"slipped {ev.n_slipped:5d}  faults {ev.n_faults:4d}  "
                  f"M0 {ev.moment_Nm:.3e}  sweeps {sweeps}")
        if not ok:
            print(f"    RUNAWAY at event {k}, lam {lam_next:.6f}: "
                  f"{MAX_SWEEPS} sweeps without reaching equilibrium. The "
                  f"cascade transfers more stress than it sheds at this "
                  f"coupling. Reported, not tuned.")
            break
        if ev.n_slipped == 0:
            raise RuntimeError(
                f"event {k} at lam {lam_next:.6g} relaxed zero slip, which "
                f"means the trigger and the relaxation disagree about what "
                f"failure is. With a strength drop this cannot happen: an "
                f"element at static strength is above dynamic strength by "
                f"(mu_s - mu_d) sigma_eff > 0.")
    return state, events


# ------------------------------------------------- the fully coupled variant --
def run_history_coupled(net, fric, K, *, shear_modulus: float = 30.0,
                        lam_max: float | None = None,
                        max_events: int = 200_000, verbose: bool = False):
    """As :func:`run_history`, but normal stress and slip direction both move.

    Each event is solved by an OUTER loop: hold the direction and the normal
    stress, run the reference cascade, recompute both from the new slip, repeat.
    This is the physical version and it is NOT guaranteed to converge, so every
    event carries a flag and a persistent failure ends the run with a statement.

    Returns ``(s_vec, events, n_outer_failures)``.
    """
    from kernel import RECEIVER, project                 # local: avoids a cycle

    n = net.n_elements
    areas = net.areas()
    i_norm = RECEIVER.index("normal")
    s_vec = np.zeros((n, 2))
    lam = 0.0
    events: list[Event] = []
    ever = np.zeros(n, bool)
    failures = 0

    def resolve(sv):
        tau = np.einsum("jbai,jb->ia", K[:, :, :2, :], sv)
        dsig = -np.einsum("jbi,jb->i", K[:, :, i_norm, :], sv)
        return tau, dsig

    d_lin = fric.tau_load_mag - fric.mu * fric.sigma_n_load
    for k in range(1, max_events + 1):
        tau_i, dsig = resolve(s_vec)
        tau_tot = lam * fric.tau_load + tau_i
        sig = fric.sigma_eff_0 + lam * fric.sigma_n_load + dsig
        mag = np.linalg.norm(tau_tot, axis=1)
        cand = d_lin > 0.0
        if not np.any(cand):
            break
        lam_i = np.full(n, np.inf)
        lam_i[cand] = lam + (fric.mu[cand] * sig[cand] - mag[cand]) / d_lin[cand]
        lam_i[lam_i < lam] = np.inf
        if not np.any(np.isfinite(lam_i)):
            break
        lam_next = float(np.nanmin(lam_i))
        if lam_max is not None and lam_next > lam_max:
            break
        inc, lam = lam_next - lam, lam_next
        before = s_vec.copy()

        ok = False
        outer = 0
        for outer in range(1, MAX_OUTER + 1):
            tau_i, dsig = resolve(s_vec)
            tau_tot = lam * fric.tau_load + tau_i
            mag = np.linalg.norm(tau_tot, axis=1)
            safe = np.where(mag > 0.0, mag, 1.0)
            dirn = tau_tot / safe[:, None]
            sig = fric.sigma_eff_0 + lam * fric.sigma_n_load + dsig
            if not np.all(sig > 0.0):
                print(f"    event {k}: effective normal stress went "
                      f"non-positive (min {sig.min():.4e} GPa). The fault has "
                      f"opened and Coulomb friction no longer describes it.")
                return s_vec, events, failures + 1
            P = project(K, dirn, dirn)
            scal = np.maximum(np.einsum("ia,ia->i", s_vec, dirn), 0.0)
            st = State(s=scal.copy(), q=P.T @ scal, lam=lam)
            d_o = (np.einsum("ia,ia->i", fric.tau_load, dirn)
                   - fric.mu * fric.sigma_n_load)
            _, inner_ok = relax(st, P, d_o, fric.mu * sig,
                                fric.mu_dynamic * sig)
            new = dirn * st.s[:, None]
            move = float(np.linalg.norm(new - s_vec))
            s_vec = new
            if inner_ok and move <= OUTER_TOL * max(
                    1.0, float(np.linalg.norm(s_vec))):
                ok = True
                break
        if not ok:
            failures += 1

        ds = np.linalg.norm(s_vec - before, axis=1)
        moved = ds > 0.0
        ever |= moved
        events.append(Event(
            index=k, lam=lam, lam_increment=inc, n_slipped=int(moved.sum()),
            n_faults=int(np.unique(net.fault_id[moved]).size)
            if moved.any() else 0,
            moment_Nm=float(shear_modulus * np.sum(areas[moved] * ds[moved])
                            * MOMENT_TO_NM),
            max_slip_km=float(ds.max() if moved.any() else 0.0),
            total_slip_km=float(ds[moved].sum()),
            n_ever=int(ever.sum()), sweeps=outer, converged=ok))
        if verbose and (k <= 5 or k % 500 == 0):
            print(f"    event {k:6d}  lam {lam:.6f}  slipped "
                  f"{int(moved.sum()):5d}  outer {outer}  ok {ok}")
        if failures > 10:
            print(f"    COUPLED VARIANT NOT CONVERGING: {failures} events hit "
                  f"{MAX_OUTER} outer iterations. Stopping and reporting, "
                  f"because a tuned answer here would be a statement about the "
                  f"solver rather than about the rock.")
            break
    return s_vec, events, failures
