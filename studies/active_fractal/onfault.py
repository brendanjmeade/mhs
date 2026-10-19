"""On-fault stress on a system of three surface-breaking rough faults, one
value per element, three kernels, for the site.

    python onfault.py solve   [--out out/onfault_state.npz] [--workers 12]
    python onfault.py field   --state out/onfault_state.npz [--k 1] [--probe]
    python onfault.py payload --field out/onfault_field.npz
    python onfault.py thresholds          # the failure-load shift, a figure

THE THIRD KERNEL is the mollified one again, with CONTINUOUS quadratic slip
(``nodal``): one slip vector per shared node of each fault, C0 across every
edge, the condition enforced AT the nodes and the rim held at zero. Its
per-element maps are read at the centroids from the nodal slip -- a readout
between collocation points, which is exactly what the staircase of a P0 slip
cannot do and a continuous one can -- and its per-element slip is the mean of
the nodal slip over the element. The same friction (resolved on each node's
own plane), the same load, the same LCP -- solved by ``static.solve_newton``
rather than by projected Gauss-Seidel, which diverges on this operator: its
symmetric part is indefinite on this system (37 negative modes at eps =
0.1 h, four at 0.2 h, against +4.8 on the selfcheck's a/h = 8 discs; where
they live is measured under ``EPS_CONTINUOUS``).

THE SYSTEM. Three rough discs cut by the free surface, meshed by ``system``:
rims sampled on the circle, traces exactly on z = 0, and C a splay whose
lower edge lies on A along a polyline the two tilings share bit for bit. B is
a parallel ramp in A's hanging wall. The roughness is the published
one-wavelength field, tapered to zero at every rim, trace and junction line;
``system`` gates all of it.

WHAT IS PLOTTED. One value per element, read at its centroid, which is the
collocation point. The solve is the Coulomb LCP with the slip-induced normal
stress coupled in (``static.coulomb_coupled``), so at a centroid the Coulomb
excess ``S = tau_s - mu sigma_n'`` is exactly ``-w``, the solver's own margin:
zero on every slipping element, the deficit on every locked one. The
slip-induced shear and normal-stress changes are the element's own, and so is
the slip. Both kernels go through the same friction, the same operators and
the same solver; only the matrix differs, and at the centroids, where the
classical element is finite, the two agree to about a per cent.

``cells(tris, k)`` still subdivides every facet into k^2 equal-area cells for
a dense readout between the centroids (the page shipped one at k = 10 until
2026-10-16; the machinery and the edge-distance byte are kept for it), and
k = 1 is the facet itself with its sample at the centroid.

ONE SPAWN POOL PER PROCESS, for the reason roughvol.py gives: a second spawn
pool raised after the first has closed deadlocks on this machine. ``solve``
runs the P0 assembly pool and saves, and the nodal matrix is assembled by
``onfault.py assemble`` in a subprocess of its own (cached beside the P0 one,
so a second ``solve`` finds it); ``field`` runs each pool pass -- the P0
readout and the nodal one -- in a subprocess (``--pass``) and merges them;
``payload`` only reads. The probe runs each pool configuration in a
subprocess for the same reason.

THE READOUT POOL IS PINNED IN THE PARENT, before the children import numpy.
OpenBLAS here is the OpenMP build at 16 threads, and libomp reads its thread
count when the library loads -- which in a spawned child happens before any
pin inside the worker function can run. roughvol and isosurf pin inside the
worker and measured a field readout at 1.45x on eight workers, against 8.5x
for assembly, which pins around pool creation. The probe measures both.

STUDY-ONLY DEPENDENCIES: cutde, as roughvol states, and ``triangle`` through
``system``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mhs                                                   # noqa: E402
from mhs import tdcs                                         # noqa: E402
import cascade as C                                          # noqa: E402
import friction as Fr                                        # noqa: E402
import kernel as K                                           # noqa: E402
import network                                               # noqa: E402
import nodal as Nd                                           # noqa: E402
import roughvol as R                                         # noqa: E402
import static as St                                          # noqa: E402
import system as Sy                                          # noqa: E402

MU, LAM = R.MU, R.LAM
H_KM = 0.25                      # half the published crack's h: more, smaller elements
EPS = 0.1 * H_KM
SLOPE = R.SLOPE
CORR_KM = R.CORR_OVER_H * R.H_KM # the published correlation length, in km (4)
REGIME, DROP_RATIO, MEAN_STRESS = R.REGIME, R.DROP_RATIO, R.MEAN_STRESS
SCHEMA = 4                       # 4: the A-alone experiment beside the system

#: The system. Every fault strikes along y and dips toward +x, with no
#: azimuth because the thrust load ``diag(-1, 0, 1)`` compresses along x and
#: strike-parallel faults are the regime's own symmetry. ``trace`` is where
#: the plane meets z = 0 and ``v_c`` how far down-dip the disc's centre sits,
#: less than the radius, so every circle is cut by the surface. A is the main
#: 30 deg thrust. B is a parallel ramp 3 km into A's hanging wall (its trace
#: 6 km from A's), where A's slip shadows it. C is a 50 deg splay rooted on A:
#: ``attach`` makes its lower edge the line where its plane meets A's
#: (``system.junction_line``), and its disc is centred half-way down that
#: width. 50 rather than steeper because a three-sigma facet tilt on top of
#: the dip must keep n_z above the frame gate below. Centres in km.
FAULTS = (
    dict(name="A", a=6.0, dip=30.0, seed=R.SEED, trace=(0.0, 0.0), v_c=4.0),
    dict(name="B", a=4.0, dip=30.0, seed=2, trace=(6.0, 3.0), v_c=3.0),
    dict(name="C", a=4.0, dip=50.0, seed=3, trace=(4.0, 0.0), attach="A"),
)

#: Minimum vertex-to-facet distance between faults that do NOT touch, in h.
#: 2 h = 0.5 km = 20 eps: outside every kernel's near field and inside the
#: interaction range, which is what makes a stress shadow readable as one
#: rather than as two surfaces inside each other's blobs. A and C touch along
#: the junction by construction; ``system`` gates that C stays on A's
#: hanging-wall side everywhere else and reports how close it comes.
CLEARANCE_OVER_H = 2.0

#: Floor on the vertex-order normal's n_z. The upward-normal convention is
#: decided by float noise at n_z = 0 (selfcheck [kernel]: 183 of 580 normals
#: once came back anti-aligned on an exactly vertical band), so the system is
#: kept three degrees off vertical everywhere.
MIN_NZ = 0.05

#: Cells per facet side. 1 is the facet itself, sampled at its centroid, the
#: collocation point. At k = 10 a cell side is h/10 = eps and the nearest cell
#: centroid to a facet edge sits 0.2887 h / k = 0.29 eps from it.
K_DEFAULT = 1

#: Load factors tried, as multiples of the system's first failure. The one
#: taken is the smallest at which A slips over most of its area and B, C are
#: each partly locked and partly slipping -- a picture of a system, not of
#: three independent cracks. Refused if none qualifies: then the geometry is
#: wrong, not the load. The band is wide at the bottom on purpose: the splay
#: is included BECAUSE it is misoriented (50 deg under a load that is optimal
#: at 30), so it slips only where the others load it, and a few per cent
#: active is its expected state. Measured on the ladder: 0.00, 0.01, 0.08,
#: 0.23, 0.39 at 1.05 .. 1.50, against B at 0.30 .. 0.99.
LAM_LADDER = (1.05, 1.10, 1.20, 1.35, 1.50)
LAM_RULE = dict(A_min=0.70, BC_min=0.05, BC_max=0.95)

#: Observer chunk for the mollified pass. traction_matrix is 72 B per pair, so
#: 1000 cells against ~1800 sources is 130 MB in a worker.
CHUNK = 1000
CHUNK_CUTDE = 2000

METRICS = ("S", "dsn", "dts")
EDGE_STEP = 0.02                 # eps per byte of the shipped edge distance
KERNELS = ("mollified", "classical", "continuous")
#: The passes of ``field`` that need a pool of their own.
POOL_PASSES = ("mollified", "continuous")
#: The two experiments the page shows: the three faults together, and A by
#: itself on the same mesh, the same matrix (its A-A block) and the same
#: load -- the baseline that says what B and C do to A.
EXPERIMENTS = ("system", "alone")

#: The continuous kernel's mollification width, TWICE the P0 kernels'.
#: Measured on this system at eps = 0.1 h: the nodal operator's symmetric part
#: has 37 negative eigenvalues out of 13,940 (fault A alone -13, B -34, C
#: -52, against a median of +240), smooth modes (vertex and midpoint values
#: correlate +0.99) of two kinds. Dropping the 207 nodes within 2 eps of the
#: free surface leaves 14: that band is a smeared source straddling z = 0,
#: the effect that makes the trace nodes too soft (B's band alone has 4
#: modes at -19 while its 43 trace nodes alone are definite at +17). The 14
#: that remain are broad body modes on each fault alone below 4 eps (A 6,
#: B 3, C 2, down to -13, spread over thousands of nodes), the small-eps/h
#: collocation asymmetry the full-space work found above a/h = 16. The
#: Coulomb LCP on it has no stable solution: projected Gauss-Seidel overflows
#: and the active-set Newton method thrashes with thousands of swaps per
#: iteration. At eps = 0.2 h the body modes are gone (every fault alone is
#: definite below 4 eps) and four remain: three on C's nodes within 2 eps of
#: the surface (-6.6, -3.8, -0.4) and one long-wavelength mode spanning A
#: and C at -0.85 of a median +128 -- the kind the P0 operators carry on this
#: system too (mollified -1.2, classical -2.5 of +200, one mode each, and
#: PGS converges on them regardless). The Newton solve converges in six
#: iterations. The price is the readout's: the nodal readout error is
#: 1.7 eps/a, so at this eps it is the P0 centroid readout's size on A, while
#: the staircase between the collocation points is gone either way
#: (selfcheck [nodal]). The trace nodes are the other price: a node ON the
#: free surface reads its own smeared slip, half of it above z = 0, and is too
#: SOFT (B's trace nodes slip 2.1 m in the median against 0.85 in the row
#: below and 0.94 for the classical trace row) where the P0 row 2.5 eps below
#: the surface is too stiff.
#: MEASURED AFTERWARDS on fault B alone (``nodal`` docstring): the too-soft
#: trace is the mollified half-space kernel's O(1) error within ~eps of the
#: free surface for a blob straddling it, which no form of the continuous
#: layout escapes (the energy form has it with the opposite sign, its trace
#: at half the row below), and it is not local: it moves the moment of the
#: whole fault by ~20 % (B alone: this kernel 1.07 of classical P0 at 0.2 h,
#: the energy form 0.82 at 0.1 h, and every form within 3 % of classical
#: once the same fault is buried 3 km). The P0 centroid, 2.5 eps deep, feels
#: it only as O(eps/depth), so the P0 kernels converge to classical and the
#: continuous layout does not until the kernel folds the part of the
#: smeared source above z = 0 back into the body (the mirror image plus a
#: Boussinesq surface term, now the kernel's default, ``mhs.kernels.fold``;
#: measured: the vertical strike-slip reference from 5x to 0.98 of exact,
#: B's Galerkin moment from 0.82 to 1.01 of classical -- ``nodal``
#: docstring). The page's numbers quoted here predate it; a rerun folds.
EPS_CONTINUOUS = 0.2 * H_KM

_W: dict = {}


# --------------------------------------------------------------- geometry --
from system import clearance, point_triangle_distance        # noqa: E402,F401


def frame_gate(tris: np.ndarray, tol: float = 1e-12) -> float:
    """Refuse unless cutde's own frame equals ``tdcs.slip_frame`` on every facet.

    cutde builds its (strike, dip, normal) rows from the VERTEX-ORDER normal
    with strike ``z x n``, so it agrees with tdcs exactly when every normal
    points up and no facet is horizontal (there cutde takes ``(0, +-1, 0)``
    where tdcs takes ``(1, 0, 0)``). This is the basis trap the head-to-head
    script warned about, made one line: the classical matrix below contracts
    cutde's local slip axes as if they were tdcs's, and this is what makes
    that legitimate.
    """
    from cutde.geometry import compute_efcs_to_tdcs_rotations
    err = np.abs(compute_efcs_to_tdcs_rotations(tris)
                 - tdcs.slip_frame(tris)).max(axis=(1, 2))
    worst = int(np.argmax(err))
    if err[worst] > tol:
        raise SystemExit(
            f"cutde's frame differs from tdcs's on {int((err > tol).sum())} "
            f"facets (worst {err[worst]:.3e} on facet {worst}): a horizontal "
            f"or downward-wound facet, and the classical matrix would contract "
            f"two different bases")
    return float(err.max())


def build(faults=FAULTS, h: float = H_KM, *, corr_km: float = CORR_KM,
          rms_slope: float = SLOPE) -> tuple[list[dict], network.Network]:
    """The faults of ``faults`` as one Network, through ``system.build_system``.

    Gated there: rim vertices on their curve, traces at z = 0, the junction
    shared bit for bit with the host carrying it as edges, each disc a
    manifold tiling, no vertex above the surface, n_z >= MIN_NZ, the
    clearance between faults that do not touch, and the splay on its host's
    hanging-wall side. Gated here: cutde's frame equals tdcs's on every facet,
    which is what lets one code path solve both kernels. ``eps`` is 0.1 h
    throughout, the study's rule.
    """
    records, net = Sy.build_system(faults, h, rms_slope=rms_slope, corr=corr_km,
                                   eps=0.1 * h, min_nz=MIN_NZ,
                                   clearance_over_h=CLEARANCE_OVER_H)
    frame_gate(net.tris)
    return records, net


# ---------------------------------------------------------------- solve --
def voigt_tensor(E6: np.ndarray) -> np.ndarray:
    """``(..., 6)`` cutde Voigt (xx, yy, zz, xy, xz, yz) -> ``(..., 3, 3)``."""
    e = np.zeros(E6.shape[:-1] + (3, 3))
    for k, (x, y) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2),
                                (1, 2))):
        e[..., x, y] = E6[..., k]
        e[..., y, x] = E6[..., k]
    return e


def hooke(e: np.ndarray) -> np.ndarray:
    """Stress from strain with (mu, lam), never through nu."""
    tr = np.trace(e, axis1=-2, axis2=-1)
    sig = 2.0 * MU * e
    sig[..., 0, 0] += LAM * tr
    sig[..., 1, 1] += LAM * tr
    sig[..., 2, 2] += LAM * tr
    return sig


def classical_matrix(tris: np.ndarray) -> np.ndarray:
    """cutde's interaction matrix at the centroids, in ``kernel``'s own layout.

    ``(n_src, n_b, n_a, n_rec)`` exactly as ``kernel.assemble`` stores it, so
    ``refuse_if_sign_wrong``, ``project``, ``project_normal`` and
    ``branching_ratio`` apply unchanged and the two kernels are solved by one
    code path. The source slip axis ``b`` is cutde's local (strike, dip),
    which :func:`frame_gate` has made tdcs's; the receiver row ``a`` is the
    traction on receiver ``i``'s upward normal resolved on its own frame.

    At its own centroid a classical element is finite -- that is the one place
    it is -- and the diagonal comes out negative, which the same gate as the
    mollified matrix checks. Not cached: six seconds.
    """
    import cutde.halfspace as hs
    frame_gate(tris)
    n = tris.shape[0]
    nu = mhs.Material(mu=MU, lam=LAM).nu_if_you_must
    cen = mhs.collocation_points(tris, 0)
    frame = tdcs.slip_frame(tris)
    E = np.asarray(hs.strain_matrix(cen.copy(), tris.copy(), nu))  # (r,6,s,3)
    Kc = np.empty((n, K.N_B, K.N_A, n))
    for b in range(K.N_B):
        sig = hooke(voigt_tensor(np.transpose(E[:, :, :, b], (0, 2, 1))))
        t = np.einsum("rsab,rb->rsa", sig, frame[:, 2, :])      # (r, s, 3)
        for a in range(K.N_A):
            Kc[:, b, a, :] = np.einsum("rsa,ra->sr", t, frame[:, a, :])
    K.refuse_if_sign_wrong(Kc)
    return np.ascontiguousarray(Kc)


def operators(Kmat: np.ndarray, fric) -> dict:
    """``P``, ``Nrm`` and the coupled LCP operator, with its definiteness."""
    P = K.project(Kmat, fric.slip_dir, fric.slip_dir)
    Nrm = K.project_normal(Kmat, fric.slip_dir)
    P_eff, d, g = St.coulomb_coupled(P, Nrm, fric)
    mineig = float(np.linalg.eigvalsh(-0.5 * (P_eff + P_eff.T)).min())
    return dict(P=P, Nrm=Nrm, P_eff=P_eff, d=d, g=g, mineig=mineig,
                B=float(K.branching_ratio(P).mean()))


def nodal_matrix(net, workers: int, *, verbose: bool = True):
    """The continuous matrix from the cache, assembled in a SUBPROCESS when
    it is not there yet: ``solve`` already owns this process's one pool."""
    sn = Nd.table(net)
    key = Nd.cache_key(net.tris, net.fault_id, MU, LAM, EPS_CONTINUOUS)
    cache = pathlib.Path(__file__).resolve().parent / "cache"
    if not (cache / f"N_{key}.npy").exists():
        if verbose:
            print(f"  nodal matrix: assembling in a subprocess on {workers} "
                  f"workers ({sn.n} nodes, eps {EPS_CONTINUOUS} km)", flush=True)
        subprocess.run([sys.executable, __file__, "assemble", "--workers",
                        str(workers)], check=True)
    return Nd.assemble(net, sn, mu=MU, lam=LAM, eps=EPS_CONTINUOUS,
                       workers=workers, verbose=verbose)


def cmd_assemble(a) -> int:
    """The nodal matrix into the cache, in its own process (its own pool)."""
    _, net = build()
    Nd.assemble(net, mu=MU, lam=LAM, eps=EPS_CONTINUOUS, workers=a.workers,
                verbose=True)
    return 0


def fractions(delta: np.ndarray, fault_id: np.ndarray) -> np.ndarray:
    return np.array([float((delta[fault_id == i] > 0).mean())
                     for i in range(fault_id.max() + 1)])


def cmd_solve(a) -> int:
    discs, net = build()
    n, fid = net.n_elements, net.fault_id
    print(f"  system: " + ", ".join(
        f"{f['name']} a={f['a']} dip={f['dip']} n={f['n']} "
        f"tilt {f['tilt_rms_deg']:.1f} deg" for f in net.meta["faults"]))
    cl = net.meta["clearance_km"]
    print(f"  {n} elements, h {H_KM} km, eps {EPS} km, clearance between "
          f"separate faults {cl:.3f} km" if cl is not None else
          f"  {n} elements, h {H_KM} km, eps {EPS} km, separate faults more "
          f"than {net.meta['clearance_reach_km']:.2f} km apart")
    for name, c in net.meta["contact"].items():
        print(f"  {name} is rooted on {c['host']}: off the junction it comes "
              f"within {c['min_distance_km']:.3f} km of it")
    print(f"  min n_z {net.meta['min_nz']:.3f}, cutde frame = tdcs frame")

    fric = Fr.build_friction(net, regime=REGIME, drop_ratio=DROP_RATIO,
                             mean_stress_ratio=MEAN_STRESS)
    # the nodal matrix first, in its own process, before this one's pool
    t0 = time.time()
    Kn, sn, _ = nodal_matrix(net, a.workers)
    tn = time.time() - t0
    free = Nd.free_mask(sn)
    fric_n = Fr.build_friction(net, regime=REGIME, drop_ratio=DROP_RATIO,
                               mean_stress_ratio=MEAN_STRESS, nodes=sn)
    print(f"  continuous: {sn.n} nodes on {n} elements, {int(sn.front.sum())} "
          f"on the fronts (locked), {int(sn.trace.sum())} on the traces (free), "
          f"eps {EPS_CONTINUOUS} km")
    t0 = time.time()
    Km, info = K.assemble(net, mu=MU, lam=LAM, workers=a.workers, verbose=True)
    t1 = time.time()
    Kc = classical_matrix(net.tris)
    print(f"  classical matrix in {time.time() - t1:.1f} s "
          f"(mollified {t1 - t0:.1f} s, continuous {tn:.1f} s); all "
          f"diagonals negative")
    ops = {"mollified": operators(Km, fric), "classical": operators(Kc, fric),
           "continuous": Nd.operators(Kn, fric_n, free)}
    del Km, Kc, Kn
    for k in KERNELS:
        print(f"    {k:10s} min eig sym(-P_eff) {ops[k]['mineig']:+.4f}  "
              f"B {ops[k]['B']:.4f}"
              + (f"  (uncoupled {ops[k]['mineig_uncoupled']:+.4f}, on the free "
                 f"nodes)" if k == "continuous" else ""))

    # Coupled unless the coupling costs definiteness for ANY kernel; then all
    # fall back to the uncoupled form, so that whatever S means, it means the
    # same thing in every map.
    coupled = all(ops[k]["mineig"] > 0.0 for k in KERNELS)
    if not coupled:
        print("  coupling loses positive definiteness: falling back to the "
              "uncoupled LCP for every kernel")
        for k, fr in (("mollified", fric), ("classical", fric),
                      ("continuous", fric_n)):
            d, g, _ = C.coefficients(fr)
            ops[k]["P_eff"], ops[k]["d"], ops[k]["g"] = ops[k]["P"], d, g
    include = {"mollified": None, "classical": None, "continuous": free}
    # The nodal operator's symmetric part is indefinite on this system (the
    # near-surface band and, at eps = 0.1 h, the body modes of each fault --
    # see EPS_CONTINUOUS) and projected Gauss-Seidel diverges on it; the
    # active-set Newton method needs no
    # definiteness and agrees with PGS wherever PGS converges (selfcheck
    # [nodal]). The P0 kernels keep PGS, so their maps are the published ones.
    solver = {"mollified": St.solve, "classical": St.solve,
              "continuous": St.solve_newton}

    # The load, by rule. Walked on the mollified operator and then applied to
    # the classical one unchanged.
    lam0 = float(np.nanmin(fric.first_failure_lambda()))
    om = ops["mollified"]
    chosen, ladder, warm = None, [], None
    for fac in LAM_LADDER:
        sol = St.solve(om["P_eff"], om["d"], om["g"], fac * lam0, warm=warm)
        warm = sol.delta
        fr = fractions(sol.delta, fid)
        ladder.append(dict(factor=fac, active=fr.tolist(), sweeps=sol.sweeps,
                           converged=bool(sol.converged)))
        print(f"    lam/lam0 {fac:.2f}: active " + " ".join(
            f"{FAULTS[i]['name']} {fr[i]:.2f}" for i in range(len(FAULTS)))
            + f"  ({sol.sweeps} sweeps)")
        ok = (fr[0] >= LAM_RULE["A_min"] and all(
            LAM_RULE["BC_min"] <= fr[i] <= LAM_RULE["BC_max"]
            for i in range(1, len(FAULTS))))
        if ok and chosen is None:
            chosen = fac
    if chosen is None:
        raise SystemExit("no load on the ladder gives A >= 70 % active and B, "
                         "C each between 5 % and 95 %: the geometry is wrong, "
                         "not the load")
    lam = chosen * lam0
    if not lam < fric.max_lambda_compressive():
        raise SystemExit(f"lam {lam} is not below the compressive bound "
                         f"{fric.max_lambda_compressive()}")
    print(f"  lam = {chosen:.2f} lam0 = {lam:.5f} GPa (lam0 {lam0:.5f})")

    # Every kernel at the one load, plus each fault ALONE on the same matrix:
    # the interaction pre-flight, in seconds, before the hours.
    state: dict = dict(tris=net.tris, fault_id=fid, lam=lam, lam0=lam0,
                       ladder=json.dumps(ladder), coupled=coupled, eps=EPS,
                       h=H_KM, mu=fric.mu, sigma_eff_0=fric.sigma_eff_0,
                       tau_load_mag=fric.tau_load_mag,
                       sigma_n_load=fric.sigma_n_load, slip_dir=fric.slip_dir,
                       meta=json.dumps(net.meta, default=str),
                       eps_continuous=EPS_CONTINUOUS,
                       node_coords=sn.coords, node_group=sn.group,
                       node_free=free, node_ids=sn.ids,
                       node_tau_load_mag=fric_n.tau_load_mag,
                       node_sigma_n_load=fric_n.sigma_n_load,
                       node_slip_dir=fric_n.slip_dir)
    table = {}
    areas = net.areas()
    for k in KERNELS:
        o = ops[k]
        nodal = k == "continuous"
        inc = include[k]
        sol = solver[k](o["P_eff"], o["d"], o["g"], lam, include=inc)
        if not sol.converged:
            raise SystemExit(f"{k}: the LCP did not converge in {sol.sweeps}")
        w = o["g"] - (lam * o["d"] + o["P_eff"].T @ sol.delta)
        if nodal:
            slip_nodes = Fr.slip_cartesian(net.tris, fric_n.slip_dir, sol.delta,
                                           planes=sn.normals)
            slip = sn.element_mean(slip_nodes)              # one per ELEMENT
            delta_el = sn.element_mean(sol.delta)
            active_el = delta_el > 0.0
            weight = sn.weights * sol.delta
            group = sn.group
        else:
            slip = Fr.slip_cartesian(net.tris, fric.slip_dir, sol.delta)
            delta_el, active_el = sol.delta, sol.active
            weight = areas * sol.delta
            group = fid
        rows = []
        for i, f in enumerate(FAULTS):
            m = fid == i
            mg = group == i
            alone = solver[k](o["P_eff"], o["d"], o["g"], lam,
                              include=mg if inc is None else (mg & inc))
            others = sol.delta.copy()
            others[mg] = 0.0
            on = mg if inc is None else (mg & inc)
            shadow = float((o["P_eff"].T @ others)[on].mean())
            mom_sys = float(weight[mg].sum())
            w_alone = (sn.weights * alone.delta if nodal else areas * alone.delta)
            mom_alone = float(w_alone[mg].sum())
            rows.append(dict(
                name=f["name"],
                active_alone=float(((sn.element_mean(alone.delta) if nodal
                                     else alone.delta)[m] > 0).mean()),
                active_system=float(active_el[m].mean()),
                moment_ratio=(mom_sys / mom_alone if mom_alone > 0 else
                              float("inf") if mom_sys > 0 else 1.0),
                moment_system_km3=mom_sys, moment_alone_km3=mom_alone,
                shadow_GPa=shadow, max_slip_m=float(sol.delta[mg].max() * 1e3),
                max_slip_alone_m=float(alone.delta[mg].max() * 1e3)))
            if i == 0:
                # THE BASELINE EXPERIMENT: A alone, on the same mesh and the
                # same matrix (its A-A block), under the same load. Kept in
                # full so `field` can read its maps.
                if nodal:
                    s_al = Fr.slip_cartesian(net.tris, fric_n.slip_dir,
                                             alone.delta, planes=sn.normals)
                    state.update(slip_alone_continuous_nodes=s_al,
                                 active_alone_continuous_nodes=alone.active)
                    slip_al, act_al = sn.element_mean(s_al), sn.element_mean(alone.delta) > 0.0
                else:
                    slip_al = Fr.slip_cartesian(net.tris, fric.slip_dir, alone.delta)
                    act_al = alone.active
                state.update({f"delta_alone_{k}": alone.delta,
                              f"slip_alone_{k}": slip_al,
                              f"active_alone_{k}": act_al,
                              f"sweeps_alone_{k}": alone.sweeps,
                              f"residual_alone_{k}": alone.residual})
            print(f"    {k:10s} {f['name']}: active alone {rows[-1]['active_alone']:.2f}"
                  f" -> system {rows[-1]['active_system']:.2f}, moment x"
                  f"{rows[-1]['moment_ratio']:.3f}, shadow "
                  f"{shadow * 1e3:+.2f} MPa, max slip "
                  f"{rows[-1]['max_slip_m']:.3f} m")
        table[k] = rows
        state.update({f"delta_{k}": sol.delta, f"w_{k}": w, f"slip_{k}": slip,
                      f"active_{k}": active_el, f"mineig_{k}": o["mineig"],
                      f"B_{k}": o["B"], f"sweeps_{k}": sol.sweeps,
                      f"residual_{k}": sol.residual})
        if nodal:
            state.update(slip_continuous_nodes=slip_nodes,
                         active_continuous_nodes=sol.active,
                         mineig_continuous_uncoupled=o["mineig_uncoupled"])
    moved = any(abs(r["active_system"] - r["active_alone"]) >= 0.05
                or abs(r["moment_ratio"] - 1.0) >= 0.10
                for r in table["mollified"][1:])
    if not moved:
        raise SystemExit("neither B nor C moves by 0.05 in active fraction or "
                         "10 % in moment when the others are present: the "
                         "faults are too far apart to be a system")
    state["table"] = json.dumps(table)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **state)
    print(f"  saved {out}")
    return 0


# ---------------------------------------------------------------- field --
def cells(tris: np.ndarray, k: int, eps: float = EPS) -> dict:
    """The k^2 equal-area cells of every facet, sampled at their centroids.

    ``corners`` are the cells' barycentric corners, the same for every facet,
    so the page rebuilds the sub-mesh from ``tris`` and ``corners`` alone and
    the value it colours each cell with IS the sample, nothing interpolated.
    No sample lies on a facet side -- cutde returns NaN exactly there -- and
    the nearest one sits ``0.2887 h / k`` from an edge, so the classical
    magnitude is set by the cell spacing exactly as the grid spacing set it on
    the isosurface page. ``edge_eps`` is each cell's distance to its facet's
    nearest edge in eps, which is what lets the page mask the stripes.
    """
    from mhs.fullspace.shape import barycentric_grid, grid_triangles
    corners = barycentric_grid(k)[grid_triangles(k)]           # (k^2, 3, 3)
    bc = corners.mean(axis=1)                                  # (k^2, 3)
    n = tris.shape[0]
    pts = np.einsum("cb,fbx->fcx", bc, tris).reshape(-1, 3)
    facet = np.repeat(np.arange(n), k * k)
    nrm = np.repeat(tdcs.unit_normals(tris), k * k, axis=0)
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    elen = np.stack([np.linalg.norm(tris[:, 1] - tris[:, 2], axis=1),
                     np.linalg.norm(tris[:, 2] - tris[:, 0], axis=1),
                     np.linalg.norm(tris[:, 0] - tris[:, 1], axis=1)], axis=1)
    hgt = 2.0 * area[:, None] / elen                           # (n, 3)
    edge = (bc[None, :, :] * hgt[:, None, :]).min(axis=2).reshape(-1)
    return dict(pts=pts, nrm=nrm, facet=facet, corners=corners,
                edge_eps=edge / eps, area=np.repeat(area / (k * k), k * k),
                facet_area=area, k=k)


def interleave(npt: int, workers: int) -> list:
    """Cell index blocks of at most CHUNK cells, STRIDED rather than
    contiguous: the folded kernel's cost sits on the rows that touch the
    surface, which element order keeps together, so a contiguous chunk of
    those rows ran for hours on one worker while the others sat idle. With
    every block holding a share of every row the pool stays level."""
    n_parts = max(1, -(-npt // CHUNK), 4 * max(1, int(workers or 1)))
    n_parts = min(n_parts, npt)
    return [np.arange(i, npt, n_parts) for i in range(n_parts)]


def _init(tris, slip, eps):
    _W.update(tris=tris, slip=slip, eps=float(eps),
              mat=mhs.Material(mu=MU, lam=LAM))


def _chunk(args):
    pts, nrm = args
    T = mhs.traction_matrix(pts, nrm, _W["tris"], _W["mat"], _W["eps"])
    return np.einsum("oisk,sk->oi", T, _W["slip"])


def _chunk_latepin(args):
    """The worker-side pin roughvol and isosurf use; the probe's control."""
    from mhs.parallel import pinned_blas
    with pinned_blas(1):
        return _chunk(args)


def mollified_traction(pts, nrm, tris, slip, workers, *, eps: float = EPS,
                       pin="parent", log=True) -> np.ndarray:
    """Elastic traction from the mollified kernel on every cell, in a pool
    created under the parent's BLAS pin (see the module docstring).
    ``workers`` of 0 or 1 runs in this process with no pool, the path the
    selfcheck takes."""
    import concurrent.futures as cf
    import contextlib
    import multiprocessing as mp
    from mhs.parallel import pinned_blas
    npt = pts.shape[0]
    splits = interleave(npt, workers)
    parts = [(pts[idx], nrm[idx]) for idx in splits]
    out = np.empty((npt, 3))
    if workers in (0, 1):
        _init(tris, slip, eps)
        for idx, part in zip(splits, parts):
            out[idx] = _chunk(part)
        return out
    fn = _chunk_latepin if pin == "worker" else _chunk
    guard = pinned_blas(1) if pin == "parent" else contextlib.nullcontext()
    t0 = time.time()
    with guard:
        with cf.ProcessPoolExecutor(max_workers=workers,
                                    mp_context=mp.get_context("spawn"),
                                    initializer=_init,
                                    initargs=(tris, slip, eps)) as ex:
            done = 0
            for i, got in enumerate(ex.map(fn, parts, chunksize=1)):
                out[splits[i]] = got
                done += got.shape[0]
                if log and (i % 10 == 0 or done == npt):
                    el = time.time() - t0
                    rate = done / max(el, 1e-9)
                    print(f"    mollified {done:8d}/{npt}  {el / 60:6.1f} min "
                          f" eta {(npt - done) / max(rate, 1e-9) / 60:6.1f} min"
                          f"  {el * 1e6 / max(done * tris.shape[0], 1):5.1f} "
                          f"us/pair", flush=True)
    return out


def classical_traction(pts, nrm, tris, slip) -> np.ndarray:
    """Traction from cutde on every cell: strain, Hooke, ``sigma . n``.

    Slip goes in through ``tdcs.from_cartesian``, which is not optional (see
    ``roughvol.classical``). Only the traction is ever read: the in-plane
    components of the classical stress jump across the surface, and nobody has
    measured what cutde returns for them on the plane. The traction does not.
    """
    import cutde.halfspace as hs
    nu = mhs.Material(mu=MU, lam=LAM).nu_if_you_must
    loc = tdcs.from_cartesian(tris, slip)
    out = np.empty((pts.shape[0], 3))
    t0 = time.time()
    for i in range(0, pts.shape[0], CHUNK_CUTDE):
        p = pts[i:i + CHUNK_CUTDE]
        E = np.asarray(hs.strain_matrix(p.copy(), tris.copy(), nu))
        sig = hooke(voigt_tensor(np.einsum("ovsk,sk->ov", E, loc)))
        out[i:i + p.shape[0]] = np.einsum("oab,ob->oa", sig,
                                          nrm[i:i + p.shape[0]])
    print(f"    classical {pts.shape[0]} cells in {(time.time() - t0) / 60:.1f}"
          f" min", flush=True)
    return out


def metrics(t: np.ndarray, facet: np.ndarray, z, coupled: bool) -> dict:
    """The three on-fault scalars from an induced traction, in GPa.

    SIGNS, stated once. ``t = sigma . n`` with the tdcs UPWARD normal, so
    ``t . n`` is tension positive, as ``kernel.project_normal``;
    compression-positive normal stress is ``-(t . n)``, as
    ``friction.resolve_tensor``. On a cell of facet ``f`` with Cartesian slip
    direction ``s_f``:

        dts     = t . s_f                       induced shear along the slip direction
        dsn     = -(t . n_f)                    induced normal stress, compression +
        tau_s   = lam |tau_load|_f + dts
        sigma_n = sigma_eff_0 + lam sigma_n_load + dsn      (coupled)
                = sigma_eff_0 + lam sigma_n_load            (uncoupled)
        S       = tau_s - mu_f sigma_n

    At a facet's centroid ``S`` is ``-w``, the LCP's own margin, to the
    solver's tolerance: that identity is what selfcheck [onfault] pins, and it
    is what makes the dense ``S`` the solution's own residual.
    """
    tris, lam = z["tris"], float(z["lam"])
    n = tris.shape[0]
    s_cart = Fr.slip_cartesian(tris, z["slip_dir"], np.ones(n))
    n_hat = tdcs.unit_normals(tris)
    dts = np.einsum("oi,oi->o", t, s_cart[facet])
    dsn = -np.einsum("oi,oi->o", t, n_hat[facet])
    tau_s = lam * z["tau_load_mag"][facet] + dts
    sig_n = z["sigma_eff_0"][facet] + lam * z["sigma_n_load"][facet]
    if coupled:
        sig_n = sig_n + dsn
    return dict(S=tau_s - z["mu"][facet] * sig_n, dsn=dsn, dts=dts)


def probe(z, c, nsrc, state_path, workers) -> int:
    """Each pool configuration in its own subprocess, then the projection.

    In subprocesses because a second spawn pool after a closed one deadlocks
    here. ``serial`` is one process with BLAS free to use every thread;
    ``parent`` is the pool pinned where the children inherit it; ``worker``
    is the pin inside the worker that roughvol and isosurf use, which the
    module docstring argues runs too late. The projection uses whichever
    configuration ``field`` will run.
    """
    npt = c["pts"].shape[0]
    rates = {}
    for cfg in dict.fromkeys(("serial", f"parent{workers}",
                              f"worker{workers}", "parent8")):
        cmd = [sys.executable, __file__, "field", "--state", str(state_path),
               "--k", str(c["k"]), "--probe-run", cfg]
        res = subprocess.run(cmd, capture_output=True, text=True)
        line = [ln for ln in res.stdout.splitlines() if ln.startswith("PROBE")]
        if res.returncode or not line:
            raise SystemExit(f"probe {cfg} failed:\n{res.stdout}\n{res.stderr}")
        rates[cfg] = float(line[-1].split("=")[1])
        print(f"  {cfg:10s} {rates[cfg]:6.2f} us/pair"
              + (f"  ({rates['serial'] / rates[cfg]:.2f}x serial)"
                 if cfg != "serial" else ""))
    use = rates[f"parent{workers}"]
    for k in (6, 8, 10, 12):
        pairs = nsrc * nsrc * k * k
        print(f"  k = {k:2d}: {pairs / 1e6:7.1f} M pairs -> "
              f"{pairs * use / 1e6 / 3600:5.2f} h mollified at {use:.1f} us/pair"
              f"; classical ~{pairs * 1.7e-6 / 60:4.1f} min")
    print(f"  this run, k = {c['k']}: {npt * nsrc / 1e6:.1f} M pairs -> "
          f"{npt * nsrc * use / 1e6 / 3600:.2f} h")
    return 0


def probe_run(z, c, cfg: str) -> int:
    """One timed configuration; prints ``PROBE cfg us_per_pair=...``."""
    tris, slip = z["tris"], z["slip_mollified"]
    m = 250
    if cfg == "serial":
        _init(tris, slip, EPS)
        _chunk((c["pts"][:50], c["nrm"][:50]))
        t0 = time.time()
        _chunk((c["pts"][:m], c["nrm"][:m]))
        us = (time.time() - t0) * 1e6 / (m * tris.shape[0])
    else:
        pin, w = ("parent", int(cfg[6:])) if cfg.startswith("parent") else \
                 ("worker", int(cfg[6:]))
        # ONE pool per process -- a second spawn pool after a closed one
        # deadlocks here -- so the pool's own start-up is inside the timing
        # and the rate is conservative by a few seconds.
        pts = np.concatenate([c["pts"][:m]] * (2 * w))
        nrm = np.concatenate([c["nrm"][:m]] * (2 * w))
        global CHUNK
        CHUNK = m
        t0 = time.time()
        mollified_traction(pts, nrm, tris, slip, w, pin=pin, log=False)
        us = (time.time() - t0) * 1e6 / (pts.shape[0] * tris.shape[0])
    print(f"PROBE {cfg} us_per_pair={us:.4f}")
    return 0


def alone_cells(z: dict, c: dict) -> np.ndarray:
    """The cells of fault A: where the A-alone experiment has a map."""
    return np.flatnonzero(z["fault_id"][c["facet"]] == 0)


def field_pass(name: str, z: dict, c: dict, tris: np.ndarray, workers: int,
               sn=None, experiment: str = "system") -> np.ndarray:
    """One kernel's traction on every cell: the pool pass of ``field``. The
    ``alone`` experiment reads A's own slip on A's cells only; B and C carry
    no slip there and have no map."""
    pts, nrm = c["pts"], c["nrm"]
    if experiment == "alone":
        sel = alone_cells(z, c)
        pts, nrm = pts[sel], nrm[sel]
    tag = "" if experiment == "system" else f"alone_"
    if name == "mollified":
        return mollified_traction(pts, nrm, tris, z[f"slip_{tag}mollified"],
                                  workers)
    if name == "continuous":
        return Nd.traction(pts, nrm, tris, sn, z[f"slip_{tag}continuous_nodes"],
                           workers, eps=float(z["eps_continuous"]), mu=MU,
                           lam=LAM)
    raise ValueError(name)


def cmd_field(a) -> int:
    z = dict(np.load(a.state, allow_pickle=False))
    _, net = build()
    if not np.array_equal(net.tris, z["tris"]):
        raise SystemExit("the state's mesh is not the one build() makes now; "
                         "re-run `onfault.py solve` first")
    tris = z["tris"]
    coupled = bool(z["coupled"])
    c = cells(tris, a.k)
    npt, nsrc = c["pts"].shape[0], tris.shape[0]
    if a.probe_run:
        return probe_run(z, c, a.probe_run)
    if a.pass_:
        # one pool pass in a process of its own; the parent merges it
        sn = Nd.table(net) if a.pass_ == "continuous" else None
        t = field_pass(a.pass_, z, c, tris, a.workers, sn, a.experiment)
        key = f"t_{a.pass_}" if a.experiment == "system" else f"t_alone_{a.pass_}"
        np.savez_compressed(a.out, **{key: t}, k=a.k, tris=tris)
        print(f"  pass {a.pass_} ({a.experiment}): saved {a.out}")
        return 0
    print(f"  k = {a.k}: {a.k * a.k} cells per facet, {npt} cells x {nsrc} "
          f"sources = {npt * nsrc / 1e6:.1f} M pairs; nearest cell to an edge "
          f"{c['edge_eps'].min():.3f} eps; {'coupled' if coupled else 'uncoupled'}"
          f" LCP at lam {float(z['lam']):.5f} GPa")
    if a.probe:
        return probe(z, c, nsrc, a.state, a.workers)

    def save(**extra):
        np.savez_compressed(
            a.out, tris=tris, k=a.k, corners=c["corners"], facet=c["facet"],
            edge_eps=c["edge_eps"], area=c["area"], facet_area=c["facet_area"],
            pts=c["pts"], lam=z["lam"], coupled=coupled, state=str(a.state),
            **stamp, **extra)
        print(f"  saved {a.out}" + (" (pool passes only, classical to come)"
                                    if "t_classical" not in extra else ""))

    # Banked before the cheap pass is attempted, and reloaded when the cells
    # match: the lesson of isosurf's lost 2.2 h. Each pool pass runs in a
    # subprocess of its own and is merged here.
    sel = alone_cells(z, c)
    passes = [(name, exp) for exp in EXPERIMENTS for name in POOL_PASSES]
    keyof = lambda name, exp: f"t_{name}" if exp == "system" else f"t_alone_{name}"
    rows = {"system": npt, "alone": sel.size}
    # A banked pass is reused only for the SAME state (the slip it was read
    # from) and the SAME kernel: keyed on the mesh alone, the first folded
    # field run quietly reused lift-era tractions of an older slip, because
    # the mesh had not changed.
    stamp = dict(state_sha=hashlib.sha256(pathlib.Path(a.state).read_bytes()).hexdigest(),
                 kernel=json.dumps(K.fold_tag(), sort_keys=True))
    banked: dict = {}
    if pathlib.Path(a.out).exists():
        with np.load(a.out, allow_pickle=False) as z0:
            same = (int(z0["k"]) == a.k and np.array_equal(z0["tris"], tris)
                    and all(key in z0 and str(z0[key]) == val
                            for key, val in stamp.items()))
            if same:
                for name, exp in passes:
                    key = keyof(name, exp)
                    if key in z0 and z0[key].shape == (rows[exp], 3):
                        banked[key] = z0[key]
                        print(f"  reusing the {name} ({exp}) pass already in {a.out}")
            else:
                print(f"  {a.out} is from another state or kernel: nothing reused")
    for name, exp in passes:
        key = keyof(name, exp)
        if key in banked:
            continue
        side = pathlib.Path(a.out).with_suffix(f".{name}.{exp}.npz")
        subprocess.run([sys.executable, __file__, "field", "--state",
                        str(a.state), "--k", str(a.k), "--out", str(side),
                        "--workers", str(a.workers), "--pass", name,
                        "--experiment", exp], check=True)
        with np.load(side, allow_pickle=False) as zs:
            banked[key] = zs[key]
        side.unlink()
        save(**banked)
    tr = dict(banked)
    tr["t_classical"] = classical_traction(c["pts"], c["nrm"], tris,
                                           z["slip_classical"])
    tr["t_alone_classical"] = classical_traction(c["pts"][sel], c["nrm"][sel],
                                                 tris, z["slip_alone_classical"])
    got = {"alone_cells": sel}
    for exp in EXPERIMENTS:
        for k in KERNELS:
            t = tr[keyof(k, exp)]
            bad = int((~np.isfinite(t)).any(axis=1).sum())
            print(f"    {k} ({exp}): {bad} non-finite cells of {t.shape[0]}")
            facet = c["facet"] if exp == "system" else c["facet"][sel]
            tag = "" if exp == "system" else "alone_"
            got.update({f"{tag}{k}_{m}": v for m, v in
                        metrics(t, facet, z, coupled).items()})
    save(**tr, **got)
    for exp in EXPERIMENTS:
        tag = "" if exp == "system" else "alone_"
        for m in METRICS:
            for k in KERNELS:
                v = got[f"{tag}{k}_{m}"]
                print(f"    {exp:6s} {m:4s} {k:10s} [{np.nanmin(v) * 1e3:+9.3f}, "
                      f"{np.nanmax(v) * 1e3:+9.3f}] MPa  p50 |.| "
                      f"{np.nanmedian(np.abs(v)) * 1e3:7.3f}")
    return 0


# -------------------------------------------------------------- payload --
def window_rule(v: np.ndarray, area: np.ndarray, mask=None) -> float:
    """Half-width of the symmetric window: the area-weighted p99 of |v| over
    the cells in ``mask``, to two significant figures of MPa."""
    sel = np.ones(v.shape[0], bool) if mask is None else mask
    x, wgt = np.abs(v[sel]), area[sel]
    order = np.argsort(x)
    cum = np.cumsum(wgt[order]) / wgt.sum()
    p99 = float(x[order][np.searchsorted(cum, 0.99)])
    mpa = p99 * 1e3
    if mpa <= 0.0:
        return 0.0
    mag = 10 ** np.floor(np.log10(mpa))
    return float(np.round(mpa / mag, 1) * mag) / 1e3


def stats(v: np.ndarray, area: np.ndarray, win: float) -> dict:
    fin = np.isfinite(v)
    x = v[fin]
    return dict(min=float(x.min()), max=float(x.max()),
                p01=float(np.percentile(x, 1)), p50=float(np.percentile(x, 50)),
                p99=float(np.percentile(x, 99)),
                nonfinite=int((~fin).sum()),
                saturated_area_fraction=float(
                    area[fin & (np.abs(v) > win)].sum() / area.sum()))


def cmd_payload(a) -> int:
    f = dict(np.load(a.field, allow_pickle=False))
    z = dict(np.load(str(f["state"]), allow_pickle=False))
    tris, fid, k = f["tris"], z["fault_id"], int(f["k"])
    lam, coupled = float(f["lam"]), bool(f["coupled"])
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    facet, area = f["facet"], f["area"]
    for kern in KERNELS:
        bad = int((~np.isfinite(f[f"t_{kern}"])).any(axis=1).sum())
        if bad:
            raise SystemExit(f"{kern}: {bad} non-finite cells; refused")

    arrays: dict = {}

    def put(name, arr, dtype):
        buf = np.ascontiguousarray(arr.astype(dtype)).tobytes()
        (out / f"{name}.bin").write_bytes(buf)
        arrays[name] = dict(dtype=np.dtype(dtype).name, shape=list(arr.shape),
                            bytes=len(buf),
                            sha256=hashlib.sha256(buf).hexdigest()[:16])

    put("onfault.tris", tris, np.float32)
    put("onfault.corners", f["corners"], np.float32)
    # 0.02 eps per byte: a facet's centroid is its inradius 0.2887 h = 2.89 eps
    # from the nearest edge, so 0.01 steps would overflow the byte.
    put("onfault.edge_eps", np.minimum(np.round(f["edge_eps"] / EDGE_STEP), 255),
        np.uint8)
    put("onfault.fault_id", fid, np.uint8)
    put("onfault.area", f["facet_area"], np.float32)   # per facet; a cell is 1/k^2 of it
    # At k = 1 the Coulomb excess is identically zero on every slipping
    # element (it is the solver's own margin), so its window is set by the
    # locked deficit: the p99 over ALL facets, like the other metrics.
    win = {}
    for m in METRICS:
        win[m] = window_rule(f[f"mollified_{m}"], area)
    st: dict = {}
    sel = f["alone_cells"]
    on_a = fid == 0
    for kern in KERNELS:
        put(f"onfault.{kern}.slip", np.linalg.norm(z[f"slip_{kern}"], axis=1),
            np.float32)
        st[kern] = {}
        for m in METRICS:
            v = f[f"{kern}_{m}"]
            put(f"onfault.{kern}.{m}", v, np.float32)
            st[kern][m] = stats(v, area, win[m])
        # A alone: a map on A's cells, NaN where there is no fault in that
        # experiment (the viewer paints those grey).
        s_al = np.linalg.norm(z[f"slip_alone_{kern}"], axis=1)
        s_al[~on_a] = np.nan
        put(f"onfault.alone.{kern}.slip", s_al, np.float32)
        st[f"alone.{kern}"] = {}
        for m in METRICS:
            v = np.full(facet.shape[0], np.nan)
            v[sel] = f[f"alone_{kern}_{m}"]
            if not np.isfinite(v[sel]).all():
                raise SystemExit(f"alone {kern} {m}: non-finite on A; refused")
            put(f"onfault.alone.{kern}.{m}", v, np.float32)
            st[f"alone.{kern}"][m] = stats(v, area, win[m])

    # The welded surface of every fault, for the wireframe and the rims.
    geom = []
    for i, fdef in enumerate(FAULTS):
        t = tris[fid == i]
        g = R.surface(t)
        g["outline"] = R.rim_outline(t)
        g["name"] = fdef["name"]
        geom.append(g)
    (out / "onfault_geometry.json").write_text(json.dumps(
        dict(faults=geom, units="km"), allow_nan=False))

    table = json.loads(str(z["table"]))
    meta = json.loads(str(z["meta"]))
    faults = []
    for i, fm in enumerate(meta["faults"]):
        m = fid == i
        faults.append(dict(
            name=fm["name"], a_km=fm["a"], dip_deg=fm["dip"], seed=fm["seed"],
            trace_km=fm["trace"], trace_length_km=fm["trace_length_km"],
            v_c_km=fm["v_c"], attach=fm["attach"], junction=fm["junction"],
            n=int(m.sum()), median_edge_km=fm["median_edge_km"],
            tilt_rms_deg=fm["tilt_rms_deg"], height_rms_km=fm["height_rms_km"],
            slope_rms=fm["slope_rms"],
            remote_shear_MPa=float(lam * z["tau_load_mag"][m].mean() * 1e3),
            remote_sigma_n_MPa=float((z["sigma_eff_0"][m]
                                      + lam * z["sigma_n_load"][m]).mean() * 1e3),
            **{kern: table[kern][i] for kern in KERNELS}))
    man = dict(
        schema=SCHEMA, units="km, GPa", git=K._git(), source="onfault.py",
        k=k, cells_per_facet=k * k, n_facets=int(tris.shape[0]),
        n_cells=int(facet.shape[0]), eps_km=EPS, h_km=H_KM,
        nearest_edge_eps=float(f["edge_eps"].min()), edge_eps_step=EDGE_STEP,
        lam_GPa=lam, lam0_GPa=float(z["lam0"]), lam_over_lam0=lam / float(z["lam0"]),
        ladder=json.loads(str(z["ladder"])), lam_rule=LAM_RULE,
        friction=dict(mu=float(z["mu"].mean()),
                      sigma_eff_GPa=float(z["sigma_eff_0"].mean()),
                      regime=REGIME, mean_stress_ratio=MEAN_STRESS,
                      coupled=coupled),
        system=dict(clearance_km=(meta["clearance_km"]
                                  if meta["clearance_km"] is not None
                                  and np.isfinite(meta["clearance_km"]) else None),
                    clearance_reach_km=meta.get("clearance_reach_km",
                                                CLEARANCE_OVER_H * H_KM),
                    contact=meta["contact"],
                    min_nz=meta["min_nz"], slope=SLOPE, corr_km=CORR_KM,
                    taper_over_h=meta["taper_over_h"],
                    junction_taper_over_h=meta["junction_taper_over_h"]),
        faults=faults,
        kernels={kern: dict(min_eig_sym_negP=float(z[f"mineig_{kern}"]),
                            B=float(z[f"B_{kern}"]), sweeps=int(z[f"sweeps_{kern}"]),
                            residual=float(z[f"residual_{kern}"]),
                            active_fraction=float(z[f"active_{kern}"].mean()),
                            **({"unknowns": "shared nodes, P2 continuous",
                                "eps_km": float(z["eps_continuous"]),
                                "n_nodes": int(z["node_coords"].shape[0]),
                                "n_free": int(z["node_free"].sum()),
                                "min_eig_sym_negP_uncoupled":
                                    float(z["mineig_continuous_uncoupled"]),
                                "active_node_fraction":
                                    float(z["active_continuous_nodes"].mean())}
                               if kern == "continuous" else
                               {"unknowns": "elements, P0"}))
                 for kern in KERNELS},
        window=dict(rule="symmetric; half-width = area-weighted p99 of the "
                         "mollified |metric| over all facets, rounded to 2 "
                         "significant figures of MPa",
                    **{m: win[m] for m in METRICS}),
        stats=st, arrays=arrays,
        metrics=dict(S="tau_s - mu sigma_n', total, GPa",
                     dsn="slip-induced normal stress, compression positive, GPa",
                     dts="slip-induced shear along the slip direction, GPa"),
        experiments=dict(
            system="A, B and C together",
            alone="A by itself: the same mesh of A, the same matrix (its A-A "
                  "block), the same load; B and C absent, so their cells "
                  "carry NaN",
            **{f"alone_{kern}": dict(
                active_fraction=float(z[f"active_alone_{kern}"][on_a].mean()),
                sweeps=int(z[f"sweeps_alone_{kern}"]),
                residual=float(z[f"residual_alone_{kern}"]),
                **{key: table[kern][0][key] for key in
                   ("moment_system_km3", "moment_alone_km3", "moment_ratio",
                    "shadow_GPa", "max_slip_m", "max_slip_alone_m")})
               for kern in KERNELS}),
        defaults=dict(kernel="mollified", metric="S", experiment="system",
                      mask_eps=0.0))
    (out / "onfault_manifest.json").write_text(
        json.dumps(man, indent=1, sort_keys=True, allow_nan=False) + "\n")
    total = sum(v["bytes"] for v in arrays.values())
    print(f"  {len(arrays)} arrays, {total / 1024 ** 2:.2f} MiB; windows "
          + ", ".join(f"{m} +-{win[m] * 1e3:g} MPa" for m in METRICS))
    for kern in KERNELS:
        print(f"    {kern:9s} saturated at the window: " + ", ".join(
            f"{m} {100 * st[kern][m]['saturated_area_fraction']:.1f} %"
            for m in METRICS))
    return 0


# ----------------------------------------------------------- thresholds --
def _onset(P, d, g, lam_lo, lam_hi, mask, include, tol=1e-3) -> float:
    """The smallest load in ``[lam_lo, lam_hi]`` at which any element of
    ``mask`` slips in the LCP restricted to ``include``, by bisection to a
    relative ``tol``."""
    sol = St.solve(P, d, g, lam_hi, include=include)
    if not (sol.delta[mask] > 0.0).any():
        raise SystemExit("no slip on the fault at the upper bound of the scan")
    while (lam_hi - lam_lo) / lam_hi > tol:
        mid = 0.5 * (lam_lo + lam_hi)
        sol = St.solve(P, d, g, mid, include=include)
        if (sol.delta[mask] > 0.0).any():
            lam_hi = mid
        else:
            lam_lo = mid
    return lam_hi


def cmd_thresholds(a) -> int:
    """The failure-load shift on the page's system: for each fault, the load
    at which it first slips ALONE (the LCP restricted to it: the remote load
    and its own self-interaction) against the load at which it first slips
    IN THE SYSTEM, and the active fraction of each fault against the load
    both ways. Static, path-free (the LCP has one solution), both P0
    kernels. What it measures: the onsets never move (the remote load reaches
    strength before any neighbour's stress does) and the neighbours act on
    COMPLETION, a shadow that throttles C."""
    here = pathlib.Path(__file__).resolve().parent
    _, net = build()
    fid, area = net.fault_id, net.areas()
    names = [f["name"] for f in net.meta["faults"]]
    fric = Fr.build_friction(net, regime=REGIME, drop_ratio=DROP_RATIO,
                             mean_stress_ratio=MEAN_STRESS)
    d, g, _ = C.coefficients(fric)
    ff = fric.first_failure_lambda()
    lam0 = float(np.nanmin(ff))
    masks = {nm: fid == i for i, nm in enumerate(names)}
    print(f"  {net.n_elements} elements; first failure with no interaction "
          f"lam0 = {lam0:.5f} GPa")
    for nm in names:
        m = masks[nm]
        print(f"    {nm}: first element fails alone at "
              f"{np.nanmin(ff[m]) / lam0:.4f} lam0; "
              f"{int(np.isinf(ff[m]).sum())} of {int(m.sum())} can never fail")
    Km, _ = K.assemble(net, mu=MU, lam=LAM, workers=a.workers, verbose=True)
    Kc = classical_matrix(net.tris)
    lams = np.concatenate([np.linspace(0.95, 1.10, 31),
                           np.linspace(1.11, 1.60, 50)]) * lam0
    results = {}
    for kname, Kmat in (("mollified", Km), ("classical", Kc)):
        P = K.project(Kmat, fric.slip_dir, fric.slip_dir)
        onsets = {}
        for nm in names:
            m = masks[nm]
            lo = 0.95 * float(np.nanmin(ff[m]))
            la = _onset(P, d, g, lo, 1.6 * lam0, m, include=m)
            ls = _onset(P, d, g, lo, 1.6 * lam0, m, include=None)
            onsets[nm] = dict(alone=la / lam0, system=ls / lam0,
                              shift=(ls - la) / la)
            print(f"  {kname}: {nm} first slips alone at {la / lam0:.4f} lam0, "
                  f"in the system at {ls / lam0:.4f}: shift "
                  f"{100 * (ls - la) / la:+.2f} %", flush=True)
        curves = {nm: dict(alone=[], system=[], moment_alone=[], moment_system=[])
                  for nm in names}
        warm_s, warm_a = None, {nm: None for nm in names}
        t0 = time.time()
        for lam in lams:
            sol = St.solve(P, d, g, lam, warm=warm_s)
            warm_s = sol.delta
            for nm in names:
                m = masks[nm]
                sa = St.solve(P, d, g, lam, include=m, warm=warm_a[nm])
                warm_a[nm] = sa.delta
                curves[nm]["system"].append(float((sol.delta[m] > 0).mean()))
                curves[nm]["alone"].append(float((sa.delta[m] > 0).mean()))
                curves[nm]["moment_system"].append(float((sol.delta * area)[m].sum()))
                curves[nm]["moment_alone"].append(float((sa.delta * area)[m].sum()))
        print(f"  {kname}: {lams.size} loads x {1 + len(names)} solves in "
              f"{time.time() - t0:.0f} s", flush=True)
        for nm in names:
            c = {k: np.array(v) for k, v in curves[nm].items()}
            j = int(np.argmin(np.abs(lams / lam0 - 1.10)))
            print(f"    {nm} at 1.10 lam0: active alone {c['alone'][j]:.3f}, in "
                  f"the system {c['system'][j]:.3f}; at 1.60: "
                  f"{c['alone'][-1]:.3f} / {c['system'][-1]:.3f}")
        results[kname] = dict(onsets=onsets, lams_over_lam0=(lams / lam0).tolist(),
                              curves=curves)
    out = here / "out"
    out.mkdir(exist_ok=True)
    (out / "onfault_thresholds.json").write_text(
        json.dumps(dict(lam0=lam0, names=names, results=results), indent=1))
    import matplotlib                                          # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                            # noqa: PLC0415
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    cols = dict(zip(names, ("#1f77b4", "#ff7f0e", "#2ca02c")))
    for i, (ax, kname) in enumerate(zip(axes, ("mollified", "classical"))):
        r = results[kname]
        x = np.array(r["lams_over_lam0"])
        for nm in names:
            c = r["curves"][nm]
            ax.plot(x, c["system"], color=cols[nm], lw=1.6, label=f"{nm}, in the system")
            ax.plot(x, c["alone"], color=cols[nm], lw=1.0, ls="--", label=f"{nm}, alone")
        ax.tick_params(direction="out", length=3, width=0.8)
        ax.set_xlim(0.95, 1.6)
        ax.set_xticks([1.0, 1.3, 1.6])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(r"$\lambda / \lambda_0$")
        ax.set_title(f"{kname} P0", fontsize=10)
        ax.text(0.97, 0.05, "ab"[i], transform=ax.transAxes, ha="right", va="bottom")
    axes[0].set_ylabel("active fraction of the fault")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig_dir = pathlib.Path(a.figure).parent
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figure, dpi=200, bbox_inches="tight")
    print(f"  -> {out / 'onfault_thresholds.json'}, {a.figure}")
    return 0


def main(argv=None) -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="system + three matrices + three LCPs")
    s.add_argument("--out", default=str(here / "out" / "onfault_state.npz"))
    s.add_argument("--workers", type=int, default=12)
    n_ = sub.add_parser("assemble", help="the nodal matrix into the cache, in "
                                         "its own process")
    n_.add_argument("--workers", type=int, default=12)
    f = sub.add_parser("field", help="dense on-fault readout, three kernels")
    f.add_argument("--state", default=str(here / "out" / "onfault_state.npz"))
    f.add_argument("--k", type=int, default=K_DEFAULT)
    f.add_argument("--out", default=None)
    f.add_argument("--workers", type=int, default=12)
    f.add_argument("--probe", action="store_true")
    f.add_argument("--probe-run", default=None, help=argparse.SUPPRESS)
    f.add_argument("--pass", dest="pass_", default=None, choices=POOL_PASSES,
                   help=argparse.SUPPRESS)
    f.add_argument("--experiment", default="system", choices=EXPERIMENTS,
                   help=argparse.SUPPRESS)
    t = sub.add_parser("thresholds", help="the failure-load shift and the "
                                          "active fraction, alone vs in the "
                                          "system, both P0 kernels")
    t.add_argument("--workers", type=int, default=1)
    t.add_argument("--figure", default=str(here.parents[1] / "website" / "src"
                                           / "figures" / "onfault_thresholds.png"))
    p = sub.add_parser("payload", help="field -> website/public/data")
    p.add_argument("--field", default=str(here / "out" / "onfault_field.npz"))
    p.add_argument("--out", default=str(here.parents[1] / "website" / "public"
                                        / "data"))
    a = ap.parse_args(argv)
    if a.cmd == "solve":
        return cmd_solve(a)
    if a.cmd == "assemble":
        return cmd_assemble(a)
    if a.cmd == "field":
        a.out = a.out or str(here / "out" / "onfault_field.npz")
        return cmd_field(a)
    if a.cmd == "thresholds":
        return cmd_thresholds(a)
    return cmd_payload(a)


if __name__ == "__main__":
    raise SystemExit(main())
