"""On-fault stress on a system of three surface-breaking rough faults, one
value per element, both kernels, for the site.

    python onfault.py solve   [--out out/onfault_state.npz] [--workers 12]
    python onfault.py field   --state out/onfault_state.npz [--k 1] [--probe]
    python onfault.py payload --field out/onfault_field.npz

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

THREE COMMANDS, for the reason roughvol.py gives: a second spawn pool raised
after the first has closed deadlocks on this machine. ``solve`` runs the
assembly pool and saves; ``field`` runs the readout pool and saves; ``payload``
only reads. The probe runs each pool configuration in a subprocess of its own
for the same reason.

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
import roughvol as R                                         # noqa: E402
import static as St                                          # noqa: E402
import system as Sy                                          # noqa: E402

MU, LAM = R.MU, R.LAM
H_KM = 0.25                      # half the published crack's h: more, smaller elements
EPS = 0.1 * H_KM
SLOPE = R.SLOPE
CORR_KM = R.CORR_OVER_H * R.H_KM # the published correlation length, in km (4)
REGIME, DROP_RATIO, MEAN_STRESS = R.REGIME, R.DROP_RATIO, R.MEAN_STRESS
SCHEMA = 2

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
KERNELS = ("mollified", "classical")

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
    t0 = time.time()
    Km, info = K.assemble(net, mu=MU, lam=LAM, workers=a.workers, verbose=True)
    t1 = time.time()
    Kc = classical_matrix(net.tris)
    print(f"  classical matrix in {time.time() - t1:.1f} s "
          f"(mollified {t1 - t0:.1f} s); both diagonals negative")
    ops = {"mollified": operators(Km, fric), "classical": operators(Kc, fric)}
    del Km, Kc
    for k in KERNELS:
        print(f"    {k:9s} min eig sym(-P_eff) {ops[k]['mineig']:+.4f}  "
              f"B {ops[k]['B']:.4f}")

    # Coupled unless the coupling costs definiteness for EITHER kernel; then
    # both fall back to the uncoupled form, so that whatever S means, it means
    # the same thing in both maps.
    coupled = all(ops[k]["mineig"] > 0.0 for k in KERNELS)
    if not coupled:
        print("  coupling loses positive definiteness: falling back to the "
              "uncoupled LCP for both kernels")
        d, g, _ = C.coefficients(fric)
        for k in KERNELS:
            ops[k]["P_eff"], ops[k]["d"], ops[k]["g"] = ops[k]["P"], d, g

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

    # Both kernels at the one load, plus each fault ALONE on the same matrix:
    # the interaction pre-flight, in seconds, before the hours.
    state: dict = dict(tris=net.tris, fault_id=fid, lam=lam, lam0=lam0,
                       ladder=json.dumps(ladder), coupled=coupled, eps=EPS,
                       h=H_KM, mu=fric.mu, sigma_eff_0=fric.sigma_eff_0,
                       tau_load_mag=fric.tau_load_mag,
                       sigma_n_load=fric.sigma_n_load, slip_dir=fric.slip_dir,
                       meta=json.dumps(net.meta, default=str))
    table = {}
    for k in KERNELS:
        o = ops[k]
        sol = St.solve(o["P_eff"], o["d"], o["g"], lam)
        if not sol.converged:
            raise SystemExit(f"{k}: the LCP did not converge in {sol.sweeps}")
        w = o["g"] - (lam * o["d"] + o["P_eff"].T @ sol.delta)
        slip = Fr.slip_cartesian(net.tris, fric.slip_dir, sol.delta)
        areas = net.areas()
        rows = []
        for i, f in enumerate(FAULTS):
            m = fid == i
            alone = St.solve(o["P_eff"], o["d"], o["g"], lam, include=m)
            others = sol.delta.copy()
            others[m] = 0.0
            shadow = float((o["P_eff"].T @ others)[m].mean())
            mom_sys = float((sol.delta[m] * areas[m]).sum())
            mom_alone = float((alone.delta[m] * areas[m]).sum())
            rows.append(dict(
                name=f["name"], active_alone=float((alone.delta[m] > 0).mean()),
                active_system=float((sol.delta[m] > 0).mean()),
                moment_ratio=(mom_sys / mom_alone if mom_alone > 0 else
                              float("inf") if mom_sys > 0 else 1.0),
                shadow_GPa=shadow, max_slip_m=float(sol.delta[m].max() * 1e3)))
            print(f"    {k:9s} {f['name']}: active alone {rows[-1]['active_alone']:.2f}"
                  f" -> system {rows[-1]['active_system']:.2f}, moment x"
                  f"{rows[-1]['moment_ratio']:.3f}, shadow "
                  f"{shadow * 1e3:+.2f} MPa, max slip "
                  f"{rows[-1]['max_slip_m']:.3f} m")
        table[k] = rows
        state.update({f"delta_{k}": sol.delta, f"w_{k}": w, f"slip_{k}": slip,
                      f"active_{k}": sol.active, f"mineig_{k}": o["mineig"],
                      f"B_{k}": o["B"], f"sweeps_{k}": sol.sweeps,
                      f"residual_{k}": sol.residual})
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
    parts = [(pts[i:i + CHUNK], nrm[i:i + CHUNK]) for i in range(0, npt, CHUNK)]
    out = np.empty((npt, 3))
    if workers in (0, 1):
        _init(tris, slip, eps)
        for i, part in enumerate(parts):
            got = _chunk(part)
            out[i * CHUNK:i * CHUNK + got.shape[0]] = got
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
                out[i * CHUNK:i * CHUNK + got.shape[0]] = got
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
            **extra)
        print(f"  saved {a.out}" + (" (mollified only, classical to come)"
                                    if "t_classical" not in extra else ""))

    # Banked before the cheap pass is attempted, and reloaded when the cells
    # match: the lesson of isosurf's lost 2.2 h.
    done = None
    if pathlib.Path(a.out).exists():
        with np.load(a.out, allow_pickle=False) as z0:
            if (int(z0["k"]) == a.k and np.array_equal(z0["tris"], tris)
                    and "t_mollified" in z0
                    and z0["t_mollified"].shape == (npt, 3)):
                done = z0["t_mollified"]
                print(f"  reusing the mollified pass already in {a.out}")
    t_m = done if done is not None else mollified_traction(
        c["pts"], c["nrm"], tris, z["slip_mollified"], a.workers)
    if done is None:
        save(t_mollified=t_m)
    t_c = classical_traction(c["pts"], c["nrm"], tris, z["slip_classical"])
    got = {}
    for k, t in (("mollified", t_m), ("classical", t_c)):
        bad = int((~np.isfinite(t)).any(axis=1).sum())
        print(f"    {k}: {bad} non-finite cells of {npt}")
        got.update({f"{k}_{m}": v for m, v in
                    metrics(t, c["facet"], z, coupled).items()})
    save(t_mollified=t_m, t_classical=t_c, **got)
    for m in METRICS:
        for k in KERNELS:
            v = got[f"{k}_{m}"]
            print(f"    {m:4s} {k:9s} [{np.nanmin(v) * 1e3:+9.3f}, "
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
    for kern in KERNELS:
        put(f"onfault.{kern}.slip", np.linalg.norm(z[f"slip_{kern}"], axis=1),
            np.float32)
        st[kern] = {}
        for m in METRICS:
            v = f[f"{kern}_{m}"]
            put(f"onfault.{kern}.{m}", v, np.float32)
            st[kern][m] = stats(v, area, win[m])

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
                            active_fraction=float(z[f"active_{kern}"].mean()))
                 for kern in KERNELS},
        window=dict(rule="symmetric; half-width = area-weighted p99 of the "
                         "mollified |metric| over all facets, rounded to 2 "
                         "significant figures of MPa",
                    **{m: win[m] for m in METRICS}),
        stats=st, arrays=arrays,
        metrics=dict(S="tau_s - mu sigma_n', total, GPa",
                     dsn="slip-induced normal stress, compression positive, GPa",
                     dts="slip-induced shear along the slip direction, GPa"),
        defaults=dict(kernel="mollified", metric="S", mask_eps=0.0))
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


def main(argv=None) -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="system + both matrices + both LCPs")
    s.add_argument("--out", default=str(here / "out" / "onfault_state.npz"))
    s.add_argument("--workers", type=int, default=12)
    f = sub.add_parser("field", help="dense on-fault readout, both kernels")
    f.add_argument("--state", default=str(here / "out" / "onfault_state.npz"))
    f.add_argument("--k", type=int, default=K_DEFAULT)
    f.add_argument("--out", default=None)
    f.add_argument("--workers", type=int, default=12)
    f.add_argument("--probe", action="store_true")
    f.add_argument("--probe-run", default=None, help=argparse.SUPPRESS)
    p = sub.add_parser("payload", help="field -> website/public/data")
    p.add_argument("--field", default=str(here / "out" / "onfault_field.npz"))
    p.add_argument("--out", default=str(here.parents[1] / "website" / "public"
                                        / "data"))
    a = ap.parse_args(argv)
    if a.cmd == "solve":
        return cmd_solve(a)
    if a.cmd == "field":
        a.out = a.out or str(here / "out" / "onfault_field.npz")
        return cmd_field(a)
    return cmd_payload(a)


if __name__ == "__main__":
    raise SystemExit(main())
