#!/usr/bin/env python
"""The Longitudinal Valley Fault under a sweep of regional stress tensors.

    python studies/active_fractal/taiwan.py assemble --workers 8
    python studies/active_fractal/taiwan.py run
    python studies/active_fractal/taiwan.py figure
    python studies/active_fractal/taiwan.py payload

THE FAULT. Two Taiwan Earthquake Model (TEM) seismogenic-structure segments
meshed by the user (``data/taiwan/``, Gmsh 4.1, lon/lat in degrees and depth
in km): segment 64 is the southern 80 km of the Longitudinal Valley Fault
and segment 41 the northern 50 km; they meet at one trace vertex. Together
2,498 triangles of median edge 1.9 km, striking about N20E, dipping about
60 deg east to 20 km, breaking the surface along the whole trace. The mesh
is used AS SUPPLIED: nothing is re-meshed, so its 0.12 km slivers are in the
matrix, and the page says so.

THE FRAME. A local east-north-up frame in km with its origin at the shared
trace vertex: x east, y north, z up, the surface at z = 0 and the body
below, which is mhs's half space. Geographic azimuths are clockwise from
north, so a direction at azimuth ``theta`` is the counter-clockwise angle
``90 - theta`` from +x, which is what ``friction.rotate_about_vertical``
takes. Every triangle is wound so its normal points up, as cutde and tdcs
read it (``onfault.frame_gate``).

THE TENSORS. ``friction``'s load is ``lambda`` times a unit deviator, lambda
being the maximum shear the regional field carries, in GPa, plus the
isotropic part ``mean_stress_ratio``. The family swept here has its most
compressive axis horizontal at azimuth ``theta`` and a REGIME parameter
``r`` in [0, 1] between the two Andersonian states a reverse-sinistral fault
can be in: ``r = 0`` is the thrust tensor (the vertical stress the least
compressive), ``r = 1`` the strike-slip tensor (the vertical stress
intermediate), and in between ``diag(-1, r, 1 - r)`` in the principal frame
renormalised to unit maximum shear, so that lambda keeps its meaning on the
whole family and ``r = 0.5`` is the axially symmetric compression where the
two regimes are equally favoured. The end members are ``LOAD_TENSORS``'
thrust and strike-slip tensors (the latter rotated so its compression axis
is +x), gated in ``selfcheck.py -k taiwan``.

THE SOLVE. The static Coulomb problem under two slip rules. FIXED RAKE:
the scalar LCP (``static.solve``), each element slipping along the shear
the tensor resolves on it, one strength per element, slip wherever the
shear reaches it and nowhere else. FREE RAKE: the cone problem
(``static.solve_cone``), each element's slip a vector in its plane that
turns to follow the total shear, regional plus what its neighbours' slip
adds. The fixed rake is a constraint on the free problem; on this fault it
costs under 1 % of moment for thrust-like tensors and up to 11 % for the
strike-slip ones, with the active set essentially unchanged. One unique
answer at every load (measured, see ``static.py``), no events. Each tensor is scanned in the load from its
own first failure ``lambda0`` (the load at which the best-oriented element
reaches strength with no interaction, uniform model) to ``6 lambda0``,
warm-started, and records every element's activation load and its slip at
each scan load; the summary quotes ``lambda0`` and the load at which half
the fault's area is active, both in MPa of maximum shear, which is what
makes tensors of different shape comparable. Two
effective-normal-stress models: uniform 150 MPa, and lithostatic minus a
pore-pressure ratio of 0.4, which makes the shallow rows weak. Two kernels:
the mollified half-space kernel with the free surface folded in (eps 0.1 h)
and the classical triangular dislocation as the control. Friction 0.6
everywhere, mean-stress ratio 2/3, the on-fault page's values. Under
the fixed rule the rake of each element is set by the tensor alone
(``friction.build_friction``), so that rake map is a prediction of the
tensor and not of the solve; under the free rule the solve's own rake is
recorded beside it.

Outputs: ``out/taiwan_sweep.npz`` (gitignored), the summary figure in
``website/src/figures/taiwan_sweep.png`` and the viewer's payload in
``website/public/data/taiwan_*`` for the site's /taiwan/ page. One spawn pool
per process (``assemble``), BLAS pinned in the parent first.
"""
from __future__ import annotations

import os
import sys

_CMD = sys.argv[1] if len(sys.argv) > 1 else ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1" if _CMD == "assemble" else "8")

import argparse                                              # noqa: E402
import hashlib                                               # noqa: E402
import json                                                  # noqa: E402
import math                                                  # noqa: E402
import pathlib                                               # noqa: E402
import time                                                  # noqa: E402

import numpy as np                                           # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mhs import tdcs                                         # noqa: E402
import cascade as C                                          # noqa: E402
import friction as Fr                                        # noqa: E402
import kernel as K                                           # noqa: E402
import network as N                                          # noqa: E402
import onfault as OF                                         # noqa: E402
import roughvol as R                                         # noqa: E402
import static as St                                          # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data" / "taiwan"
OUT = HERE / "out"
FIGURES = HERE.parents[1] / "website" / "src" / "figures"
PUBLIC = HERE.parents[1] / "website" / "public" / "data"

SCHEMA = 1

#: The two TEM segments, south first: fault_id 0 is the southern segment.
SEGMENTS = (("south", "taiwan_segment_0030_segmesh64.msh"),
            ("north", "taiwan_segment_0030_segmesh41.msh"))

#: Equirectangular projection on a sphere of this radius: km per degree of
#: latitude, and of longitude times cos(lat0). Good to a part in a thousand
#: over the 130 km here, which is far below the mesh's own 1.9 km.
R_EARTH_KM = 6371.0
KM_PER_DEG = R_EARTH_KM * math.pi / 180.0

#: Vertices this close to the surface are ON it (the files carry +-9e-13).
Z_SNAP = 1e-6

#: Floor on n_z. ``onfault.MIN_NZ`` (0.05) guards the upward-normal convention
#: against float noise on an exactly vertical band; here every facet is wound
#: up explicitly and checked, and the mesh as supplied has two slivers at
#: n_z 0.015 (dip 87-89 deg) that are kept rather than re-meshed, so the floor
#: is only what rules out an exactly vertical facet.
NZ_FLOOR = 1e-3

#: The sweep (module docstring). Azimuths are geographic, clockwise from
#: north; the fault's normal is near 110.
THETAS = (90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0)
RS = (0.0, 0.25, 0.5, 0.75, 1.0)
MODELS = ("uniform", "lithostatic")
KERNELS = ("mollified", "classical")
#: The slip rule: ``fixed`` is the scalar LCP along the tensor's rake
#: (``static.solve``), ``free`` the cone problem in which each element's
#: slip vector may turn (``static.solve_cone``). Both are run; the page
#: offers both.
RAKES = ("fixed", "free")
#: Multiples of ``lambda0``, the UNIFORM model's first failure for the tensor
#: (the lithostatic model's own first failure is a near-surface sliver at a
#: fraction of a MPa, so both models are scanned on the uniform model's
#: loads and compared at the same absolute load). To 6 lambda0: a 60-deg
#: fault under fault-normal thrust activates over a factor of five in load.
LOADS = (1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
#: The loads whose slip maps the page ships (the scan keeps them all).
SHIP_LOADS = (1.5, 2.0, 3.0, 4.0, 6.0)
#: The load the manifest's headline numbers quote.
LOAD_QUOTED = 3.0
#: The active area fraction whose load the figure reports (``lam_half``).
HALF = 0.5

MU_F = 0.6
K_MEAN = OF.MEAN_STRESS
PORE_RATIO = 0.4
DROP_RATIO = OF.DROP_RATIO                 # unused by the LCP; build_friction wants one

MODEL_LABEL = {"uniform": "uniform 150 MPa",
               "lithostatic": "lithostatic, pore ratio 0.4"}
RAKE_LABEL = {"fixed": "fixed rake: slip along the tensor's shear",
              "free": "free rake: slip vector on the Coulomb disc"}
KERNEL_LABEL = {"mollified": "mollified P0, folded (mhs)",
                "classical": "classical TDE (cutde)"}


# ------------------------------------------------------------------ mesh --
def read_gmsh(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """``(nodes (N, 3), tris (M, 3))`` from a Gmsh 4.1 ASCII file; triangles
    only, 0-based. Thirty lines rather than a meshio dependency: the format's
    ``$Nodes`` block is tags then coordinates per entity, ``$Elements`` is one
    tag and three node tags per triangle, and anything else is refused."""
    lines = path.read_text().split("\n")
    i = lines.index("$MeshFormat")
    if not lines[i + 1].startswith("4.1 0"):
        raise SystemExit(f"{path.name}: not Gmsh 4.1 ASCII ({lines[i + 1]!r})")
    i = lines.index("$Nodes") + 1
    n_blocks, n_nodes = (int(v) for v in lines[i].split()[:2])
    i += 1
    tags, coords = [], []
    for _ in range(n_blocks):
        m = int(lines[i].split()[3])
        i += 1
        tags.extend(int(lines[i + j]) for j in range(m))
        i += m
        coords.extend([float(v) for v in lines[i + j].split()] for j in range(m))
        i += m
    if len(tags) != n_nodes:
        raise SystemExit(f"{path.name}: {len(tags)} node tags for {n_nodes} nodes")
    index = {t: k for k, t in enumerate(tags)}
    i = lines.index("$Elements") + 1
    n_blocks, n_elems = (int(v) for v in lines[i].split()[:2])
    i += 1
    tris = []
    for _ in range(n_blocks):
        kind, m = (int(v) for v in lines[i].split()[2:4])
        i += 1
        if kind != 2:
            raise SystemExit(f"{path.name}: element type {kind} is not a "
                             f"3-node triangle; only triangles are read")
        tris.extend([index[int(v)] for v in lines[i + j].split()[1:4]]
                    for j in range(m))
        i += m
    if len(tris) != n_elems:
        raise SystemExit(f"{path.name}: {len(tris)} triangles for {n_elems} elements")
    return np.array(coords, float), np.array(tris, np.int64)


def project(lonlatz: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Degrees and km -> the local east-north-up frame in km (docstring)."""
    out = np.empty_like(lonlatz)
    out[:, 0] = (lonlatz[:, 0] - lon0) * KM_PER_DEG * math.cos(math.radians(lat0))
    out[:, 1] = (lonlatz[:, 1] - lat0) * KM_PER_DEG
    out[:, 2] = np.where(np.abs(lonlatz[:, 2]) < Z_SNAP, 0.0, lonlatz[:, 2])
    return out


def wind_up(tris: np.ndarray) -> np.ndarray:
    """Every triangle wound so its vertex-order normal points up."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = n[:, 2] < 0.0
    out = tris.copy()
    out[flip, 1], out[flip, 2] = tris[flip, 2], tris[flip, 1]
    return out


def junction(segments: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """The trace vertex the two segments share, in degrees: the origin."""
    (s_nodes, _), (n_nodes, _) = segments
    s_top = s_nodes[np.abs(s_nodes[:, 2]) < Z_SNAP]
    n_top = n_nodes[np.abs(n_nodes[:, 2]) < Z_SNAP]
    s_end = s_top[np.argmax(s_top[:, 1])]
    n_end = n_top[np.argmin(n_top[:, 1])]
    if np.abs(s_end[:2] - n_end[:2]).max() > 1e-9:
        raise SystemExit(f"the segments do not meet: south ends at "
                         f"{s_end[:2]}, north starts at {n_end[:2]}")
    return float(s_end[0]), float(s_end[1])


def load_lvf() -> tuple[N.Network, dict]:
    """The two segments as one Network in the local frame, plus a record of
    what was read. Gated here: no vertex above the surface, every normal up
    and off vertical (``NZ_FLOOR``), and cutde's frame equal to tdcs's
    (``onfault.frame_gate``)."""
    segs = [read_gmsh(DATA / f) for _, f in SEGMENTS]
    lon0, lat0 = junction(segs)
    tris, fid, per = [], [], []
    for i, ((name, fname), (nodes, idx)) in enumerate(zip(SEGMENTS, segs)):
        xyz = project(nodes, lon0, lat0)
        t = wind_up(xyz[idx])
        tris.append(t)
        fid.append(np.full(t.shape[0], i, np.int64))
        edges = np.concatenate([np.linalg.norm(t[:, 1] - t[:, 0], axis=1),
                                np.linalg.norm(t[:, 2] - t[:, 1], axis=1),
                                np.linalg.norm(t[:, 0] - t[:, 2], axis=1)])
        nrm = tdcs.unit_normals(t)
        dip = np.degrees(np.arccos(nrm[:, 2]))
        frame = tdcs.slip_frame(t)
        strike_from_x = np.degrees(np.arctan2(frame[:, 0, 1], frame[:, 0, 0]))
        top = (np.abs(t[:, :, 2]) < Z_SNAP)
        trace_edges = []
        for p, q in ((0, 1), (1, 2), (2, 0)):
            on = top[:, p] & top[:, q]
            trace_edges.append(np.linalg.norm(t[on, p] - t[on, q], axis=1))
        trace_km = float(np.concatenate(trace_edges).sum())
        per.append(dict(name=name, file=fname, n=int(t.shape[0]),
                        n_vertices=int(nodes.shape[0]),
                        lon_range=[float(nodes[:, 0].min()), float(nodes[:, 0].max())],
                        lat_range=[float(nodes[:, 1].min()), float(nodes[:, 1].max())],
                        depth_km=float(-t[:, :, 2].min()),
                        trace_km=trace_km,
                        n_trace=int(top.any(axis=1).sum()),
                        dip_deg=[float(np.percentile(dip, 10)), float(np.median(dip)),
                                 float(np.percentile(dip, 90))],
                        strike_from_x_deg=float(np.median(strike_from_x)),
                        strike_azimuth_deg=float((90.0 - np.median(strike_from_x)) % 360.0),
                        edge_km=[float(edges.min()), float(np.median(edges)),
                                 float(edges.max())],
                        area_km2=float(0.5 * np.linalg.norm(
                            np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())))
    tris = np.concatenate(tris)
    fid = np.concatenate(fid)
    if np.any(tris[:, :, 2] > 0.0):
        raise SystemExit("a vertex lies above the surface after the snap")
    nrm = tdcs.unit_normals(tris)
    if nrm[:, 2].min() < NZ_FLOOR:
        raise SystemExit(f"a facet is vertical: n_z {nrm[:, 2].min():.4f} < {NZ_FLOOR}")
    wound = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 2]
    if not np.all(wound > 0.0):
        raise SystemExit("a facet is still wound downward")
    OF.frame_gate(tris)
    edges = np.concatenate([np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1),
                            np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1),
                            np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)])
    h = float(np.median(edges))
    meta = dict(kind="taiwan", source="Taiwan Earthquake Model segments 64 and 41",
                files=[f for _, f in SEGMENTS], lon0=lon0, lat0=lat0,
                km_per_deg=KM_PER_DEG, h_km=h, eps_km=0.1 * h,
                frame="x east, y north, z up, origin at the shared trace vertex",
                faults=per, min_nz=float(nrm[:, 2].min()), nz_floor=NZ_FLOOR)
    net = N.Network(
        tris=tris, fault_id=fid,
        length=np.array([p["trace_km"] for p in per]),
        width=np.array([p["depth_km"] / math.sin(math.radians(p["dip_deg"][1]))
                        for p in per]),
        azimuth=np.array([p["strike_from_x_deg"] for p in per]),
        dip=np.array([p["dip_deg"][1] for p in per]),
        center=np.stack([tris[fid == i].reshape(-1, 3).mean(axis=0)
                         for i in range(len(per))]),
        h=h, eps=0.1 * h, meta=meta)
    return net, meta


def describe(net: N.Network, meta: dict) -> None:
    print(f"  LVF: {net.n_elements} elements, origin lon {meta['lon0']:.5f} lat "
          f"{meta['lat0']:.5f}, h {meta['h_km']:.2f} km, eps {meta['eps_km']:.3f} km")
    for p in meta["faults"]:
        print(f"    {p['name']:5s} {p['file']}: {p['n']:4d} elements, trace "
              f"{p['trace_km']:.1f} km, to {p['depth_km']:.0f} km, dip "
              f"{p['dip_deg'][0]:.0f}/{p['dip_deg'][1]:.0f}/{p['dip_deg'][2]:.0f}, "
              f"strike N{p['strike_azimuth_deg']:.0f}E, edges {p['edge_km'][0]:.2f}/"
              f"{p['edge_km'][1]:.2f}/{p['edge_km'][2]:.2f} km, {p['area_km2']:.0f} km2",
              flush=True)


# --------------------------------------------------------------- tensors --
#: The strike-slip end member with its compression axis along +x, so that the
#: family below shares one compression axis: LOAD_TENSORS' strike-slip tensor
#: has it at -45 deg from x (friction.py), hence the rotation.
_SS_X = Fr.rotate_about_vertical(Fr.LOAD_TENSORS["strike_slip"], 45.0)


def tensor(theta: float, r: float) -> np.ndarray:
    """The unit deviator with its most compressive axis horizontal at
    geographic azimuth ``theta`` and regime ``r`` (module docstring),
    tension-positive as ``friction`` reads it."""
    if not 0.0 <= r <= 1.0:
        raise ValueError(f"r must lie in [0, 1]; got {r}")
    D = (1.0 - r) * Fr.LOAD_TENSORS["thrust"] + r * _SS_X
    w = np.linalg.eigvalsh(D)
    D = D / (0.5 * (w[-1] - w[0]))
    return Fr.rotate_about_vertical(D, 90.0 - float(theta))


def friction(net: N.Network, theta: float, r: float, model: str) -> Fr.Friction:
    return Fr.build_friction(
        net, regime=tensor(theta, r), mu_mean=MU_F, drop_ratio=DROP_RATIO,
        sigma_n_mode=model, pore_ratio=PORE_RATIO, mean_stress_ratio=K_MEAN)


def rake_deg(fric: Fr.Friction) -> np.ndarray:
    """Aki-Richards rake of the fixed slip direction: 0 left-lateral, 90
    reverse, 180 right-lateral (tdcs's dip row points up-dip)."""
    return np.degrees(np.arctan2(fric.slip_dir[:, 1], fric.slip_dir[:, 0]))


# ------------------------------------------------------------- commands --
def cmd_assemble(a) -> int:
    net, meta = load_lvf()
    describe(net, meta)
    t0 = time.time()
    Km, info = K.assemble(net, mu=OF.MU, lam=OF.LAM, workers=a.workers, verbose=True)
    print(f"  matrix {Km.shape} ready in {time.time() - t0:.0f} s (key {info['key']})",
          flush=True)
    return 0


def solve_any(P, d, g, lam, warm):
    """PGS, and the active-set Newton solver where PGS does not converge
    (an indefinite sym(-P) is possible on a mesh this uneven)."""
    sol = St.solve(P, d, g, lam, warm=warm)
    if sol.converged:
        return sol, 0
    sol = St.solve_newton(P, d, g, lam, warm=warm)
    return sol, 1


def cmd_run(a) -> int:
    net, meta = load_lvf()
    describe(net, meta)
    n = net.n_elements
    areas = net.areas()
    asum = areas.sum()
    Km, info = K.assemble(net, mu=OF.MU, lam=OF.LAM, workers=1, verbose=True)
    Kc = OF.classical_matrix(net.tris)
    mats = {"mollified": Km, "classical": Kc}
    shear = {k: np.ascontiguousarray(v[:, :, :2, :]) for k, v in mats.items()}
    nR, nk, nm = len(RAKES), len(KERNELS), len(MODELS)
    nt, nr, nl = len(THETAS), len(RS), len(LOADS)
    loads = np.array(LOADS)
    act = np.full((nR, nk, nm, nt, nr, n), np.nan)
    slip = np.zeros((nR, nk, nm, nt, nr, nl, n), np.float32)
    srake = np.full((nR, nk, nm, nt, nr, nl, n), np.nan, np.float32)
    frac = np.zeros((nR, nk, nm, nt, nr, nl))
    mom = np.zeros((nR, nk, nm, nt, nr, nl))
    sweeps = np.zeros((nR, nk, nm, nt, nr, nl), np.int64)
    method = np.zeros((nR, nk, nm, nt, nr, nl), np.int8)
    converged = np.zeros((nR, nk, nm, nt, nr, nl), bool)
    lam0 = np.zeros((nm, nt, nr))
    lam_tensile = np.zeros((nm, nt, nr))
    never = np.zeros((nm, nt, nr), np.int64)
    rake = np.zeros((nt, nr, n))
    dcoef = np.zeros((nt, nr, n))
    tau_mag = np.zeros((nt, nr, n))
    sn_load = np.zeros((nt, nr, n))
    mineig = np.zeros((nk, nt, nr))
    branch = np.zeros((nk, nt, nr))
    iq = LOADS.index(LOAD_QUOTED)
    t_all = time.time()
    for it, theta in enumerate(THETAS):
        for ir, r in enumerate(RS):
            fr0 = friction(net, theta, r, MODELS[0])
            rake[it, ir] = rake_deg(fr0)
            tau_mag[it, ir] = fr0.tau_load_mag
            sn_load[it, ir] = fr0.sigma_n_load
            dcoef[it, ir] = C.coefficients(fr0)[0]
            P = {}
            for ik, kern in enumerate(KERNELS):
                P[kern] = K.project(mats[kern], fr0.slip_dir, fr0.slip_dir)
                S = -0.5 * (P[kern] + P[kern].T)
                mineig[ik, it, ir] = float(np.linalg.eigvalsh(S)[0])
                branch[ik, it, ir] = float(np.median(K.branching_ratio(P[kern])))
            for im, model in enumerate(MODELS):
                fric = fr0 if im == 0 else friction(net, theta, r, model)
                d, g_s, _ = C.coefficients(fric)
                ff = fric.first_failure_lambda()
                lam0[im, it, ir] = float(np.nanmin(ff))
                never[im, it, ir] = int(np.isinf(ff).sum())
                lam_tensile[im, it, ir] = fric.max_lambda_compressive()
                lam_ref = lam0[0, it, ir]           # the uniform model's
                for ik, kern in enumerate(KERNELS):
                    for iR, rule in enumerate(RAKES):
                        warm = None
                        t0 = time.time()
                        for il, f in enumerate(LOADS):
                            lam = f * lam_ref
                            if rule == "fixed":
                                sol, how = solve_any(P[kern], d, g_s, lam, warm)
                                warm = sol.delta
                                mag = sol.delta
                                rk = np.where(mag > 0.0, rake[it, ir], np.nan)
                                sw, ok = sol.sweeps, sol.converged
                            else:
                                sol = St.solve_cone(shear[kern], fric.tau_load,
                                                    fric.strength_at(lam), lam, warm=warm)
                                warm = sol.slip
                                mag, rk, how = sol.magnitude, sol.rake_deg, 0
                                sw, ok = sol.sweeps, sol.converged
                            new = (mag > 0.0) & np.isnan(act[iR, ik, im, it, ir])
                            act[iR, ik, im, it, ir, new] = f
                            slip[iR, ik, im, it, ir, il] = mag
                            srake[iR, ik, im, it, ir, il] = rk
                            frac[iR, ik, im, it, ir, il] = float((areas * (mag > 0.0)).sum() / asum)
                            mom[iR, ik, im, it, ir, il] = float((areas * mag).sum() * OF.MU * 1e18)
                            sweeps[iR, ik, im, it, ir, il] = sw
                            method[iR, ik, im, it, ir, il] = how
                            converged[iR, ik, im, it, ir, il] = ok
                        print(f"  az {theta:3.0f} r {r:.2f} {model:11s} {kern:9s} {rule:5s}: "
                              f"lam0 {1e3 * lam0[im, it, ir]:6.1f} MPa, active "
                              f"{frac[iR, ik, im, it, ir, iq]:.3f} at {LOAD_QUOTED} lam0, "
                              f"{frac[iR, ik, im, it, ir, -1]:.3f} at {LOADS[-1]}; moment "
                              f"{mom[iR, ik, im, it, ir, iq]:.3e}; {nl} loads {time.time() - t0:4.0f} s, "
                              f"sweeps {int(sweeps[iR, ik, im, it, ir].sum())}, not converged "
                              f"{int((~converged[iR, ik, im, it, ir]).sum())}, min eig "
                              f"{mineig[ik, it, ir]:+.3f}", flush=True)
    print(f"  sweep done in {(time.time() - t_all) / 60:.1f} min; not converged: "
          f"{int((~converged).sum())} of {converged.size}", flush=True)
    OUT.mkdir(exist_ok=True)
    path = OUT / "taiwan_sweep.npz"
    np.savez_compressed(
        path, thetas=np.array(THETAS), rs=np.array(RS), loads=loads,
        models=np.array(MODELS), kernels=np.array(KERNELS), rakes=np.array(RAKES),
        tris=net.tris, fault_id=net.fault_id, areas=areas,
        act=act, slip=slip, srake=srake, frac=frac, mom=mom, sweeps=sweeps,
        method=method, converged=converged, lam0=lam0, lam_tensile=lam_tensile,
        never=never, rake=rake, dcoef=dcoef, tau_mag=tau_mag, sn_load=sn_load,
        mineig=mineig, branch=branch, key=info["key"],
        meta=json.dumps(meta), git=str(K._git()))
    print(f"  saved {path}", flush=True)
    return 0


# --------------------------------------------------------------- figure --
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction="out", length=3)


def _letter(ax, s):
    ax.text(0.98, 0.98, s, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold")


def lam_half(frac: np.ndarray, loads: np.ndarray, lam_ref: float,
             level: float = HALF) -> float:
    """The absolute load (GPa) at which the active area fraction first
    reaches ``level``, interpolated linearly between scan loads; NaN when the
    scan never gets there."""
    above = np.flatnonzero(frac >= level)
    if above.size == 0:
        return float("nan")
    j = int(above[0])
    if j == 0:
        return float(loads[0] * lam_ref)
    f0, f1 = frac[j - 1], frac[j]
    t = (level - f0) / (f1 - f0) if f1 > f0 else 1.0
    return float((loads[j - 1] + t * (loads[j] - loads[j - 1])) * lam_ref)


def rotation_deg(srake_free: np.ndarray, rake_fixed: np.ndarray) -> np.ndarray:
    """Per-element angle between the free rake and the tensor's, in [0, 180],
    NaN where the free rule has the element locked."""
    d = np.abs((srake_free - rake_fixed + 180.0) % 360.0 - 180.0)
    return np.where(np.isfinite(srake_free), d, np.nan)


def figure(z, out: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    thetas, rs, loads = z["thetas"], z["rs"], z["loads"]
    areas = z["areas"]
    w = areas / areas.sum()
    nt, nr = len(thetas), len(rs)
    iq = list(loads).index(LOAD_QUOTED)
    cmap = plt.get_cmap("viridis")
    cols = [cmap(0.1 + 0.8 * i / (nr - 1)) for i in range(nr)]
    half = np.full((2, 2, 2, nt, nr), np.nan)          # rake, kernel, model
    for iR in range(2):
        for ik in range(2):
            for im in range(2):
                for it in range(nt):
                    for ir in range(nr):
                        half[iR, ik, im, it, ir] = 1e3 * lam_half(
                            z["frac"][iR, ik, im, it, ir], loads, z["lam0"][0, it, ir])
    ratio = z["mom"][1, 0, :, :, :, iq] / z["mom"][0, 0, :, :, :, iq]   # free / fixed, mollified
    rot_med = np.full((2, nt, nr), np.nan)
    for im in range(2):
        for it in range(nt):
            for ir in range(nr):
                rot = rotation_deg(z["srake"][1, 0, im, it, ir, iq], z["rake"][it, ir])
                on = np.isfinite(rot) & (z["slip"][0, 0, im, it, ir, iq] > 0)
                if on.any():
                    o = np.argsort(rot[on]); ww = w[on][o]
                    rot_med[im, it, ir] = rot[on][o][np.searchsorted(np.cumsum(ww), 0.5 * ww.sum())]
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.2))
    ax = axes[0, 0]
    for ir, r in enumerate(rs):
        ax.plot(thetas, 1e3 * z["lam0"][0, :, ir], "-o", color=cols[ir], ms=3,
                label=f"r = {r:.2f}")
    ax.set_ylim(0, None)
    ax.set_ylabel("first failure $\\lambda_0$ (MPa)")
    ax.set_title("uniform 150 MPa: the load that fails the best element", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2,
              title="regime r (0 thrust, 1 strike-slip)", title_fontsize=8)
    _style(ax); _letter(ax, "a")
    for im, model in enumerate(MODELS):
        ax = axes[0, 1 + im]
        for ir, r in enumerate(rs):
            ax.plot(thetas, half[1, 0, im, :, ir], "-o", color=cols[ir], ms=3)
            ax.plot(thetas, half[0, 0, im, :, ir], "--", color=cols[ir], lw=0.9)
        ax.set_ylim(0, None)
        ax.set_ylabel("load with half the area active (MPa)")
        ax.set_title(f"{MODEL_LABEL[model]}: solid free rake, dashed fixed", fontsize=9)
        _style(ax); _letter(ax, "bc"[im])
    ax = axes[0, 3]
    for ir, r in enumerate(rs):
        ax.plot(thetas, ratio[0, :, ir], "-o", color=cols[ir], ms=3)
        ax.plot(thetas, ratio[1, :, ir], ":", color=cols[ir], lw=1.0)
    ax.axhline(1, color="0.75", lw=0.8)
    ax.set_ylim(0.98, None)
    ax.set_ylabel(f"moment, free / fixed rake, at {LOAD_QUOTED:g} $\\lambda_0$")
    ax.set_title("solid uniform, dotted lithostatic", fontsize=9)
    _style(ax); _letter(ax, "d")
    ax = axes[1, 0]
    for ir, r in enumerate(rs):
        ax.plot(thetas, [float(np.sum(w * z["rake"][it, ir])) for it in range(nt)],
                "-o", color=cols[ir], ms=3)
    ax.axhline(90, color="0.75", lw=0.8)
    ax.set_ylim(0, 180)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_ylabel("area-mean rake the tensor imposes (deg)")
    ax.set_title("0 left-lateral, 90 reverse, 180 right-lateral", fontsize=9)
    _style(ax); _letter(ax, "e")
    ax = axes[1, 1]
    for ir, r in enumerate(rs):
        ax.plot(thetas, [float(np.sum(w * (z["dcoef"][it, ir] <= 0.0)))
                         for it in range(nt)], "-o", color=cols[ir], ms=3)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("area that can never fail")
    ax.set_title("normal stress rising faster than shear, at any load", fontsize=9)
    _style(ax); _letter(ax, "f")
    ax = axes[1, 2]
    it = int(np.argmin(np.abs(thetas - 130.0)))
    for ir, r in enumerate(rs):
        ax.plot(loads, z["frac"][1, 0, 0, it, ir], "-", color=cols[ir])
        ax.plot(loads, z["frac"][1, 0, 1, it, ir], ":", color=cols[ir])
    ax.set_ylim(0, 1.02)
    ax.set_xlim(loads[0], loads[-1])
    ax.set_xlabel("load  $\\lambda / \\lambda_0$")
    ax.set_ylabel("active area fraction (free rake)")
    ax.set_title(f"$S_{{Hmax}}$ at {thetas[it]:.0f}: solid uniform, dotted lithostatic",
                 fontsize=9)
    _style(ax); _letter(ax, "g")
    ax = axes[1, 3]
    for ir, r in enumerate(rs):
        ax.plot(thetas, rot_med[0, :, ir], "-o", color=cols[ir], ms=3)
        ax.plot(thetas, rot_med[1, :, ir], ":", color=cols[ir], lw=1.0)
    ax.set_ylim(0, None)
    ax.set_ylabel(f"area-median rake rotation at {LOAD_QUOTED:g} $\\lambda_0$ (deg)")
    ax.set_title("free rake against the tensor's, on the slipping area", fontsize=9)
    _style(ax); _letter(ax, "h")
    for ax in axes.ravel():
        if ax is not axes[1, 2]:
            ax.set_xticks(thetas)
    for ax in axes[1]:
        if ax is not axes[1, 2]:
            ax.set_xlabel("$S_{Hmax}$ azimuth (deg, clockwise from north)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return half, ratio, rot_med


def cmd_figure(a) -> int:
    z = dict(np.load(OUT / "taiwan_sweep.npz", allow_pickle=False))
    out = pathlib.Path(a.out) / "taiwan_sweep.png"
    half, ratio, rot_med = figure(z, out)
    loads = list(z["loads"])
    iq = loads.index(LOAD_QUOTED)
    print(f"  {out}")
    for iR, rule in enumerate(RAKES):
        for im, model in enumerate(MODELS):
            print(f"  {rule} rake, {model}: load with half the area active (MPa), mollified, az x r")
            for it, th in enumerate(z["thetas"]):
                print(f"    az {th:3.0f}: " + " ".join(f"{half[iR, 0, im, it, ir]:6.0f}"
                                                      for ir in range(len(z["rs"]))))
    print(f"  lam0 (MPa), uniform, az x r")
    for it, th in enumerate(z["thetas"]):
        print(f"    az {th:3.0f}: " + " ".join(f"{1e3 * z['lam0'][0, it, ir]:6.0f}"
                                              for ir in range(len(z["rs"]))))
    print(f"  moment free / fixed at {LOAD_QUOTED} lam0, mollified uniform, az x r")
    for it, th in enumerate(z["thetas"]):
        print(f"    az {th:3.0f}: " + " ".join(f"{ratio[0, it, ir]:6.3f}"
                                              for ir in range(len(z["rs"]))))
    print(f"  area-median rotation (deg) at {LOAD_QUOTED} lam0, mollified uniform, az x r")
    for it, th in enumerate(z["thetas"]):
        print(f"    az {th:3.0f}: " + " ".join(f"{rot_med[0, it, ir]:6.2f}"
                                              for ir in range(len(z["rs"]))))
    for iR, rule in enumerate(RAKES):
        good = np.isfinite(half[iR, 0]) & np.isfinite(half[iR, 1])
        q = half[iR, 1][good] / half[iR, 0][good]
        print(f"  {rule}: kernels, half-active load classical/mollified {np.median(q):.3f} "
              f"median, {q.min():.3f}-{q.max():.3f}; max |active fraction difference| "
              f"{np.abs(z['frac'][iR, 0] - z['frac'][iR, 1]).max():.3f}; moment ratio "
              f"classical/mollified at {LOAD_QUOTED} lam0 "
              f"{np.median(z['mom'][iR, 1, ..., iq] / z['mom'][iR, 0, ..., iq]):.3f}")
    print(f"  min eig sym(-P): mollified {z['mineig'][0].min():+.3f}, classical "
          f"{z['mineig'][1].min():+.3f}; newton solves {int((z['method'] == 1).sum())}; "
          f"not converged {int((~z['converged']).sum())}")
    return 0


# -------------------------------------------------------------- payload --
def case_key(kern: str, model: str, theta: float, r: float) -> str:
    return f"{kern}.{model}.az{theta:03.0f}.r{r:.2f}"


def cmd_payload(a) -> int:
    z = dict(np.load(OUT / "taiwan_sweep.npz", allow_pickle=False))
    meta = json.loads(str(z["meta"]))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tris, fid, areas = z["tris"], z["fault_id"], z["areas"]
    loads = [float(v) for v in z["loads"]]
    thetas, rs = [float(v) for v in z["thetas"]], [float(v) for v in z["rs"]]
    if (~z["converged"]).any():
        raise SystemExit(f"{int((~z['converged']).sum())} solves did not converge; "
                         "refused")
    arrays: dict = {}

    def put(name, arr, dtype):
        buf = np.ascontiguousarray(np.asarray(arr).astype(dtype)).tobytes()
        (out / f"{name}.bin").write_bytes(buf)
        arrays[name] = dict(dtype=np.dtype(dtype).name, shape=list(np.shape(arr)),
                            bytes=len(buf), sha256=hashlib.sha256(buf).hexdigest()[:16])

    def wmedian(v, mask):
        if not mask.any():
            return None
        o = np.argsort(v[mask]); ww = w[mask][o]
        return float(v[mask][o][np.searchsorted(np.cumsum(ww), 0.5 * ww.sum())])

    def wquantile(v, mask, q):
        if not mask.any():
            return None
        o = np.argsort(v[mask]); ww = w[mask][o]
        return float(v[mask][o][min(np.searchsorted(np.cumsum(ww), q * ww.sum()), o.size - 1)])

    put("taiwan.tris", tris, np.float32)
    put("taiwan.corners", np.eye(3)[None], np.float32)
    put("taiwan.area", areas, np.float32)
    put("taiwan.fault_id", fid, np.uint8)
    w = areas / areas.sum()
    cases: dict = {}
    ship = [loads.index(f) for f in SHIP_LOADS]
    # The Coulomb criterion is linear in the load, so an element whose
    # effective normal stress the load drives through zero keeps "failing"
    # past the point where it would physically open; the page states how
    # much area that is at each load (under 1.2 % on either model by 6 lam0).
    net, _ = load_lvf()
    sig0 = {m: friction(net, thetas[0], rs[0], m).sigma_eff_0 for m in MODELS}
    for it, th in enumerate(thetas):
        for ir, r in enumerate(rs):
            put(f"taiwan.az{th:03.0f}.r{r:.2f}.rake", z["rake"][it, ir], np.float32)
            put(f"taiwan.az{th:03.0f}.r{r:.2f}.d", z["dcoef"][it, ir], np.float32)
            for im, model in enumerate(MODELS):
                tension = [float(np.sum(w * (sig0[model] + f * z["lam0"][0, it, ir]
                                             * z["sn_load"][it, ir] <= 0.0)))
                           for f in loads]
                for ik, kern in enumerate(KERNELS):
                    for iR, rule in enumerate(RAKES):
                        key = f"{rule}.{case_key(kern, model, th, r)}"
                        put(f"taiwan.{key}.act", z["act"][iR, ik, im, it, ir], np.float32)
                        for il in ship:
                            put(f"taiwan.{key}.slip{loads[il]:.2f}",
                                z["slip"][iR, ik, im, it, ir, il], np.float32)
                            if rule == "free":
                                put(f"taiwan.{key}.srake{loads[il]:.2f}",
                                    z["srake"][iR, ik, im, it, ir, il], np.float32)
                        c = dict(
                            # the load the scan multiplies: the uniform model's
                            # first failure; the model's own is beside it
                            lam0_MPa=1e3 * float(z["lam0"][0, it, ir]),
                            lam0_own_MPa=1e3 * float(z["lam0"][im, it, ir]),
                            lam_tensile_over_lam0=float(z["lam_tensile"][im, it, ir]
                                                        / z["lam0"][0, it, ir]),
                            never_fail=int(z["never"][im, it, ir]),
                            active_fraction=[float(v) for v in z["frac"][iR, ik, im, it, ir]],
                            moment_Nm=[float(v) for v in z["mom"][iR, ik, im, it, ir]],
                            max_slip_m=[1e3 * float(z["slip"][iR, ik, im, it, ir, il].max())
                                        for il in range(len(loads))],
                            lam_half_MPa=1e3 * lam_half(z["frac"][iR, ik, im, it, ir],
                                                        z["loads"], z["lam0"][0, it, ir]),
                            never_fail_area=float(np.sum(w * (z["dcoef"][it, ir] <= 0.0))),
                            tension_area=tension,
                            rake_mean_deg=float(np.sum(w * z["rake"][it, ir])),
                            min_eig_sym_negP=float(z["mineig"][ik, it, ir]),
                            branching_ratio=float(z["branch"][ik, it, ir]),
                            newton_solves=int((z["method"][iR, ik, im, it, ir] == 1).sum()),
                            sweeps=int(z["sweeps"][iR, ik, im, it, ir].sum()))
                        if rule == "free":
                            fx = z["slip"][0, ik, im, it, ir]
                            fr_ = z["slip"][1, ik, im, it, ir]
                            rot = [rotation_deg(z["srake"][1, ik, im, it, ir, il], z["rake"][it, ir])
                                   for il in range(len(loads))]
                            both = [(fx[il] > 0) & (fr_[il] > 0) for il in range(len(loads))]
                            c.update(
                                moment_vs_fixed=[float(z["mom"][1, ik, im, it, ir, il]
                                                       / z["mom"][0, ik, im, it, ir, il])
                                                 if z["mom"][0, ik, im, it, ir, il] > 0 else None
                                                 for il in range(len(loads))],
                                rotation_median_deg=[wmedian(rot[il], both[il]) for il in range(len(loads))],
                                rotation_p90_deg=[wquantile(rot[il], both[il], 0.9) for il in range(len(loads))],
                                n_reversed=[int(np.sum(np.isfinite(rot[il]) & (rot[il] > 90.0)))
                                            for il in range(len(loads))],
                                n_on=[int(np.sum((fr_[il] > 0) & (fx[il] == 0))) for il in range(len(loads))],
                                n_off=[int(np.sum((fr_[il] == 0) & (fx[il] > 0))) for il in range(len(loads))])
                        cases[key] = c
    geom = []
    for i, p in enumerate(meta["faults"]):
        t = tris[fid == i]
        g = R.surface(t)
        g["outline"] = R.rim_outline(t)
        g["name"] = p["name"]
        geom.append(g)
    (out / "taiwan_geometry.json").write_text(json.dumps(
        dict(faults=geom, units="km"), allow_nan=False))
    man = dict(
        schema=SCHEMA, units="km, GPa, degrees", git=str(z["git"]), source="taiwan.py",
        matrix_key=str(z["key"]), n_elements=int(tris.shape[0]),
        h_km=meta["h_km"], eps_km=meta["eps_km"], frame=meta["frame"],
        origin=dict(lon=meta["lon0"], lat=meta["lat0"]), km_per_deg=meta["km_per_deg"],
        faults=meta["faults"], data_source=meta["source"],
        sweep=dict(thetas=thetas, rs=rs, models=list(MODELS), kernels=list(KERNELS),
                   rakes=list(RAKES), loads=loads, ship_loads=list(SHIP_LOADS),
                   quoted_load=LOAD_QUOTED),
        friction=dict(mu=MU_F, mean_stress_ratio=K_MEAN, pore_ratio=PORE_RATIO,
                      sigma_eff_uniform_GPa=Fr.SIGMA_EFF_UNIFORM, rho_g_GPa_per_km=Fr.RHO_G),
        labels=dict(models=MODEL_LABEL, kernels=KERNEL_LABEL, rakes=RAKE_LABEL),
        tensor="most compressive axis horizontal at azimuth theta; r = 0 thrust, "
               "r = 1 strike-slip, diag(-1, r, 1 - r) between, unit maximum shear",
        rake="Aki-Richards: 0 left-lateral, 90 reverse, 180 right-lateral; under the "
             "fixed rule set per element by the tensor, under the free rule solved",
        cases=cases, arrays=arrays,
        defaults=dict(rake="free", kernel="mollified", model="uniform", theta=130.0,
                      r=0.5, load=2.0, metric="slip"))
    for c in cases.values():
        for k in ("lam_half_MPa", "lam_tensile_over_lam0"):
            if not np.isfinite(c[k]):
                c[k] = None       # half the area never active by 6 lam0; never tensile
    (out / "taiwan_manifest.json").write_text(
        json.dumps(man, indent=1, sort_keys=True, allow_nan=False) + "\n")
    total = sum(v["bytes"] for v in arrays.values())
    print(f"  {len(arrays)} arrays, {total / 1024 ** 2:.2f} MiB -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("assemble", help="the mollified matrix into the cache, in "
                                        "its own process")
    s.add_argument("--workers", type=int, default=8)
    sub.add_parser("run", help="the stress-tensor sweep, both kernels, both slip rules")
    f = sub.add_parser("figure", help="the summary figure from out/taiwan_sweep.npz")
    f.add_argument("--out", default=str(FIGURES))
    p = sub.add_parser("payload", help="out/taiwan_sweep.npz -> website/public/data")
    p.add_argument("--out", default=str(PUBLIC))
    a = ap.parse_args(argv)
    if a.cmd == "assemble":
        return cmd_assemble(a)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "figure":
        return cmd_figure(a)
    return cmd_payload(a)


if __name__ == "__main__":
    raise SystemExit(main())
