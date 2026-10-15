#!/usr/bin/env python
"""The rough-crack von Mises volume, mollified against classical, for the site.

    python roughvol.py solve  [--out state.npz]
    python roughvol.py volume --state state.npz [--spacing 0.4] [--workers 8]

WHY TWO COMMANDS. ``kernel.assemble`` runs a spawn pool and the volume needs a
second one; on this machine a second spawn pool raised after the first has
closed deadlocks (parent at 0 % CPU, no children). So the solve saves and the
volume reloads, which also means the eighty-minute volume is not re-run to
change a colour bar.

WHAT THE PICTURE IS FOR. The companion measurement -- a head-to-head on the
interaction matrix -- found the classical triangular dislocation element as
good as the mollified kernel at every roughness tested, so this volume is NOT
a claim that mollification is needed to assemble a matrix. It is the opposite
quantity: the stress FIELD. A collocation scheme only ever asks for traction at
an element's own centroid, where the classical finite part exists; a field asks
for stress everywhere, and on a contiguous mesh with varying slip the classical
kernel leaves a residual dislocation line on every interior edge whose stress
diverges as 1/d. Those lines are mesh artefacts -- the physical crack is smooth
there -- and they are what the difference state shows.

The encoding and the manifest schema are the volume viewer's, matching
``website/scripts/make_payload.py``, because ``website/src/viz`` loads both
payloads with one reader. The schema is therefore pinned by that reader, and
``--check-schema`` compares the written manifest's keys against the shipped
``manifest.json`` so a drift fails here rather than as a blank canvas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cascade as C                                          # noqa: E402
import friction as Fr                                       # noqa: E402
import kernel as K                                          # noqa: E402
import network                                              # noqa: E402
import static as St                                         # noqa: E402
from payload import (LEVELS, NODATA, SCHEMA, ROUNDTRIP_MARGIN,  # noqa: E402
                     decode_log_u8, encode_log_u8)

MU, LAM = 30.0, 30.0             # GPa; nu = 0.25
DEPTH = -20.0                    # km, the crack's mean depth
A_KM, H_KM = 6.0, 0.5            # radius and element size
DIP_DEG = 30.0                   # the Coulomb-optimal dip for this regime
SLOPE, CORR_OVER_H, SEED = 0.25, 8.0, 1
LAM_FACTOR = 1.05                # load, as a multiple of the first failure
REGIME, DROP_RATIO, MEAN_STRESS = "thrust", 0.9, 0.6667

#: The sampled box, km. Tight on the crack: the comparison lives within a few
#: eps of the surface, so a box reaching to the free surface would spend every
#: texel on the region where the two kernels agree to one part in ten thousand.
LO = np.array([-10.0, -10.0, -27.0])
HI = np.array([10.0, 10.0, -13.0])

#: Floor on the difference array, as a fraction of its own maximum. Four
#: decades: the state arrays quantise at about 1.6 % relative, so anything
#: below this is under the payload's own resolution rather than a measurement.
DIFF_FLOOR = 1e-4

FIELD = "von_mises"
STATES = ("mollified", "classical")
DIFF = "diff_classical_minus_mollified"

_W: dict = {}


def build() -> tuple[network.RoughDisc, "network.Network"]:
    """The mesh and its Network wrapper, in one place so solve and volume agree."""
    s, c = np.sin(np.deg2rad(DIP_DEG)), np.cos(np.deg2rad(DIP_DEG))
    r = network.rough_disc(A_KM, H_KM, rms_slope=SLOPE, corr_over_h=CORR_OVER_H,
                           depth=DEPTH, normal=(s, 0.0, c), seed=SEED)
    n = r.tris.shape[0]
    net = network.Network(
        tris=r.tris, fault_id=np.zeros(n, np.int64), length=np.array([0.0]),
        width=np.array([0.0]), azimuth=np.array([0.0]), dip=np.array([0.0]),
        center=r.tris.reshape(-1, 3).mean(0)[None, :], h=H_KM, eps=0.1 * H_KM,
        meta=dict(kind="rough", slope=SLOPE, corr_over_h=CORR_OVER_H))
    return r, net


def invariants(sig: np.ndarray) -> np.ndarray:
    """von Mises, the same form the site's other payload writes."""
    tr = np.trace(sig, axis1=-2, axis2=-1) / 3.0
    dev = sig - tr[..., None, None] * np.eye(3)
    return np.sqrt(1.5 * np.einsum("...ij,...ij->...", dev, dev))


# ------------------------------------------------------------------ stage 1 --
def cmd_solve(a) -> int:
    r, net = build()
    n = r.tris.shape[0]
    print(f"  rough crack a={A_KM} h={H_KM} slope={SLOPE} corr={CORR_OVER_H}h "
          f"dip={DIP_DEG} -> {n} elements, tilt rms {r.tilt_rms_deg:.2f} deg")
    mat, _ = K.assemble(net, mu=MU, lam=LAM, workers=a.workers, verbose=False)
    fric = Fr.build_friction(net, regime=REGIME, drop_ratio=DROP_RATIO,
                             mean_stress_ratio=MEAN_STRESS)
    P = K.project(mat, fric.slip_dir, fric.slip_dir)
    del mat
    d, g_s, _ = C.coefficients(fric)
    lam0 = float(np.nanmin(fric.first_failure_lambda()))
    sol = St.solve(P, d, g_s, LAM_FACTOR * lam0)
    B = float(K.branching_ratio(P).mean())
    mineig = float(np.linalg.eigvalsh(-0.5 * (P + P.T)).min())
    del P
    slip = Fr.slip_cartesian(r.tris, fric.slip_dir, sol.delta)
    print(f"  lam/lam0 = {LAM_FACTOR}: {int(sol.active.sum())}/{n} active "
          f"({sol.active.mean():.1%}), max slip {sol.delta.max() * 1e3:.3f} m")
    print(f"  B {B:.4f}, min eig sym(-P) {mineig:.3f}")
    np.savez(a.out, tris=r.tris, slip=slip, delta=sol.delta, active=sol.active,
             eps=net.eps, h=H_KM, a=A_KM, dip=DIP_DEG, slope=SLOPE,
             tilt=r.tilt_rms_deg, lam=LAM_FACTOR * lam0, lam0=lam0, B=B,
             mineig=mineig, corr_km=r.corr_km, height_rms=r.height_rms)
    print(f"  saved {a.out}")
    return 0


# ------------------------------------------------------------------ stage 2 --
def _init(tris, slip, eps):
    import mhs
    _W.update(tris=tris, slip=slip, eps=float(eps),
              mat=mhs.Material(mu=MU, lam=LAM))


def _chunk(pts):
    import mhs
    from mhs.parallel import pinned_blas
    with pinned_blas(1):
        S = mhs.stress_matrix(pts, _W["tris"], _W["mat"], _W["eps"])
        return invariants(np.einsum("omnsk,sk->omn", S, _W["slip"]))


def mollified(pts, tris, slip, eps, workers) -> np.ndarray:
    import concurrent.futures as cf
    import multiprocessing as mp
    npt, chunk = pts.shape[0], 400
    parts = [pts[i:i + chunk] for i in range(0, npt, chunk)]
    out = np.empty(npt)
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers,
                                mp_context=mp.get_context("spawn"),
                                initializer=_init,
                                initargs=(tris, slip, eps)) as ex:
        done = 0
        for i, got in enumerate(ex.map(_chunk, parts, chunksize=1)):
            out[i * chunk:i * chunk + got.size] = got
            done += got.size
            if i % 50 == 0 or done == npt:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(f"    mollified {done:7d}/{npt}  {el / 60:5.1f} min  "
                      f"eta {(npt - done) / max(rate, 1e-9) / 60:5.1f} min",
                      flush=True)
    return out


def classical(pts, tris, slip, chunk=4000) -> np.ndarray:
    """The same field from cutde, with NO regularisation.

    The slip must be handed over in cutde's own local triad. ``from_cartesian``
    is that conversion and it is not optional: contracting a Cartesian slip
    against cutde's strain matrix silently compares two different bases, which
    on a horizontal element reads as a vanishing self-term rather than as an
    error.
    """
    import cutde.halfspace as hs
    import mhs
    from mhs import tdcs
    nu = mhs.Material(mu=MU, lam=LAM).nu_if_you_must
    loc = tdcs.from_cartesian(tris, slip)
    V6 = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    out = np.empty(pts.shape[0])
    t0 = time.time()
    for i in range(0, pts.shape[0], chunk):
        p = pts[i:i + chunk]
        E = np.asarray(hs.strain_matrix(p.copy(), tris.copy(), nu))
        Es = np.einsum("ovsk,sk->ov", E, loc)
        e = np.zeros((p.shape[0], 3, 3))
        for k, (x, y) in enumerate(V6):
            e[:, x, y] = Es[:, k]
            e[:, y, x] = Es[:, k]
        tr = e[:, 0, 0] + e[:, 1, 1] + e[:, 2, 2]
        sg = 2.0 * MU * e
        for x in range(3):
            sg[:, x, x] += LAM * tr
        out[i:i + p.shape[0]] = invariants(sg)
    print(f"    classical {pts.shape[0]} points in {(time.time()-t0)/60:.1f} min",
          flush=True)
    return out


def rim_outline(tris: np.ndarray, tol_over_h: float = 1e-9) -> list[list[float]]:
    """The crack's rim as one closed loop of vertices.

    The viewer draws ``fault.outline`` as a closed polyline, so the rim is sent
    as the ordered boundary rather than as a bounding box: a dipping rough disc
    inside its own bounding box would read as a second body in the half space.
    Boundary edges are those used by exactly one triangle, which is the same
    ``edge_sharing`` test the mesh is gated on.
    """
    tol = tol_over_h * H_KM
    key: dict = {}
    for i, T in enumerate(tris):
        for p, q in ((0, 1), (1, 2), (2, 0)):
            kp = tuple(np.round(T[p] / tol).astype(np.int64))
            kq = tuple(np.round(T[q] / tol).astype(np.int64))
            key.setdefault(tuple(sorted([kp, kq])), []).append((i, p, q))
    bnd = [v[0] for v in key.values() if len(v) == 1]
    pos = {}
    adj: dict = {}
    for i, p, q in bnd:
        kp = tuple(np.round(tris[i][p] / tol).astype(np.int64))
        kq = tuple(np.round(tris[i][q] / tol).astype(np.int64))
        pos[kp], pos[kq] = tris[i][p], tris[i][q]
        adj.setdefault(kp, []).append(kq)
        adj.setdefault(kq, []).append(kp)
    start = next(iter(adj))
    loop, prev, cur = [start], None, start
    while True:
        nxt = [v for v in adj[cur] if v != prev]
        if not nxt or nxt[0] == start:
            break
        prev, cur = cur, nxt[0]
        loop.append(cur)
    if len(loop) != len(adj):
        raise SystemExit(f"the rim is not one loop: walked {len(loop)} of "
                         f"{len(adj)} boundary vertices")
    return [[round(float(c), 4) for c in pos[k]] for k in loop]


def surface(tris: np.ndarray, tol_over_h: float = 1e-9) -> dict:
    """The whole crack as a welded vertex table plus triangle indices.

    The rim alone says where the crack is and nothing about its shape, and the
    lines this page is about are its INTERIOR edges, which a rim cannot show.
    Welded rather than 1044 loose triangles so that the drawn wireframe is the
    mesh's own edge set rather than three overlapping copies of it.

    WELDED FROM ``tris``, not taken from ``RoughDisc.verts``: that table is the
    one the roughness was applied to, which is BEFORE the rotation to the dip,
    so indexing it reproduces a horizontal crack 3.06 km away from the one the
    field was computed on. Welding the rotated triangles cannot drift, and the
    check below is then worth making.

    Refused unless the indexed mesh reproduces ``tris`` to the welding
    tolerance and every edge is shared by one or two triangles: a surface drawn
    from a table that is not the one the field was computed on would be a
    picture of a different crack.
    """
    verts, idx = network.weld_vertices(tris, tol_over_h * H_KM)
    err = float(np.abs(verts[idx] - tris).max())
    if err > 1e-6 * H_KM:
        raise SystemExit(f"the welded table does not reproduce the mesh: "
                         f"max|diff| = {err:.3e} km")
    share = network.edge_sharing(idx)
    if share.size != 3:
        raise SystemExit(f"the mesh is not manifold: edge sharing {share}")
    return {"vertices": [[round(float(c), 4) for c in v] for v in verts],
            "triangles": [[int(i) for i in t] for t in idx],
            "n_vertices": int(verts.shape[0]),
            "boundary_edges": int(share[1]), "interior_edges": int(share[2])}


def roundtrip(v, buf, meta) -> tuple[float, float]:
    """(measured, arithmetic bound) worst relative error of the quantisation."""
    back = decode_log_u8(buf, meta)
    good = np.isfinite(v) & (v > 0.0) & np.isfinite(back)
    if not good.any():
        return 0.0, 0.0
    worst = float(np.max(np.abs(back[good] - v[good]) / np.abs(v[good])))
    dec = meta["log_max"] - meta["log_min"]
    return worst, float(10.0 ** (dec / (2.0 * (LEVELS - 1))) - 1.0)


def cmd_volume(a) -> int:
    z = np.load(a.state)
    tris, slip, eps = z["tris"], z["slip"], float(z["eps"])
    r, _ = build()
    # The cheap half is always rebuilt, so a cached volume cannot be published
    # against a mesh it was not computed on.
    if not np.array_equal(r.tris, tris):
        raise SystemExit("the state's mesh is not the one build() makes now; "
                         "re-run `solve` before `volume`")
    dims = np.maximum(2, np.floor((HI - LO) / a.spacing).astype(int) + 1)
    ax = [LO[i] + a.spacing * np.arange(dims[i]) for i in range(3)]
    Z, Y, X = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")   # x fastest
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    npt = pts.shape[0]
    print(f"  grid {[int(d) for d in dims]} at {a.spacing} km = {npt} points "
          f"x {tris.shape[0]} sources = {npt * tris.shape[0] / 1e6:.1f} M pairs")
    if pts[:, 2].max() >= 0.0:
        raise SystemExit("the box reaches the free surface; every point here "
                         "must be inside the body for an all-ones region mask")

    if a.volume and pathlib.Path(a.volume).exists():
        c = np.load(a.volume)
        if not np.array_equal(c["tris"], tris):
            raise SystemExit(f"{a.volume} was computed on a different mesh")
        if not (np.array_equal(c["dims"], dims)
                and np.allclose(c["lo"], LO)
                and np.allclose(c["spacing"], a.spacing)):
            raise SystemExit(f"{a.volume} was computed on a different grid")
        got = {"mollified": c["vm_mhs"], "classical": c["vm_cutde"]}
        print(f"  reusing the field from {a.volume}")
    else:
        got = {"mollified": mollified(pts, tris, slip, eps, a.workers),
               "classical": classical(pts, tris, slip)}
        if a.volume:
            np.savez_compressed(a.volume, vm_mhs=got["mollified"],
                                vm_cutde=got["classical"], dims=dims, lo=LO,
                                spacing=np.full(3, a.spacing), tris=tris,
                                slip=slip)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = {"dims": [int(d) for d in dims],
            "origin": [float(x) for x in LO],
            "spacing": [a.spacing] * 3, "n_points": int(npt)}
    man = {"schema": SCHEMA, "grid": grid, "nodata": NODATA, "levels": LEVELS,
           "source": {"volume_run": "roughvol.py", "sampled_from": None,
                      "git": _git()},
           "states": {}, "arrays": {}}

    def put(name, buf, meta):
        (out / f"{name}.bin").write_bytes(buf)
        man["arrays"][name] = dict(meta, bytes=len(buf),
                                   sha256=hashlib.sha256(buf).hexdigest()[:16])

    # Every point is inside the body, so the mask is all ones -- written anyway
    # because the shader multiplies by it and discards below 0.5.
    rbuf = np.ones(npt, np.uint8).tobytes()
    rkey = f"region.{hashlib.sha256(rbuf).hexdigest()[:16]}"
    put(rkey, rbuf, {"encoding": "codes_u8", "filter": "nearest",
                     "values": [1]})

    checks = []
    for s in STATES:
        buf, meta = encode_log_u8(got[s])
        key = f"{s}.{FIELD}"
        put(key, buf, meta)
        man["states"][s] = {"kind": "state", "fields": {FIELD: key},
                            "region": rkey}
        checks.append((key, *roundtrip(got[s], buf, meta)))

    # The raw difference spans eleven decades, because far from the crack the
    # two kernels agree down to float noise, and 254 levels spread over eleven
    # decades spends most of them below the point where a difference is
    # information at all: a state array's own round-trip is about 1.6 %, so a
    # difference four decades under the largest is already far below what this
    # payload can represent. Under the floor the texel encodes as NODATA and
    # the shader discards it, so the difference state draws ONLY where the two
    # kernels actually part company.
    d = np.abs(got["classical"] - got["mollified"])
    floor = DIFF_FLOOR * d.max()
    drawn = d >= floor
    d = np.where(drawn, d, 0.0)
    buf, meta = encode_log_u8(d)
    key = f"{DIFF}.{FIELD}"
    put(key, buf, meta)
    man["states"][DIFF] = {"kind": "difference", "fields": {FIELD: key},
                           "region": rkey}
    checks.append((key, *roundtrip(d, buf, meta)))

    ratio = got["classical"] / np.maximum(got["mollified"], 1e-30)
    man["meta"] = {
        "mu_GPa": MU, "lam_GPa": LAM, "nu": LAM / (2.0 * (LAM + MU)),
        "eps_km": eps, "h_km": H_KM, "a_km": A_KM, "dip_deg": DIP_DEG,
        "eps_over_h": eps / H_KM, "rms_slope": SLOPE,
        "corr_over_h": CORR_OVER_H, "seed": SEED,
        "n_triangles": int(tris.shape[0]),
        "tilt_rms_deg": float(z["tilt"]), "height_rms_km": float(z["height_rms"]),
        "lam_over_lam0": LAM_FACTOR, "active_fraction": float(z["active"].mean()),
        "max_slip_m": float(z["delta"].max() * 1e3),
        "branching_ratio": float(z["B"]), "min_eig_sym": float(z["mineig"]),
        "ratio_median": float(np.median(ratio)),
        "ratio_p99": float(np.percentile(ratio, 99)),
        "ratio_max": float(ratio.max()),
        "grid_spacing_over_eps": a.spacing / eps,
        "diff_floor_fraction": DIFF_FLOOR,
        "diff_drawn_fraction": float(drawn.mean()),
        "units": "km, GPa"}

    # THE SLICES OPEN AT THE FAR WALLS, not through the middle. Centred slices
    # are what the half-space payload wants, where the fault is a plane at the
    # edge of the box; here the crack sits at the box's centre by construction,
    # so x = y = z = 0.5 puts three opaque planes through the subject and the
    # mesh is 90 % hidden on load. The camera sits at +x, -y, +z, so the far
    # walls are x low, y high, z low: the field is still drawn on three faces
    # and the crack is in the open in front of them.
    man["defaults"] = {"state": "mollified", "field": FIELD,
                       "slices": {"x": 0.04, "y": 0.96, "z": 0.04}}

    (out / "rough_geometry.json").write_text(json.dumps({
        "box": {"origin": grid["origin"], "spacing": grid["spacing"],
                "dims": grid["dims"]},
        "fault": {"outline": rim_outline(tris),
                  "n_triangles": int(tris.shape[0]),
                  **surface(tris)}}) + "\n")
    man["geometry"] = "rough_geometry.json"

    print("  round-trip, measured against the encoder's arithmetic bound:")
    bad = []
    for key, got_err, bound in sorted(checks, key=lambda c: -c[1]):
        m = man["arrays"][key]
        over = got_err > ROUNDTRIP_MARGIN * bound + 1e-12
        bad += [key] if over else []
        print(f"    {key:40s} {m['log_max'] - m['log_min']:5.2f} dec  "
              f"{got_err * 100:5.2f} % (bound {bound * 100:5.2f} %)"
              f"{'  <-- OVER BOUND' if over else ''}")
    if bad:
        raise SystemExit(f"quantisation exceeded its own bound on {bad}; the "
                         f"encoder is wrong, not the data. Do not publish.")
    man["quantisation"] = {k: {"measured_rel_err": g, "bound_rel_err": b}
                           for k, g, b in checks}

    (out / "rough_manifest.json").write_text(
        json.dumps(man, indent=1, sort_keys=True) + "\n")
    total = sum(v["bytes"] for v in man["arrays"].values())
    print(f"  {len(man['arrays'])} arrays, {total / 1024 ** 2:.2f} MiB -> {out}")
    print(f"  ratio classical/mollified: median {man['meta']['ratio_median']:.4f}, "
          f"p99 {man['meta']['ratio_p99']:.3f}, max {man['meta']['ratio_max']:.2f}")

    if a.check_schema:
        ref = json.loads((out / "manifest.json").read_text())
        for k in ("schema", "grid", "nodata", "levels", "source", "states",
                  "arrays", "geometry", "defaults"):
            if k not in man:
                raise SystemExit(f"manifest is missing {k!r}")
        miss = set(ref["grid"]) - set(man["grid"])
        if miss:
            raise SystemExit(f"grid is missing {sorted(miss)}")
        one = man["arrays"][f"{STATES[0]}.{FIELD}"]
        refone = ref["arrays"][next(k for k, v in ref["arrays"].items()
                                    if v.get("encoding") == "log10_u8")]
        miss = {k for k in ("encoding", "log_min", "log_max", "bytes",
                            "sha256") if k not in one}
        if miss:
            raise SystemExit(f"array metadata is missing {sorted(miss)}")
        print(f"  schema checks out against manifest.json "
              f"(reference array keys {sorted(refone)})")
    return 0


def _git():
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(pathlib.Path(__file__).parent),
                            "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="mesh + Coulomb solve -> state.npz")
    s.add_argument("--out", default="rough_state.npz")
    s.add_argument("--workers", type=int, default=8)
    v = sub.add_parser("volume", help="the 3-D field both ways -> the payload")
    v.add_argument("--state", required=True)
    v.add_argument("--spacing", type=float, default=0.4)
    v.add_argument("--workers", type=int, default=8)
    v.add_argument("--volume", default=None,
                   help="cache the evaluated field here, and reuse it if the "
                        "mesh and grid match")
    v.add_argument("--check-schema", action="store_true", default=True)
    v.add_argument("--out", default=str(pathlib.Path(__file__).resolve()
                                        .parents[2] / "website" / "public"
                                        / "data"))
    a = ap.parse_args(argv)
    return cmd_solve(a) if a.cmd == "solve" else cmd_volume(a)


if __name__ == "__main__":
    raise SystemExit(main())
