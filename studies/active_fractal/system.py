"""Surface-breaking rough faults with smooth rims and conformal junctions.

The on-fault page's system (``onfault.py``), as a mesher. Three things the
lattice disc of :func:`network.disc_mesh` cannot do, and how each is done:

RIMS ARE CURVES, NOT STAIRCASES. The lattice disc keeps the triangles whose
centroid falls inside the circle, so its rim is a staircase of lattice edges,
and the roughness then wiggles that staircase out of plane. Here the outline
is sampled ON the circle, and on the chords where the circle is cut, at a
spacing of h, and the interior is a constrained Delaunay triangulation
(Shewchuk's Triangle, through the ``triangle`` package) with the boundary
pre-split and the ``YY`` switch set, so no segment is ever subdivided and every
rim vertex lies on the curve it samples to round-off.

THE FREE SURFACE IS A CHORD. A fault that breaks the surface is its disc cut
by v = 0 in the fault's own down-dip coordinate: the trace is a straight
boundary polyline whose vertices have z = 0 exactly, set rather than computed,
and no triangle straddles z = 0, which is the one thing mhs refuses.

A JUNCTION IS A SHARED POLYLINE. A splay whose lower edge lies on another
fault is its disc cut by the junction chord, and the host receives the same
chord as an interior constraint. The chord's vertices are computed once, in
3-D, and written into both meshes, so the two tilings share them bit for bit
and the host's edges along the chord are the splay's boundary edges there.

ROUGHNESS FADES TO ZERO AT EVERY LINE. The displacement is the published
one-wavelength field along the fault normal (:func:`network.smooth_field`,
its amplitude solved for the requested facet slope exactly as
:func:`network.rough_disc` does) times a taper that vanishes on the outline
and on the junction, so a rim vertex stays on its curve, a trace vertex at
z = 0, a junction vertex shared. The taper is wider at a junction (12 h
against 6 h at a rim) because two rough surfaces meeting at 20 degrees would
otherwise cross within a kilometre of the line at a three-sigma peak; the
side gate below decides, not that estimate.

A STADIUM IS A DISC PULLED APART. A spec with ``run=L`` is the disc with
its two halves moved L/2 apart along strike and joined by straight runs, so a
fault can be made longer along strike than down dip without changing its
radius, its roughness or its cut by the surface. The rims are still curves,
the trace is still a chord, and the same taper, roughness and gates apply.
A splay is never a stadium (its junction chord is computed for a disc).

Every fault strikes along y and dips toward +x at ``dip`` degrees, with its
trace line at ``(x_t, y_t, 0)``: in-plane coordinates are u along strike and
v down-dip, ``X(u, v) = (x_t + v cos dip, y_t + u, -v sin dip)``, and the
upward normal is ``(sin dip, 0, cos dip)``, the frame ``tdcs`` and cutde agree
on. A fault may also be given by ``centre=(x, y, z)`` as the lattice discs
were; that is converted to a trace and a centre distance, and a disc whose
circle does not reach the surface is simply not cut.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

import network

#: Taper width, in h, over which the roughness fades to zero at a rim or a
#: trace, and at a junction line. See the module docstring for the 12.
TAPER_OVER_H = 6.0
JUNCTION_TAPER_OVER_H = 12.0

#: Triangle's area ceiling, as a multiple of the equilateral area at edge h.
#: Triangle fills a quality mesh with triangles well under its ceiling, so a
#: ceiling of the equilateral area gives a median edge of about 0.8 h; this
#: factor is calibrated so the median edge of a 6 km disc comes out at h.
AREA_FACTOR = 1.55

#: Minimum angle asked of Triangle. 30 is the classical quality bound; with
#: ``YY`` (no Steiner points on any segment) it is still met here because the
#: outline's corners, where a chord meets an arc, are all obtuse.
MIN_ANGLE_DEG = 30.0

#: Tolerance, in h, for a rim vertex to count as ON its circle or chord.
RIM_TOL_OVER_H = 1e-9


def frame(dip_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(strike, down_dip, normal)`` unit vectors for a fault striking along
    y and dipping toward +x; the normal points up, and ``down_dip x strike``
    equals it, so a triangle wound (down_dip, strike) is wound upward."""
    s, c = math.sin(math.radians(dip_deg)), math.cos(math.radians(dip_deg))
    return (np.array([0.0, 1.0, 0.0]), np.array([c, 0.0, -s]),
            np.array([s, 0.0, c]))


def place(uv: np.ndarray, trace, dip_deg: float) -> np.ndarray:
    """``(u, v)`` on the fault plane -> 3-D, km."""
    _, d, _ = frame(dip_deg)
    uv = np.asarray(uv, float)
    out = np.empty((uv.shape[0], 3))
    out[:, 0] = trace[0] + uv[:, 1] * d[0]
    out[:, 1] = trace[1] + uv[:, 0]
    out[:, 2] = uv[:, 1] * d[2]
    return out


def unplace(xyz: np.ndarray, trace, dip_deg: float) -> np.ndarray:
    """3-D -> ``(u, v)`` for points ON the fault plane."""
    xyz = np.asarray(xyz, float)
    c = math.cos(math.radians(dip_deg))
    return np.stack([xyz[:, 1] - trace[1], (xyz[:, 0] - trace[0]) / c], axis=1)


# ---------------------------------------------------------------- outline --
@dataclasses.dataclass(frozen=True)
class Outline:
    """A closed loop in ``(u, v)``: ``verts[i] -> verts[i+1]`` is segment i,
    whose tag is ``"arc"``, ``"trace"`` or ``"junction"``."""
    verts: np.ndarray
    tags: tuple
    a: float
    v_c: float
    v_cut: float | None
    run: float = 0.0

    @property
    def centres(self) -> np.ndarray:
        """``(k, 2)`` centres of the arcs: one for a disc, two for a stadium."""
        if self.run > 0.0:
            return np.array([[0.5 * self.run, self.v_c], [-0.5 * self.run, self.v_c]])
        return np.array([[0.0, self.v_c]])

    def off_circle(self) -> float:
        """The largest distance (km) of an arc vertex from the nearer circle."""
        ai = self.vertices_on("arc")
        rr = np.linalg.norm(self.verts[ai][:, None, :] - self.centres[None, :, :],
                            axis=2)
        return float(np.abs(rr - self.a).min(axis=1).max())

    @property
    def segments(self) -> np.ndarray:
        m = self.verts.shape[0]
        return np.stack([np.arange(m), (np.arange(m) + 1) % m], axis=1)

    def vertices_on(self, tag: str) -> np.ndarray:
        """Indices of the vertices that are an endpoint of a ``tag`` segment."""
        seg = self.segments[[t == tag for t in self.tags]]
        return np.unique(seg.ravel())


def outline(a: float, v_c: float, h: float, v_cut: float | None = None, *,
            run: float = 0.0) -> Outline:
    """The disc of radius ``a`` centred at ``(0, v_c)``, cut by the surface
    ``v = 0`` when the circle crosses it and by ``v = v_cut`` when given,
    sampled at spacing about ``h``: arc vertices exactly on the circle, chord
    vertices exactly on their line (``v`` set to 0 or ``v_cut``, not computed).
    With ``run > 0`` the two half-discs sit ``run`` apart along strike and
    straight runs join them (tag ``"run"``), a stadium; the arcs are then on
    the circles about ``(+-run / 2, v_c)``.

    The loop runs counter-clockwise in ``(u, v)``: from the surface on the
    +u side down the +u arc, across the junction chord toward -u, up the -u
    arc to the surface, and back along the trace.
    """
    if not (a > 0.0 and h > 0.0 and v_c >= 0.0 and run >= 0.0):
        raise ValueError(f"need a, h > 0 and v_c, run >= 0; got {a}, {h}, "
                         f"{v_c}, {run}")
    cut_s = v_c < a
    cut_j = v_cut is not None
    if cut_j and not (0.0 < v_cut < v_c + a):
        raise ValueError(f"v_cut = {v_cut} is not inside the disc's reach "
                         f"(0, {v_c + a})")
    if cut_j and run > 0.0:
        raise ValueError("a splay is a disc: its junction chord is computed "
                         "for one; a stadium cannot attach")
    r = 0.5 * run                              # the arc centres sit at +-r

    def P(t, cu):
        return np.array([cu + a * math.cos(t), v_c + a * math.sin(t)])

    def arc(t0, t1, cu):
        n = max(1, int(round(a * (t1 - t0) / h)))
        t = t0 + (t1 - t0) * np.arange(n) / n
        return np.stack([cu + a * np.cos(t), v_c + a * np.sin(t)], axis=1)

    def chord(p0, p1):
        n = max(1, int(round(float(np.linalg.norm(p1 - p0)) / h)))
        s = np.arange(n) / n
        return p0 + s[:, None] * (p1 - p0)

    if not cut_s and not cut_j and run == 0.0:
        n = max(3, int(round(2.0 * math.pi * a / h)))
        t = 2.0 * math.pi * np.arange(n) / n
        verts = np.stack([a * np.cos(t), v_c + a * np.sin(t)], axis=1)
        return Outline(verts, ("arc",) * n, a, v_c, None)

    pieces = []
    if cut_s:
        al = math.asin(v_c / a)                # surface at theta = -al, pi + al
    if cut_j:
        be = math.asin((v_cut - v_c) / a)      # junction at theta = be, pi - be
    if cut_s and cut_j:
        if not be > -al:
            raise ValueError("the junction chord lies above the surface")
        pieces = [(arc(-al, be, 0.0), "arc"),
                  (chord(P(be, 0.0), P(math.pi - be, 0.0)), "junction"),
                  (arc(math.pi - be, math.pi + al, 0.0), "arc"),
                  (chord(P(math.pi + al, 0.0), P(-al, 0.0)), "trace")]
    elif cut_s and run == 0.0:                # one arc: the disc, unchanged
        pieces = [(arc(-al, math.pi + al, 0.0), "arc"),
                  (chord(P(math.pi + al, 0.0), P(-al, 0.0)), "trace")]
    elif cut_s:
        pieces = [(arc(-al, 0.5 * math.pi, r), "arc"),
                  (chord(P(0.5 * math.pi, r), P(0.5 * math.pi, -r)), "run"),
                  (arc(0.5 * math.pi, math.pi + al, -r), "arc"),
                  (chord(P(math.pi + al, -r), P(-al, r)), "trace")]
    elif cut_j:
        pieces = [(arc(math.pi - be, 2.0 * math.pi + be, 0.0), "arc"),
                  (chord(P(be, 0.0), P(math.pi - be, 0.0)), "junction")]
    else:                                      # a buried stadium
        pieces = [(arc(-0.5 * math.pi, 0.5 * math.pi, r), "arc"),
                  (chord(P(0.5 * math.pi, r), P(0.5 * math.pi, -r)), "run"),
                  (arc(0.5 * math.pi, 1.5 * math.pi, -r), "arc"),
                  (chord(P(1.5 * math.pi, -r), P(1.5 * math.pi, r)), "run")]
    verts = np.concatenate([p for p, _ in pieces])
    tags = tuple(t for p, t in pieces for _ in range(p.shape[0]))
    out = Outline(verts, tags, a, v_c, v_cut, run)
    # chord endpoints are shared with the arcs and carry the arc's round-off:
    # put every chord vertex exactly on its line
    verts[out.vertices_on("trace"), 1] = 0.0
    if cut_j:
        verts[out.vertices_on("junction"), 1] = float(v_cut)
    return out


# -------------------------------------------------------------------- cdt --
def cdt(verts: np.ndarray, segments: np.ndarray, h: float, *,
        min_angle: float = MIN_ANGLE_DEG) -> tuple[np.ndarray, np.ndarray]:
    """Constrained Delaunay triangulation of the region bounded by
    ``segments`` (plus any interior constraints among them), with no Steiner
    point on any segment and a quality bound. Returns ``(verts2, tris)``;
    the input vertices come first, unchanged, which is gated here because the
    junction depends on it."""
    import triangle
    area = AREA_FACTOR * (math.sqrt(3.0) / 4.0) * h * h
    out = triangle.triangulate(
        dict(vertices=np.asarray(verts, float),
             segments=np.asarray(segments, np.int32)),
        f"pq{min_angle:g}a{area:.12g}YYQ")
    v2, t2 = np.asarray(out["vertices"], float), np.asarray(out["triangles"], np.int64)
    m = verts.shape[0]
    if v2.shape[0] < m or not np.array_equal(v2[:m], verts):
        raise RuntimeError("Triangle did not keep the input vertices first and "
                           "unchanged; the junction cannot be shared")
    return v2, t2


# -------------------------------------------------------------- roughness --
def _point_segment_distance(p: np.ndarray, s0: np.ndarray, s1: np.ndarray
                            ) -> np.ndarray:
    """``(m, k)`` distances from points ``p`` to segments ``s0 -> s1``."""
    d = s1 - s0                                            # (k, 2)
    dd = (d * d).sum(axis=1)
    t = ((p[:, None, :] - s0[None, :, :]) * d[None, :, :]).sum(axis=2) / dd
    t = np.clip(t, 0.0, 1.0)
    q = s0[None, :, :] + t[:, :, None] * d[None, :, :]
    return np.linalg.norm(p[:, None, :] - q, axis=2)


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def taper(uv: np.ndarray, lines: list[tuple[np.ndarray, np.ndarray, float]]
          ) -> np.ndarray:
    """``min_k smoothstep(d_k / w_k)`` over the polylines ``(s0, s1, width)``:
    exactly 0 on every line, 1 beyond every width."""
    T = np.ones(uv.shape[0])
    for s0, s1, w in lines:
        d = _point_segment_distance(uv, s0, s1).min(axis=1)
        T = np.minimum(T, smoothstep(d / w))
    return T


def roughen(uv: np.ndarray, idx: np.ndarray, lines, *, rms_slope: float,
            corr: float, a: float, seed: int, n_modes: int = 64
            ) -> tuple[np.ndarray, float, np.ndarray]:
    """``(w, amp, f)``: the normal displacement of every vertex in km, the
    amplitude, and the unit field. The amplitude is solved on the UNTAPERED
    field, so the interior carries the requested facet slope and the taper
    only removes it near the lines; the realized slope is reported, not
    assumed. Refuses the same inputs :func:`network.rough_disc` refuses."""
    if rms_slope < 0.0:
        raise ValueError(f"rms_slope must be >= 0; got {rms_slope!r}")
    if corr > 2.0 * a:
        raise ValueError(f"corr = {corr} exceeds the disc diameter {2 * a}: "
                         f"a rigid tilt, not roughness")
    if rms_slope == 0.0:
        return np.zeros(uv.shape[0]), 0.0, np.zeros(uv.shape[0])
    rng = np.random.default_rng(seed)
    f = network.smooth_field(rng, uv, corr, n_modes=n_modes)
    s1 = network._facet_slope_rms(uv, idx, f)
    if not s1 > 0.0:
        raise RuntimeError("the roughness field is flat on this mesh")
    amp = float(rms_slope) / s1
    return amp * taper(uv, lines) * f, amp, f


# ---------------------------------------------------------------- system --
def _spec_trace(f: dict) -> tuple[tuple[float, float], float]:
    """``((x_t, y_t), v_c)`` from a spec given either way."""
    if "centre" in f:
        x, y, z = (float(c) for c in f["centre"])
        s, c = math.sin(math.radians(f["dip"])), math.cos(math.radians(f["dip"]))
        v_c = -z / s
        return (x - v_c * c, y), v_c
    v_c = f.get("v_c")
    return ((float(f["trace"][0]), float(f["trace"][1])),
            None if v_c is None else float(v_c))


def junction_line(host: dict, splay: dict) -> dict:
    """Where the splay's plane meets the host's: both strike along y, so the
    line is ``x = x_J, z = z_J``. Refused unless it is below the surface and
    down-dip of the splay's trace, which is what makes it the splay's LOWER
    edge."""
    (xh, _), _ = _spec_trace(host)
    (xs, _), _ = _spec_trace(splay)
    th, ts = (math.tan(math.radians(host["dip"])),
              math.tan(math.radians(splay["dip"])))
    if abs(ts - th) < 1e-12:
        raise ValueError(f"{splay['name']} and {host['name']} are parallel")
    x_j = (xs * ts - xh * th) / (ts - th)
    z_j = -(x_j - xs) * ts
    if not (z_j < 0.0 and x_j > xs):
        raise ValueError(
            f"{splay['name']} meets {host['name']} at x = {x_j:.3f}, "
            f"z = {z_j:.3f}, which is not below the surface down-dip of "
            f"{splay['name']}'s trace: not a splay")
    return dict(x=x_j, z=z_j,
                v_splay=(x_j - xs) / math.cos(math.radians(splay["dip"])),
                v_host=(x_j - xh) / math.cos(math.radians(host["dip"])),
                angle_deg=abs(splay["dip"] - host["dip"]))


def build_system(spec, h: float, *, rms_slope: float, corr: float,
                 eps: float | None = None, min_nz: float = 0.05,
                 clearance_over_h: float = 2.0,
                 taper_over_h: float = TAPER_OVER_H,
                 junction_taper_over_h: float = JUNCTION_TAPER_OVER_H
                 ) -> tuple[list[dict], network.Network]:
    """The faults of ``spec`` as one :class:`network.Network`, gated.

    Each entry: ``name, a, dip, seed`` and either ``trace=(x, y), v_c`` or
    ``centre=(x, y, z)``; ``attach=<host name>`` makes it a splay whose lower
    edge lies on the host. Returns the per-fault records (3-D vertices and
    indices, the outline, what moved) and the network.
    """
    eps = 0.1 * h if eps is None else float(eps)
    by_name = {f["name"]: f for f in spec}
    for f in spec:
        if f.get("attach") and f["attach"] not in by_name:
            raise ValueError(f"{f['name']} attaches to unknown {f['attach']}")
        if f.get("attach") and by_name[f["attach"]].get("attach"):
            raise ValueError("a splay cannot attach to a splay")

    # 1. outlines, and the junction chords computed once in 3-D
    rec: dict[str, dict] = {}
    host_constraints: dict[str, list] = {f["name"]: [] for f in spec}
    for f in spec:
        trace, v_c = _spec_trace(f)
        j = None
        v_cut = None
        if f.get("attach"):
            j = junction_line(by_name[f["attach"]], f)
            v_cut = j["v_splay"]
            if v_c is None:                 # a splay's default: mid-way
                v_c = 0.5 * v_cut
        if v_c is None:
            raise ValueError(f"{f['name']}: give v_c (or centre), or attach")
        o = outline(f["a"], v_c, h, v_cut, run=float(f.get("run", 0.0)))
        r = dict(name=f["name"], spec=f, trace=trace, v_c=v_c, outline=o,
                 junction=j)
        if j is not None:
            ji = o.vertices_on("junction")
            order = np.argsort(o.verts[ji, 0])           # along +u
            ji = ji[order]
            pts3 = place(o.verts[ji], trace, f["dip"])   # the shared points
            pts3[:, 0] = j["x"]                          # exactly on the line
            pts3[:, 2] = j["z"]
            r["junction_pts"] = pts3
            r["junction_outline_idx"] = ji
            j.update(half_width=float(o.verts[ji, 0].max()),
                     n_segments=int(ji.size - 1))
            host_constraints[f["attach"]].append((f["name"], pts3))
        rec[f["name"]] = r

    # 2. meshes, in 2-D, then placed; junction vertices snapped to the shared
    #    3-D points in both meshes
    for f in spec:
        r = rec[f["name"]]
        o = r["outline"]
        verts, segs = o.verts.copy(), o.segments
        lines = []
        for s0, s1, tag in zip(verts[segs[:, 0]], verts[segs[:, 1]], o.tags):
            w = (junction_taper_over_h if tag == "junction" else taper_over_h) * h
            lines.append((s0, s1, w))
        extra = []
        for sname, pts3 in host_constraints[f["name"]]:
            uv = unplace(pts3, r["trace"], f["dip"])
            m0 = verts.shape[0]
            verts = np.concatenate([verts, uv])
            k = uv.shape[0]
            segs = np.concatenate([segs, np.stack([np.arange(m0, m0 + k - 1),
                                                   np.arange(m0 + 1, m0 + k)], 1)])
            lines.append((uv[:-1], uv[1:], junction_taper_over_h * h))
            extra.append((sname, np.arange(m0, m0 + k), pts3))
        # group the lines by polyline: taper() wants (s0 array, s1 array, w)
        grouped = {}
        for s0, s1, w in lines:
            s0, s1 = np.atleast_2d(s0), np.atleast_2d(s1)
            g = grouped.setdefault(w, [[], []])
            g[0].append(s0)
            g[1].append(s1)
        lines = [(np.concatenate(a_), np.concatenate(b_), w)
                 for w, (a_, b_) in grouped.items()]
        v2, t2 = cdt(verts, segs, h)
        xyz = place(v2, r["trace"], f["dip"])
        t2 = t2[:, ::-1]                       # wound upward, see frame()
        # the trace sits at z = 0 exactly; the shared points are written in
        tr = o.vertices_on("trace")
        xyz[tr, 2] = 0.0
        if r["junction"] is not None:
            xyz[r["junction_outline_idx"]] = r["junction_pts"]
        for sname, idxs, pts3 in extra:
            xyz[idxs] = pts3
        r.update(uv=v2, idx=t2, xyz_flat=xyz, lines=lines,
                 host_of=[(sname, idxs) for sname, idxs, _ in extra],
                 trace_idx=tr, boundary_idx=np.arange(o.verts.shape[0]))

    # 3. roughness along the normal, tapered to zero on every line
    for f in spec:
        r = rec[f["name"]]
        w, amp, field = roughen(r["uv"], r["idx"], r["lines"],
                                rms_slope=rms_slope, corr=corr, a=f["a"],
                                seed=f["seed"])
        _, _, n_hat = frame(f["dip"])
        xyz = r["xyz_flat"] + w[:, None] * n_hat
        xyz[r["trace_idx"], 2] = 0.0
        r.update(xyz=xyz, w=w, amp_km=amp, field=field,
                 tris=xyz[r["idx"]], height_rms=float(w.std()),
                 slope_rms=network._facet_slope_rms(r["uv"], r["idx"], w))
        n = np.cross(r["tris"][:, 1] - r["tris"][:, 0],
                     r["tris"][:, 2] - r["tris"][:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        cosang = np.clip(np.abs(n @ n_hat), 0.0, 1.0)
        r["tilt_rms_deg"] = float(np.sqrt((np.degrees(np.arccos(cosang)) ** 2).mean()))
        r["n"] = int(r["idx"].shape[0])
        r["median_edge"] = float(np.median(np.linalg.norm(
            r["tris"] - np.roll(r["tris"], 1, axis=1), axis=2)))

    # 4. gates
    tol = RIM_TOL_OVER_H * h
    for f in spec:
        r = rec[f["name"]]
        o = r["outline"]
        off = o.off_circle()
        if off > tol:
            raise SystemExit(f"{f['name']}: an arc vertex is {off:.3e} km off "
                             f"its circle")
        if np.any(r["xyz"][r["trace_idx"], 2] != 0.0):
            raise SystemExit(f"{f['name']}: a trace vertex is not at z = 0")
        if np.any(r["xyz"][:, 2] > 0.0):
            k = int(np.argmax(r["xyz"][:, 2]))
            raise SystemExit(
                f"{f['name']}: roughness lifted vertex {k} to z = "
                f"{r['xyz'][k, 2]:+.4f} km above the surface; widen the taper "
                f"or lower the slope")
        share = network.edge_sharing(r["idx"])
        if share.size != 3 or share[0] != 0:
            raise SystemExit(f"{f['name']}: not a manifold tiling, edge "
                             f"sharing {share}")
        nz = np.cross(r["tris"][:, 1] - r["tris"][:, 0],
                      r["tris"][:, 2] - r["tris"][:, 0])
        nz = nz[:, 2] / np.linalg.norm(nz, axis=1)
        if nz.min() < min_nz:
            raise SystemExit(f"{f['name']}: {int((nz < min_nz).sum())} facets "
                             f"have n_z < {min_nz} (min {nz.min():.4f})")
        if r["junction"] is not None:
            host = rec[f["attach"]]
            hidx = dict(host["host_of"])[f["name"]]
            mine = r["xyz"][r["junction_outline_idx"]]
            if not np.array_equal(host["xyz"][hidx], mine):
                raise SystemExit(f"{f['name']}/{host['name']}: the junction "
                                 f"vertices are not shared bit for bit")
            # the host's edges along the chord are the splay's boundary edges
            hset = set(map(tuple, np.sort(hidx[np.stack([np.arange(hidx.size - 1),
                                                         np.arange(1, hidx.size)], 1)], 1)))
            he = np.sort(np.concatenate([host["idx"][:, [0, 1]], host["idx"][:, [1, 2]],
                                         host["idx"][:, [2, 0]]]), axis=1)
            have = set(map(tuple, he))
            if not hset <= have:
                raise SystemExit(f"{host['name']} does not carry every junction "
                                 f"segment as an edge")
            # and no triangle of the host crosses the chord (the chord is a
            # constraint): every chord edge is shared by exactly two triangles
            cnt = {}
            for e in map(tuple, he):
                cnt[e] = cnt.get(e, 0) + 1
            if any(cnt[e] != 2 for e in hset):
                raise SystemExit(f"{host['name']}: a junction edge is not "
                                 f"interior to the host")

    tris = np.ascontiguousarray(np.concatenate([rec[f["name"]]["tris"] for f in spec]))
    fault_id = np.concatenate([np.full(rec[f["name"]]["n"], i, np.int64)
                               for i, f in enumerate(spec)])
    # clearance between faults that are NOT attached; contact gate for the
    # attached ones
    attached = {(f["name"], f["attach"]) for f in spec if f.get("attach")}
    allowed = np.ones((len(spec), len(spec)), bool)
    for i, fi in enumerate(spec):
        for k, fk in enumerate(spec):
            if (fi["name"], fk["name"]) in attached or (fk["name"], fi["name"]) in attached:
                allowed[i, k] = allowed[k, i] = False
    dmin, pair = clearance(tris, fault_id, clearance_over_h * h, allowed=allowed)
    if dmin < clearance_over_h * h:
        raise SystemExit(f"faults {spec[pair[0]]['name']} and "
                         f"{spec[pair[1]]['name']} come within {dmin:.3f} km, "
                         f"under the {clearance_over_h * h:.3f} km clearance")
    contact = {}
    for f in spec:
        if not f.get("attach"):
            continue
        r, host = rec[f["name"]], rec[f["attach"]]
        off = np.setdiff1d(np.arange(r["xyz"].shape[0]), r["junction_outline_idx"])
        side, d = side_of(r["xyz"][off], host["tris"])
        if not np.all(side > 0.0):
            raise SystemExit(f"{f['name']}: {int((side <= 0).sum())} vertices "
                             f"lie on or below {host['name']}'s rough surface; "
                             f"widen the junction taper or lower the slope")
        contact[f["name"]] = dict(host=f["attach"], min_distance_km=float(d.min()))

    nz_all = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nz_all = nz_all[:, 2] / np.linalg.norm(nz_all, axis=1)
    records = [rec[f["name"]] for f in spec]
    meta = dict(
        kind="system", h=h, eps=eps, slope=rms_slope, corr_km=corr,
        taper_over_h=taper_over_h, junction_taper_over_h=junction_taper_over_h,
        faults=[dict(name=f["name"], a=f["a"], dip=f["dip"], seed=f["seed"],
                     trace=list(r["trace"]), v_c=r["v_c"],
                     attach=f.get("attach"), n=r["n"],
                     run_km=r["outline"].run,
                     trace_length_km=float(2.0 * math.sqrt(max(f["a"] ** 2 - r["v_c"] ** 2, 0.0))
                                           + r["outline"].run),
                     median_edge_km=r["median_edge"],
                     tilt_rms_deg=r["tilt_rms_deg"], height_rms_km=r["height_rms"],
                     slope_rms=r["slope_rms"], amp_km=r["amp_km"],
                     junction=r["junction"])
                for f, r in zip(spec, records)],
        clearance_km=dmin if np.isfinite(dmin) else None,
        clearance_reach_km=clearance_over_h * h,
        contact=contact, min_nz=float(nz_all.min()))
    net = network.Network(
        tris=tris, fault_id=fault_id,
        length=np.array([2.0 * f["a"] + rec[f["name"]]["outline"].run for f in spec]),
        width=np.array([2.0 * f["a"] for f in spec]),
        azimuth=np.zeros(len(spec)),
        dip=np.array([float(f["dip"]) for f in spec]),
        center=np.array([r["tris"].reshape(-1, 3).mean(0) for r in records]),
        h=h, eps=eps, meta=meta)
    return records, net


# ---------------------------------------------------------- distance gates --
def point_triangle_distance(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """``(m,)`` distance from ``p[i]`` to triangle ``tri[i]``, pairwise.

    The closest point of a triangle is either the point's projection, when
    that projection lands inside, or a point on one of the three edges; both
    are closed forms, so this is exact rather than iterated.
    """
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    n = np.cross(ab, ac)
    d00, d01, d11 = (ab * ab).sum(1), (ab * ac).sum(1), (ac * ac).sum(1)
    d20, d21 = (ap * ab).sum(1), (ap * ac).sum(1)
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    inside = (v >= 0.0) & (w >= 0.0) & (v + w <= 1.0)
    dplane = np.abs((ap * n).sum(1)) / np.linalg.norm(n, axis=1)

    def seg(s0, s1):
        d = s1 - s0
        t = np.clip(((p - s0) * d).sum(1) / (d * d).sum(1), 0.0, 1.0)
        return np.linalg.norm(p - (s0 + t[:, None] * d), axis=1)

    dedge = np.minimum(np.minimum(seg(a, b), seg(b, c)), seg(c, a))
    return np.where(inside, dplane, dedge)


def clearance(tris: np.ndarray, fault_id: np.ndarray, reach: float,
              allowed: np.ndarray | None = None
              ) -> tuple[float, tuple[int, int]]:
    """Minimum vertex-to-facet distance between facets of DIFFERENT faults,
    over the fault pairs ``allowed`` (all, by default).

    Vertex-to-facet, not centroid-to-centroid: a rough disc has relief of
    half a kilometre and it is the relief that can touch. Pairs farther than
    ``reach`` apart are not examined, so the returned minimum is exact when it
    is below ``reach`` and only a bound otherwise, which is all a gate needs.
    """
    from scipy.spatial import cKDTree
    cen = tris.mean(axis=1)
    circ = float(np.linalg.norm(tris - cen[:, None, :], axis=2).max())
    verts = tris.reshape(-1, 3)
    vf = np.repeat(fault_id, 3)
    hits = cKDTree(cen).query_ball_point(verts, reach + circ)
    vi = np.concatenate([np.full(len(h), i) for i, h in enumerate(hits)])
    fi = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits])
    keep = fault_id[fi] != vf[vi]
    if allowed is not None:
        keep &= allowed[vf[vi], fault_id[fi]]
    vi, fi = vi[keep], fi[keep]
    if vi.size == 0:
        return float("inf"), (-1, -1)
    d = point_triangle_distance(verts[vi], tris[fi])
    best = int(np.argmin(d))
    return float(d[best]), (int(vf[vi[best]]), int(fault_id[fi[best]]))


def side_of(pts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point, ``(signed offset along the nearest facet's normal,
    distance)`` to the surface ``tris``: positive on the normal's side."""
    from scipy.spatial import cKDTree
    cen = tris.mean(axis=1)
    circ = float(np.linalg.norm(tris - cen[:, None, :], axis=2).max())
    tree = cKDTree(cen)
    d_c, _ = tree.query(pts)
    side = np.empty(pts.shape[0])
    dist = np.empty(pts.shape[0])
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    for i, p in enumerate(pts):
        cand = np.asarray(tree.query_ball_point(p, d_c[i] + circ), np.int64)
        d = point_triangle_distance(np.repeat(p[None], cand.size, 0), tris[cand])
        k = cand[int(np.argmin(d))]
        dist[i] = d.min()
        side[i] = float((p - cen[k]) @ n[k])
    return side, dist
