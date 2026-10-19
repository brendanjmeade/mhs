"""Continuous P2 slip on a network: the nodal twin of ``kernel`` and the solve.

WHAT CHANGES AND WHAT DOES NOT. In ``kernel.assemble`` an unknown is an
element: one slip vector per facet, collocated at its centroid, and the
on-fault condition is enforced there and read between the centroids as a
staircase whose risers are line dislocations (the on-fault page's cores read
8 % of the stress drop off, both kernels, and refinement does not move it).
Here an unknown is a SHARED NODE of ``mhs.shared_nodes``: the welded vertices
and the edge midpoints, one Cartesian slip vector each, continuous across
every edge, collocated at the nodes themselves. The matrix is
``mhs.interaction_matrix(..., order=2, layout="continuous")``, source-major
exactly as ``kernel`` stores it, so ``kernel.project``, ``project_normal``,
``refuse_if_sign_wrong``, ``static.solve``, ``coulomb_coupled`` and
``released_energy`` run on it UNCHANGED -- the LCP never asks what an unknown
is. ``friction.build_friction(net, nodes=sn)`` resolves the load on each
node's own plane, and that is the whole of the friction side.

THE RIM IS LOCKED, not solved. A crack's slip vanishes at its front, and the
continuous layout can say so: its boundary nodes inside the body
(``SharedNodes.front``) are held at zero through ``static.solve``'s
``include`` mask, which costs one skipped index. Nodes on a surface trace
(``SharedNodes.trace``, z = 0) stay free -- a fault that breaks the surface
slips there. A splay's rooted edge is a front too: the host's slip runs
through the junction line, the splay's is its own, and ``groups=fault_id``
is what keeps the two tables apart there.

ROWS, NOT SOURCES, across the pool. A source chunk owns a column slice only
in the element-major layout; a shared node's column is written by every
element touching it. So the matrix is built by ``mhs.parallel.by_rows``,
each worker seeing every source for a block of receiver nodes, and the
per-source hoisted work is repeated once per block -- negligible at the
block sizes it chooses. Measured serial cost at P2 is ~110 us per
observer-source pair against ~40 at P0, and a disc has about twice as many
nodes as elements, so the matrix costs some six times the P0 one.

DEFINITENESS IS NOT INHERITED, and at production size it is lost. On the
on-fault page's system at eps = 0.1 h the uncoupled nodal operator's
symmetric part has 37 negative eigenvalues out of 13,940 (fault A alone
-13, B -34, C -52, against a median of +240): smooth modes, vertex and
midpoint values correlating +0.99, of two kinds. Dropping the 207 nodes
within 2 eps of the free surface leaves 14 -- that band is a smeared source
straddling z = 0, the effect that makes the trace nodes too soft -- and
those 14 are broad body modes on each fault alone (A 6, B 3, C 2, down to
-13), the small-eps/h collocation asymmetry the full-space work saw at
a/h >= 16; the junction is not the cause. The Coulomb LCP on it has no
stable solution: projected Gauss-Seidel overflows, ``static.solve_newton``
thrashes. At eps = 0.2 h the body modes are gone and four remain: three on
C's nodes within 2 eps of the surface (-6.6, -3.8, -0.4) and one
long-wavelength mode spanning A and C at -0.85 of +128, which the P0
operators carry on this system as well (mollified -1.2, classical -2.5 of
+200, one mode each; PGS converges on them regardless). The same operator is
definite on the selfcheck's a/h = 8 discs (+4.8 uncoupled, +1.8 coupled) and
solves cleanly. The remedy, measured on the gate's disc and on this
system's fault B alone and not built, is the SYMMETRIC GALERKIN form (test
= trial = the P2 Lagrange basis, the analytic rows at quadrature points of
each element contracted with the test weights): its symmetric part is
definite at every eps (disc: +0.24 of a median 1.3-1.7, the same at 0.1 h
and 0.05 h, where collocation reads +13 and -33; B alone at 0.1 h: +0.053
of the median, the physical h/a mode, +6.4 against collocation's -34 in
traction units, stable to 3 % between 9 and 25 quadrature points, +0.046
with the mu N coupling, the band within 2 eps of the surface +0.66 of its
own median), so the LCP is the KKT system of a convex QP: on B at 0.1 h,
where collocation's PGS overflows and Newton stalls, Galerkin's PGS
converges in 99 sweeps and Newton in 6 to the same active set, and the
Coulomb residual between the nodes stays at 0.27 % of the strength
(median, p90 1.1 %). Its cost is k rows per element (9-point rule: 4.4x
this assembly) and a solve error about twice nodal collocation's at the
same eps (3.4 % against 1.6 % at 0.1 h, 1.6 % against 1.1 % at 0.05 h,
first order in eps either way), recoverable by lowering eps, which no
longer costs definiteness; the coefficients enter mass-weighted (M d, M g:
a vertex row carries no load, its basis function having zero mean); and
the trace row comes out too STIFF (0.22 m against 0.51 the row below on
B), the top elements' quadrature points reading the smeared trace slip as
P0's first row does -- the same O(eps/depth) effect as the too-soft
collocation node, with the opposite sign. Three cheaper ideas measured
out: positive test functions (hats and edge bubbles, the same test space)
give the same solution with an indefinite symmetric part (-6 at 0.05 h),
so averaged rows improve the degree, not the guarantee; reading a
near-surface node's row 2 eps deeper along dip moves it most of an element
away, drops the self-term 13x and takes the page's count from 4 modes to
163; and the mass-weighted collocation matrix M K, definite on the disc
but with four negative band modes on B, and whose symmetrization ruins the
solve (12 % and 40 %), because collocation's asymmetry is content, not
error.

WHAT THE ENERGY FORM EXPOSED, measured on fault B alone (the page's load,
uncoupled Coulomb; the session scratchpad holds the full numbers). On the
surface-breaking B the forms disagree by 20-30 % in moment: classical P0
1.00, mollified P0 0.86 / 0.94 / 0.98 of it at eps 0.1 / 0.05 / 0.025 h
(first order to classical), continuous-P2 collocation at 0.2 h 1.07 (trace
vertices 2.2 m, too soft), Galerkin P2 0.82 at both 0.1 h and 0.05 h with
its trace at half the row below, Galerkin P0 0.76 / 0.71. THE SAME FAULT
BURIED 3 km, where no smeared source straddles the free surface, puts every
form within 3 % of classical P0 for a uniform stress drop (mollified P0
+1.1 %, P2 collocation +0.1 %, P2 Galerkin +1.3 %, P0 Galerkin -3.1 %),
with the energy forms definite (softest mode +8.9 / +9.4 of medians 250 /
234 in traction units; classical P0 +4.4 of 219) and P2 collocation still
indefinite (-6.6: the body modes). So the disagreement on the
surface-breaking fault is THE KERNEL AT THE FREE SURFACE, not the form. On
a long vertical strike-slip fault breaking the surface (antiplane exact
solution, D/h 8) classical P0 is within 1 % at every depth and mollified
P0 within 4 % / 2 % at 0.1 / 0.05 h; continuous-P2 collocation is unusable
at every eps (trace 50x / 5x / 0.27x exact at 0.2 / 0.1 / 0.05 h: the
trace band's negative modes); and the Galerkin P2, within 2-8 % in the
body at 0.05 h with 9 points, gets WORSE with 100 points on the
surface-touching elements and stops converging in eps (trace 3x). The
mollified half-space kernel is O(1/eps) wrong within ~eps of the free
surface for a blob straddling it (the above-surface part's image lands
inside the body): an O(1), eps-independent error once integrated, which the
P0 centroid 2.5 eps deep feels only as O(eps/depth). It is not local
either: it couples into the fault's softest mode and moves the moment of
the whole fault -- integrating B's surface-touching elements only below a
band and rescaling by area takes the moment from 0.82 to 0.97 (2 eps) or
1.31 (3 eps) of classical and makes the operator indefinite, so a band is
not a fix. THE FIX IS IN THE KERNEL, and its mechanism is measured: the
regularised half-space problem must take as its body force the smeared slip
FOLDED back into the body (the part above z = 0 replaced by its mirror
image: mass-preserving, the even extension that is exact in antiplane
strain), never the kernel formula continued above the surface. Its solution
is the full-space field of the element PLUS its mirror element with
reflected slip (both clq closed forms) plus a Boussinesq correction for the
normal traction the symmetric pair leaves on z = 0 -- zero for antiplane
strain, O(1) for dip slip. On the vertical strike-slip reference at 0.1 h,
with only the surface-touching row switched to element + mirror and mhs
everywhere else: collocation P2 goes from 5.0x exact at the trace (body 1.5
to 7.3x) to 0.978 at every depth, Galerkin P2 from 3.3x to 0.984, and the
shear traction the solution leaves on the free surface from 25 to 0.008 of
the stress drop, a check that needs no reference solution. The Boussinesq
term is built the same way (``mhs.kernels.fold``): the symmetric
pair's normal traction on z = 0, twice the element's own elastic sigma_zz
there, convolved over the surface with the classical Boussinesq point-load
stress on a receiver-centred polar grid that resolves the smeared crease
along the trace, the singular part subtracted at the receiver's foot point
and restored by Love's closed-form disk field. Gated on a buried element,
where element + mirror + Boussinesq must reproduce mhs's lifted image
(1.9e-2 at 12 eps depth, 1.6e-3 at 1 km) and cutde's classical kernel as
well as mhs does (5e-3). On fault B (dip 33 deg median, dip slip,
surface-breaking) at 0.1 h against classical P0: the uniform-drop moment
goes from 0.90 (Galerkin) / 0.95 (collocation) of classical to 1.013 for
both; the page-load Coulomb LCP from 0.82 (Galerkin; collocation diverges)
to 1.008 / 1.021 with every depth bin within 3 % (collocation 5 %); the
Galerkin form stays positive definite (+0.075 of 1.65) and the collocation
form indefinite (-10.9 of 239); the traction left on the free surface
drops from 14-255 to 0.2 of the drop. Only the surface-touching row needs
the fold, and only at receivers within one element height of the surface
(the lift below changes nothing on the vertical reference). So on
surface-breaking faults too, the Galerkin form with the folded kernel
delivers definiteness and accuracy together, and the collocation form does
not. The fold is now the kernel's default (``mhs.kernels.fold``, gated by
``verify_fold``), so every entry point this module calls folds the top row;
the page's numbers above were measured with the lift and change on a rerun.

THE READOUT between the nodes is ``mhs.traction_matrix(pts, nrm, tris, ...,
layout="continuous")`` contracted with the nodal slip, which is what
:func:`traction` chunks across a pool; one value per ELEMENT, for the page,
is :meth:`SharedNodes.element_mean` of a nodal field, or the traction at the
centroid, which is now a readout between collocation points rather than the
solver's own residual.
"""
from __future__ import annotations

import functools
import hashlib
import json
import pathlib
import time

import numpy as np

import mhs
from mhs import defaults as mhs_defaults
from mhs.parallel import by_rows, pinned_blas

import friction as Fr
import kernel as K
import static as St

RECEIVER, SOURCE = K.RECEIVER, K.SOURCE
N_A, N_B = K.N_A, K.N_B
BYTES_PER_PAIR = K.BYTES_PER_PAIR

#: Observer chunk for the readout pool: traction_matrix in the continuous
#: layout is 72 B per (observer, node) pair, and a node count is ~2x the
#: element count, so 500 cells against 14k nodes is 500 MB in a worker.
CHUNK = 500

_W: dict = {}


def table(net) -> mhs.SharedNodes:
    """The network's shared nodes, each fault welded on its own."""
    return mhs.shared_nodes(net.tris, groups=net.fault_id)


def free_mask(sn: mhs.SharedNodes, locked=None) -> np.ndarray:
    """The unknowns the solve may move: everything but the front, or
    everything but ``locked`` when the caller has a reason."""
    return ~(sn.front if locked is None else np.asarray(locked, bool))


def required_bytes(n_nodes: int) -> int:
    return int(n_nodes) ** 2 * BYTES_PER_PAIR


def cache_key(tris, fault_id, mu, lam, eps) -> str:
    """Keyed on the vertices AND the groups: the same mesh welded as one
    fault or as several is two different matrices."""
    t = np.ascontiguousarray(np.asarray(tris, float))
    g = np.ascontiguousarray(np.asarray(fault_id, np.int64))
    payload = {"geometry_sha256": hashlib.sha256(t.tobytes()).hexdigest(),
               "groups_sha256": hashlib.sha256(g.tobytes()).hexdigest(),
               "n": int(t.shape[0]), "mu": mu, "lam": lam, "eps": eps,
               "receiver": list(RECEIVER), "source": list(SOURCE),
               "layout": "continuous P2, source_major_(n_src,n_b,n_a,n_rec)",
               "kernel": K.fold_tag()}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _assemble_raw(tris, sn, mu, lam, eps, workers) -> np.ndarray:
    """``(n_rec, n_src, n_a, n_b)`` at the nodes, straight from the library."""
    mat = mhs.Material(mu=mu, lam=lam)
    call = functools.partial(mhs.interaction_matrix, tris, mat, eps, order=2,
                             layout="continuous", shared=sn,
                             receiver=RECEIVER, source=SOURCE)
    if workers in (0, 1):
        return call()
    return by_rows(call, sn.n, workers=workers)


def assemble(net, sn: mhs.SharedNodes | None = None, *, mu: float = 30.0,
             lam: float = 30.0, eps: float | None = None,
             workers: int | None = None,
             cache_dir: pathlib.Path | None = None, mmap: bool = False,
             verbose: bool = True):
    """The source-major NODAL interaction matrix for ``net``, from cache or
    fresh: ``(K, sn, info)`` with ``K[j, b, a, i]`` the traction at node
    ``i`` along direction ``a`` of its frame from unit slip along direction
    ``b`` of node ``j``'s frame. The cache lives beside ``kernel``'s."""
    sn = table(net) if sn is None else sn
    eps = float(net.eps if eps is None else eps)
    n = sn.n
    key = cache_key(net.tris, net.fault_id, mu, lam, eps)
    cache_dir = (pathlib.Path(__file__).resolve().parent / "cache"
                 if cache_dir is None else pathlib.Path(cache_dir))
    npy, side = cache_dir / f"N_{key}.npy", cache_dir / f"N_{key}.json"
    if npy.exists() and side.exists():
        info = json.loads(side.read_text())
        if verbose:
            print(f"  cache hit {npy.name}  ({info['bytes'] / 1024 ** 3:.2f} "
                  f"GiB, assembled in {info['assemble_s']:.1f} s)")
        Km = np.load(npy, mmap_mode="r" if mmap else None)
        if Km.shape != (n, N_B, N_A, n):
            raise RuntimeError(f"cached matrix {npy.name} has shape {Km.shape}, "
                               f"expected {(n, N_B, N_A, n)}: corrupt, delete it")
        return Km, sn, info
    need = required_bytes(n)
    if verbose:
        print(f"  assembling {n} x {n} nodes x {N_A} x {N_B} = "
              f"{need / 1024 ** 3:.2f} GiB (+ the same again for the "
              f"transpose), {sn.n_tri} elements")
    was = K.raise_ceiling(n, quiet=not verbose)
    t0 = time.time()
    try:
        raw = _assemble_raw(net.tris, sn, mu, lam, eps, workers)
    finally:
        if was is not None:
            mhs_defaults.MATRIX_MAX_BYTES = was
    Km = K.to_source_major(raw)
    del raw
    dt = time.time() - t0
    K.refuse_if_sign_wrong(Km)
    info = {"key": key, "n_nodes": n, "n_elements": int(sn.n_tri),
            "n_vertices": int(sn.n_vertices), "n_front": int(sn.front.sum()),
            "n_trace": int(sn.trace.sum()), "bytes": int(Km.nbytes),
            "assemble_s": dt, "us_per_pair": 1e6 * dt / (n * sn.n_tri),
            "mu_GPa": mu, "lam_GPa": lam, "eps_km": eps, "workers": workers,
            "receiver": list(RECEIVER), "source": list(SOURCE),
            "layout": "continuous P2; (n_src, n_b, n_a, n_rec), C-contiguous",
            "git": K._git(), "net": net.meta}
    if verbose:
        print(f"  assembled in {dt:.1f} s = {info['us_per_pair']:.2f} us per "
              f"node-element pair on {workers} workers")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, Km)
    side.write_text(json.dumps(info, indent=1, sort_keys=True, default=str))
    if verbose:
        print(f"  cached -> {npy.name}")
    return Km, sn, info


def operators(Kmat: np.ndarray, fric, free: np.ndarray) -> dict:
    """``P``, ``Nrm``, the coupled operator and its definiteness ON THE FREE
    ROWS -- the locked nodes are not unknowns, so they do not vote."""
    P = K.project(Kmat, fric.slip_dir, fric.slip_dir)
    Nrm = K.project_normal(Kmat, fric.slip_dir)
    P_eff, d, g = St.coulomb_coupled(P, Nrm, fric)
    f = np.flatnonzero(free)
    sub = P_eff[np.ix_(f, f)]
    mineig = float(np.linalg.eigvalsh(-0.5 * (sub + sub.T)).min())
    subP = P[np.ix_(f, f)]
    mineig_p = float(np.linalg.eigvalsh(-0.5 * (subP + subP.T)).min())
    return dict(P=P, Nrm=Nrm, P_eff=P_eff, d=d, g=g, mineig=mineig,
                mineig_uncoupled=mineig_p,
                B=float(K.branching_ratio(P)[f].mean()))


def _init(tris, sn, slip, eps, mu, lam):
    _W.update(tris=tris, sn=sn, slip=slip, eps=float(eps),
              mat=mhs.Material(mu=mu, lam=lam))


def _chunk(args):
    pts, nrm = args
    T = mhs.traction_matrix(pts, nrm, _W["tris"], _W["mat"], _W["eps"],
                            order=2, layout="continuous", shared=_W["sn"])
    return np.einsum("oigk,gk->oi", T, _W["slip"])


def traction(pts, nrm, tris, sn, slip_nodes, workers, *, eps, mu=30.0,
             lam=30.0, log=True) -> np.ndarray:
    """Elastic traction from the nodal slip on every point, in a pool pinned
    in the parent (``onfault``'s rule). ``workers`` of 0 or 1 runs here."""
    import concurrent.futures as cf
    import multiprocessing as mp
    npt = pts.shape[0]
    # strided blocks, for the reason onfault.interleave gives
    n_parts = min(npt, max(1, -(-npt // CHUNK), 4 * max(1, int(workers or 1))))
    splits = [np.arange(i, npt, n_parts) for i in range(n_parts)]
    parts = [(pts[idx], nrm[idx]) for idx in splits]
    out = np.empty((npt, 3))
    if workers in (0, 1):
        _init(tris, sn, slip_nodes, eps, mu, lam)
        for idx, part in zip(splits, parts):
            out[idx] = _chunk(part)
        return out
    t0 = time.time()
    with pinned_blas(1):
        with cf.ProcessPoolExecutor(max_workers=workers,
                                    mp_context=mp.get_context("spawn"),
                                    initializer=_init,
                                    initargs=(tris, sn, slip_nodes, eps, mu,
                                              lam)) as ex:
            done = 0
            for i, got in enumerate(ex.map(_chunk, parts, chunksize=1)):
                out[splits[i]] = got
                done += got.shape[0]
                if log and (i % 10 == 0 or done == npt):
                    el = time.time() - t0
                    rate = done / max(el, 1e-9)
                    print(f"    continuous {done:8d}/{npt}  {el / 60:6.1f} min"
                          f"  eta {(npt - done) / max(rate, 1e-9) / 60:6.1f} min"
                          f"  {el * 1e6 / max(done * sn.n_tri, 1):5.1f} "
                          f"us/pair", flush=True)
    return out


def metrics(t: np.ndarray, node: np.ndarray, z: dict, coupled: bool) -> dict:
    """``onfault.metrics`` at points attributed to NODES: ``z`` carries the
    nodal friction arrays and the node frames' slip directions."""
    lam = float(z["lam"])
    s_cart = z["s_cart"]
    n_hat = z["normals"]
    dts = np.einsum("oi,oi->o", t, s_cart[node])
    dsn = -np.einsum("oi,oi->o", t, n_hat[node])
    tau_s = lam * z["tau_load_mag"][node] + dts
    sig_n = z["sigma_eff_0"][node] + lam * z["sigma_n_load"][node]
    if coupled:
        sig_n = sig_n + dsn
    return dict(S=tau_s - z["mu"][node] * sig_n, dsn=dsn, dts=dts)
