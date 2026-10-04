#!/usr/bin/env python
"""Generate ``mhs/kernels/_image_table.py``: the image R-family monomial table.

WHY THIS EXISTS AS A GENERATOR. The shipped package declares ``numpy`` and
``numba`` only -- the mollified Mindlin kernel cannot be built with sympy at
import time the way the oracle does it (14 s, and sympy is an ``[oracle]``
extra). So the symbolic reduction runs ONCE, here, and the result ships as
data. Generated code nobody can regenerate is unreviewable, which is why this
file is in the tree and why ``verify_image_table`` re-runs it and compares.

WHAT IS BEING REDUCED. The mollified Mindlin Green's function splits into

    G = G_direct(R1)  +  G_image(R2)  +  G_image(Q)

by which radical each term carries. The DIRECT half is clq's business and is
already closed-form. The ``Q`` half -- ``1/Q`` and ``log Q`` with
``Q = R2 - (z + z0)`` -- stays in quadrature. This generator handles the
middle one, the ``R2`` half, which ``verify_vertical_fault`` clause [e]
identifies as the family whose quadrature STAGNATES on the surface trace.

THE REDUCTION. With ``D = x - y'`` for ``y'`` the reflected source, the Mindlin
convention's ``(x, y, z + z0)`` IS ``D`` -- because ``y'_3 = -y_3`` makes
``x_3 - y'_3 = x_3 + y_3``. Hence ``z0 = D3 - z`` with ``z`` the observer's
absolute depth, a constant over the element. So every R-family term becomes

    coeff(z, mu, nu) * D1^a D2^b D3^c / R2^n,      n odd

which is exactly the tensor moment ``lift()`` builds from a moment table on the
REFLECTED triangle (reflection is an isometry, so integrating over the triangle
in ``D`` equals integrating over the reflected triangle in ``x - y'``).

THE COEFFICIENT FORM, asserted rather than assumed: every coefficient is

    [polynomial of degree <= 2 in z and in nu, rational coefficients]
    / (pi * mu * (1 - nu))

so the table is exact integers and rationals -- no floating point is baked in,
and no expression strings are eval'd at import.

DERIVATIVE CONVENTION, which is where the image kernel differs from clq's.
clq's kernels depend on ``d = x - y`` alone, so a source derivative is minus an
observer derivative and one set of moments serves both. The IMAGE kernel also
depends on ``z0``, so that shortcut is false here:

    d/d(obs) = ( d/dx,  d/dy,  d/dz )
    d/d(src) = (-d/dx, -d/dy,  d/dz0)

Getting this wrong is an O(1) error on the vertical component only, which at
nu = 1/4 on a horizontal element is easy to mistake for a sign convention.

Run:  python tools/gen_image_table.py [--check]
``--check`` writes nothing and exits nonzero if the tree's copy is stale.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

import sympy as sp

OUT = (pathlib.Path(__file__).resolve().parents[1]
       / "src" / "mhs" / "kernels" / "_image_table.py")

# ---------------------------------------------------------------- symbols ----
x, y, z, z0 = sp.symbols("x y z z0", real=True)
mu_s, nu_s = sp.symbols("mu nu", positive=True)
R1, R2, Q = sp.symbols("R1 R2 Q", positive=True)
D1, D2, D3 = sp.symbols("D1 D2 D3", real=True)

#: Chain rules for the three radicals, carried as atoms so the family split is
#: reliable: after sp.expand, a composite like Q = R2 - (z + z0) is dissolved
#: and ``has(Q)`` stops working.
DR = {
    R1: {x: x / R1, y: y / R1, z: (z - z0) / R1, z0: -(z - z0) / R1},
    R2: {x: x / R2, y: y / R2, z: (z + z0) / R2, z0: (z + z0) / R2},
    Q:  {x: x / R2, y: y / R2, z: (z + z0) / R2 - 1, z0: (z + z0) / R2 - 1},
}

#: The common scale every coefficient carries.
COMMON = 1 / (sp.pi * mu_s * (1 - nu_s))


def d(expr, var):
    """d expr / d var, with R1, R2, Q carried as symbols."""
    out = sp.diff(expr, var)
    for S, rules in DR.items():
        if var in rules:
            out = out + sp.diff(expr, S) * rules[var]
    return out


def d_obs(expr, p: int):
    """Observer derivative: (d/dx, d/dy, d/dz)."""
    return d(expr, (x, y, z)[p])


def d_src(expr, m: int):
    """Source derivative: (-d/dx, -d/dy, +d/dz0). NOT minus the observer one."""
    return -d(expr, x) if m == 0 else (-d(expr, y) if m == 1 else d(expr, z0))


# ------------------------------------------------------------- potentials ----
def potentials(direction: str):
    """Apostol's (b, beta) for one unit force direction."""
    K, A0 = 1 / (4 * sp.pi * mu_s), -z0
    if direction == "z":
        b = [sp.S(0), sp.S(0),
             K * (1 / R1 + (3 - 4 * nu_s) / R2 + 2 * z0 * (z + z0) / R2 ** 3)]
        beta = K * (A0 / R1 + (3 - 4 * nu_s) * A0 / R2
                    - 4 * (1 - nu_s) * (1 - 2 * nu_s) * sp.log(Q))
    else:
        t = x if direction == "x" else y
        b = [K * (1 / R1 + 1 / R2) if direction == "x" else sp.S(0),
             K * (1 / R1 + 1 / R2) if direction == "y" else sp.S(0),
             2 * K * (A0 / R2 ** 2 - (1 - 2 * nu_s) / Q) * t / R2]
        beta = 2 * K * (1 - 2 * nu_s) * (A0 / R2 - (1 - 2 * nu_s)) * t / Q
    return b, beta


def green():
    """G[i][j]: displacement i from a unit force j, mollified Mindlin."""
    G = [[None] * 3 for _ in range(3)]
    for j, dr in enumerate(("x", "y", "z")):
        b, beta = potentials(dr)
        rb = x * b[0] + y * b[1] + z * b[2] + beta
        for i in range(3):
            G[i][j] = b[i] - d(rb, (x, y, z)[i]) / (4 * (1 - nu_s))
    return G


def r_family(expr):
    """The Q-free, R2-dependent part: the closed-form half."""
    out = sp.S(0)
    for t in sp.Add.make_args(sp.expand(expr)):
        if (not t.has(Q)) and t.has(R2):
            out = out + t
    return out


def q_family(expr):
    """The Q-dependent part: evaluated POINTWISE and integrated by quadrature.

    This half is not closed form, and the measurement that says it does not
    have to be is clause [d] of verify_vertical_fault: at eps/h = 0.1 the image
    quadrature is eps-INDEPENDENT and n_quad = 16 reaches machine precision even
    for an element reaching z = 0. It is the R-family that stagnates, because it
    reaches ``R2^-9`` where this reaches ``Q^-3``.

    Shipping without it is not an option, though: it is a real part of the
    kernel -- roughly a fifth of the on-fault stress near the trace -- so
    omitting it would be wrong by a plausible-looking amount, which is the
    failure mode ``matrices._not_yet`` exists to refuse.
    """
    out = sp.S(0)
    for t in sp.Add.make_args(sp.expand(expr)):
        if t.has(Q):
            out = out + t
    return out


# -------------------------------------------------------------- reduction ----
def to_monomials(expr, with_q: bool = False):
    """``{(a, b, c, n[, q]): coeff}`` for
    ``coeff * D1^a D2^b D3^c / (R2^n Q^q)``.

    ``with_q=False`` is the R-family and asserts ``n`` odd, because clq's
    hierarchy is odd-``n`` only and an even power would mean the family split
    had leaked. ``with_q=True`` is the Q-family, where even ``n`` and negative
    ``n`` (a positive power of ``R2``) both occur legitimately -- that half is
    evaluated pointwise, so no hierarchy constrains it.
    """
    e = sp.expand(expr.subs(z0, D3 - z).subs({x: D1, y: D2}))
    table = defaultdict(lambda: sp.S(0))
    for t in sp.Add.make_args(e):
        # R2 can hide inside an unexpanded Add in a denominator, e.g.
        # 1/(-8 pi R2^5 mu nu + 8 pi R2^5 mu); as_powers_dict cannot see that,
        # so cancel to one fraction and read degrees off numerator/denominator.
        num, den = sp.fraction(sp.cancel(sp.together(t)))
        n = int(sp.degree(den, R2) - sp.degree(num, R2))
        q = int(sp.degree(den, Q) - sp.degree(num, Q))
        a = int(sp.degree(num, D1) - sp.degree(den, D1))
        b = int(sp.degree(num, D2) - sp.degree(den, D2))
        c = int(sp.degree(num, D3) - sp.degree(den, D3))
        if min(a, b, c) < 0:
            raise AssertionError(f"negative D power in {t}")
        if not with_q and n % 2 == 0:
            raise AssertionError(
                f"even inverse power R2^-{n} in {t}: clq's hierarchy is odd-n "
                f"only, so an even power means the family split is wrong")
        if not with_q and q != 0:
            raise AssertionError(f"Q leaked into the R-family: {t}")
        coeff = sp.simplify(t * R2 ** n * Q ** q
                            / (D1 ** a * D2 ** b * D3 ** c))
        if coeff.free_symbols & {D1, D2, D3, R2, Q}:
            raise AssertionError(f"coefficient keeps a geometry symbol: {coeff}")
        key = (a, b, c, n, q) if with_q else (a, b, c, n)
        table[key] += coeff
    return {k: sp.simplify(v) for k, v in table.items() if sp.simplify(v) != 0}


def rationalise(coeff) -> tuple[tuple[int, int, int, int], ...]:
    """coeff -> ((pz, pnu, num, den), ...) meaning

        coeff = sum  num/den * z^pz * nu^pnu  /  (pi * mu * (1 - nu))

    Asserts the structure rather than trusting it: anything that is not
    polynomial in (z, nu) after dividing out COMMON is a generator bug, not
    something to coerce.
    """
    rest = sp.simplify(coeff / COMMON)
    if rest.free_symbols - {z, nu_s}:
        raise AssertionError(
            f"coefficient does not reduce to poly(z, nu)/(pi mu (1-nu)): "
            f"{coeff} leaves {rest.free_symbols - {z, nu_s}}")
    poly = sp.Poly(sp.expand(rest), z, nu_s)
    out = []
    for (pz, pnu), q in sorted(poly.terms()):
        q = sp.Rational(q)
        out.append((int(pz), int(pnu), int(q.p), int(q.q)))
    return tuple(out)


def build():
    """(R-family DG, R-family DDG, Q-family DG, Q-family DDG) records."""
    G = green()
    # dG_ij / d src_m, split once. A d/d obs derivative cannot move a term
    # between families afterwards, because no chain rule mixes R2 with Q or R1.
    DGr = [[[r_family(d_src(G[i][j], m)) for m in range(3)]
            for j in range(3)] for i in range(3)]
    DGq = [[[q_family(d_src(G[i][j], m)) for m in range(3)]
            for j in range(3)] for i in range(3)]
    dg_r, ddg_r, dg_q, ddg_q = [], [], [], []
    for i in range(3):
        for j in range(3):
            for m in range(3):
                for key, co in sorted(to_monomials(DGr[i][j][m]).items()):
                    dg_r.append((i, j, m) + key + (rationalise(co),))
                for key, co in sorted(
                        to_monomials(DGq[i][j][m], with_q=True).items()):
                    dg_q.append((i, j, m) + key + (rationalise(co),))
                for p in range(3):
                    er = r_family(d_obs(DGr[i][j][m], p))
                    for key, co in sorted(to_monomials(er).items()):
                        ddg_r.append((i, j, p, m) + key + (rationalise(co),))
                    eq = q_family(d_obs(DGq[i][j][m], p))
                    for key, co in sorted(
                            to_monomials(eq, with_q=True).items()):
                        ddg_q.append((i, j, p, m) + key + (rationalise(co),))
    return dg_r, ddg_r, dg_q, ddg_q


def fmt_records(name, recs, head_len, comment):
    lines = [f"#: {comment}", f"{name} = ("]
    for r in recs:
        idx = ", ".join(str(v) for v in r[:head_len])
        terms = ", ".join("(%d, %d, %d, %d)" % t for t in r[-1])
        lines.append(f"    ({idx}, ({terms},)),")
    lines.append(")")
    return "\n".join(lines)


HEADER = '''"""Image R-family monomial table -- GENERATED, do not edit by hand.

Regenerate with ``python tools/gen_image_table.py``; ``verify_image_table``
re-runs the generator and fails if this file is stale. The generator's module
docstring carries the derivation, the conventions and the reason the table is
data rather than code.

Each record is

    (indices..., a, b, c, n, ((pz, pnu, num, den), ...))

meaning that the image R-family contribution to that component includes

    sum[num/den * z^pz * nu^pnu] / (pi * mu * (1 - nu))
        * D1^a D2^b D3^c / R2^n

with ``D = x - y'`` the observer minus the REFLECTED source, ``z`` the
observer's absolute depth, and ``n`` always odd so clq's moment hierarchy
covers it. Integrated over an element, ``D1^a D2^b D3^c / R2^n`` is the
corresponding Cartesian slot of the rank-``(a+b+c)`` tensor moment at order
``n`` on the reflected triangle.

``DG_RECORDS`` indices are ``(i, j, m)`` for ``dG_ij / d src_m``;
``DDG_RECORDS`` are ``(i, j, p, m)`` for ``d2 G_ij / d obs_p d src_m``.
"""
'''


def render(dg_r, ddg_r, dg_q, ddg_q) -> str:
    rank = max([r[3] + r[4] + r[5] for r in dg_r]
               + [r[4] + r[5] + r[6] for r in ddg_r])
    orders = sorted(set(r[6] for r in dg_r) | set(r[7] for r in ddg_r))
    return "\n\n".join([
        HEADER.rstrip("\n"),
        fmt_records("DG_RECORDS", dg_r, 7,
                    "R-FAMILY, closed form. "
                    "(i, j, m, a, b, c, n, coeff): dG_ij / d src_m"),
        fmt_records("DDG_RECORDS", ddg_r, 8,
                    "R-FAMILY, closed form. (i, j, p, m, a, b, c, n, coeff): "
                    "d2 G_ij / d obs_p d src_m"),
        fmt_records("Q_DG_RECORDS", dg_q, 8,
                    "Q-FAMILY, evaluated pointwise for quadrature. "
                    "(i, j, m, a, b, c, n, q, coeff)"),
        fmt_records("Q_DDG_RECORDS", ddg_q, 9,
                    "Q-FAMILY, evaluated pointwise for quadrature. "
                    "(i, j, p, m, a, b, c, n, q, coeff)"),
        f"#: Sanity constants the consumer asserts against, so a truncated or\n"
        f"#: hand-edited table is caught rather than silently halving a kernel.\n"
        f"N_DG = {len(dg_r)}\n"
        f"N_DDG = {len(ddg_r)}\n"
        f"N_Q_DG = {len(dg_q)}\n"
        f"N_Q_DDG = {len(ddg_q)}\n"
        f"MAX_RANK = {rank}\n"
        f"N_ORDERS = {orders}\n",
    ]) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="write nothing; fail if the tree's copy is stale")
    args = ap.parse_args()

    print("deriving the image kernel (27 DG + 81 DDG components, "
          "split R / Q) ...", flush=True)
    dg_r, ddg_r, dg_q, ddg_q = build()
    text = render(dg_r, ddg_r, dg_q, ddg_q)
    print(f"  R-family  DG monomials: {len(dg_r):5d}   "
          f"DDG: {len(ddg_r):5d}   (closed form)")
    print(f"  Q-family  DG monomials: {len(dg_q):5d}   "
          f"DDG: {len(ddg_q):5d}   (quadrature)")

    if args.check:
        if not OUT.exists():
            print(f"FAIL: {OUT} does not exist")
            return 1
        if OUT.read_text() != text:
            print(f"FAIL: {OUT} is stale -- rerun without --check")
            return 1
        print(f"ok: {OUT} is up to date")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
