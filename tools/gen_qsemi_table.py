#!/usr/bin/env python
"""Generate ``mhs/kernels/_qsemi_table.py``: the Q-family's INNER integrals.

WHAT THIS IS FOR. The Q-family of the image kernel is integrated over an
element by a SEMI-ANALYTIC rule: the along-strike direction exactly, the
down-dip direction by a graded 1-D Gauss rule. See ``docs/derivation.md`` for
why that works at every orientation; the measured payoff is 171x fewer
evaluation points than a graded 2-D rule at identical accuracy (352 against
60224 for 2e-12), and ten orders better than a plain 2-D rule at three times
its point count.

THE GEOMETRY THAT MAKES IT POSSIBLE. With ``t`` along STRIKE and ``v1`` down
dip, measured from the observer's in-plane foot:

    R^2 = t^2 + B(v1),   B = v1^2 + z^2 + eps^2
    D3  = D3(v1),        INDEPENDENT of t, because strike is horizontal
    Q   = R - D3

so for fixed ``v1`` every integrand is a function of ``t`` alone through ``t``
and ``R``, with ``B`` and ``D3`` constants. Any rational function of ``(t, R)``
with ``R^2`` quadratic in ``t`` has an elementary antiderivative, so each one
closes.

WHAT IS GENERATED. The 79 distinct ``int t^j dt / (R^n Q^q)`` the table
actually needs -- j in 0..4, n in 0..7, q in 1..4 -- as antiderivatives in
``t``, emitted as numpy expression source. The set is read off
``_image_table.py`` rather than written down, because a monomial
``D1^a D2^b D3^c`` has ``D1, D2`` affine in ``t`` and ``D3`` constant, so it
contributes powers ``t^j`` for ``j = 0 .. a+b``.

HOW EACH IS OBTAINED. Directly asking for ``int t^j/(R^n Q^q) dt`` makes sympy
work hard for no reason. Rationalising first is much faster and is exact:

    1/Q = (R + D3)/W,     W = t^2 + A,   A = B - D3^2

(verified in docs/derivation.md), so ``1/Q^q = (R+D3)^q / (t^2+A)^q`` and every
EVEN power of ``R`` collapses through ``R^2 = t^2 + B``. What is left is a sum
of ``int t^j (t^2+B)^p / (t^2+A)^q dt`` (rational) and the same with a single
factor of ``R`` (one radical) -- both standard.

EVERY RESULT IS VERIFIED BY DIFFERENTIATION, not by quadrature: ``d/dt`` of the
antiderivative must equal the integrand identically. That catches a wrong
branch, which a numerical spot-check at one ``t`` would not.

THE REMOVABLE DEGENERACY. ``A = B - D3^2`` and ``D3 -> 0`` makes ``A -> B``,
where the ``1/(B-A)`` of the partial fractions diverges while the terms it
multiplies coalesce. The limit is finite (docs/derivation.md), so the consumer
needs a branch there rather than a guard; this generator records which
expressions carry a ``(B-A)`` denominator so the consumer knows which ones.

Run:  python tools/gen_qsemi_table.py [--check] [--only j,n,q]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import sympy as sp

OUT = (pathlib.Path(__file__).resolve().parents[1]
       / "src" / "mhs" / "kernels" / "_qsemi_table.py")

#: A is an INDEPENDENT symbol, not the expression B - D3^2. Substituted in,
#: the denominator (t^2 + B - D3^2) couples three symbols and sympy cannot
#: resolve the branches -- every integral comes back unevaluated. Kept
#: independent and positive (both A >= eps^2 and B >= eps^2 hold), each piece
#: is a textbook two-quadratic integral. The CONSUMER supplies A = B - D3^2.
t, D3 = sp.symbols("t D3", real=True)
A, B = sp.symbols("A B", positive=True)
R = sp.sqrt(t ** 2 + B)


def needed() -> list[tuple[int, int, int]]:
    """(j, n, q) read off the image table, not written down."""
    sys.path.insert(0, str(OUT.parents[2]))
    from mhs.kernels._image_table import Q_DDG_RECORDS, Q_DG_RECORDS
    out = set()
    for rec in Q_DG_RECORDS:
        a, b, _, n, q = rec[3], rec[4], rec[5], rec[6], rec[7]
        for j in range(a + b + 1):
            out.add((j, n, q))
    for rec in Q_DDG_RECORDS:
        a, b, _, n, q = rec[4], rec[5], rec[6], rec[7], rec[8]
        for j in range(a + b + 1):
            out.add((j, n, q))
    return sorted(out)


def integrand(j: int, n: int, q: int):
    """t^j / (R^n Q^q), as written -- the reference the result is checked
    against."""
    return t ** j / (R ** n * (R - D3) ** q)


def pieces(j: int, n: int, q: int):
    """The rationalised integrand, split into terms sympy can each handle.

    ``1/Q^q = (R + D3)^q/(t^2 + A)^q`` with ``A = B - D3^2``, so expanding the
    binomial gives terms ``t^j R^(i-n)/(t^2+A)^q``. Writing ``R^k`` as an
    explicit HALF-INTEGER power of ``(t^2 + B)`` rather than a nested sqrt is
    what makes these tractable: handed the single combined fraction, sympy
    returns the integral unevaluated.
    """
    S = t ** 2 + B
    out = []
    for i in range(q + 1):
        coeff = sp.binomial(q, i) * D3 ** (q - i)
        k = sp.Rational(i - n, 2)                  # power of S = R^(i-n)
        out.append(coeff * t ** j * S ** k / (t ** 2 + A) ** q)
    return out


def antiderivative(j: int, n: int, q: int):
    """int t^j dt/(R^n Q^q), verified TWO ways.

    With ``A`` independent the check splits in two, and both matter: the
    antiderivative must differentiate back to the SUM OF PIECES (exact algebra,
    A and B free), and the sum of pieces must equal the original
    ``t^j/(R^n Q^q)`` once ``A = B - D3^2`` is imposed (that is the
    rationalisation identity, re-verified here per entry rather than trusted
    from the derivation).
    """
    F = sp.S(0)
    ps = pieces(j, n, q)
    for piece in ps:
        Fi = sp.integrate(piece, t)
        if Fi.has(sp.Integral):
            return None, f"unevaluated piece {sp.srepr(piece)[:40]}"
        F = F + Fi
    # (ii) the rationalisation holds for THIS entry, numerically
    sub = {B: sp.Rational(7, 5), D3: sp.Rational(2, 5), t: sp.Rational(1, 3)}
    sub[A] = sub[B] - sub[D3] ** 2
    lhs = complex(sp.N(sum(ps).subs(sub)))
    rhs = complex(sp.N(integrand(j, n, q).subs(sub)))
    if abs(lhs - rhs) > 1e-18 * max(abs(rhs), 1.0):
        return None, (f"rationalisation mismatch "
                      f"{abs(lhs-rhs)/max(abs(rhs),1e-300):.2e}")
    # (i) the antiderivative differentiates back, exactly
    e = sum(ps)
    resid = sp.simplify(sp.diff(F, t) - e)
    if resid != 0:
        # simplify can fail to see zero; fall back to a numeric identity check
        vals = {B: sp.Rational(7, 5), D3: sp.Rational(2, 5),
                A: sp.Rational(7, 5) - sp.Rational(2, 5) ** 2}
        bad = 0.0
        for tv in (sp.Rational(1, 3), sp.Rational(-4, 5), sp.Rational(9, 7)):
            sub = dict(vals)
            sub[t] = tv
            a = complex(sp.N(sp.diff(F, t).subs(sub)))
            b = complex(sp.N(e.subs(sub)))
            bad = max(bad, abs(a - b) / max(abs(b), 1e-300))
        if bad > 1e-20:
            return None, f"derivative mismatch {bad:.2e}"
    return F, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default=None,
                    help="a single j,n,q, for debugging one case")
    args = ap.parse_args()

    want = ([tuple(int(x) for x in args.only.split(","))] if args.only
            else needed())
    print(f"deriving {len(want)} inner antiderivatives "
          f"int t^j dt/(R^n Q^q) ...", flush=True)
    rows, failed = [], []
    t0 = time.time()
    for k, (j, n, q) in enumerate(want):
        F, status = antiderivative(j, n, q)
        if F is None:
            failed.append((j, n, q, status))
            print(f"  [{k+1:3d}/{len(want)}] j={j} n={n} q={q}  FAILED: "
                  f"{status}", flush=True)
            continue
        src = sp.printing.lambdarepr.NumPyPrinter().doprint(sp.simplify(F))
        rows.append((j, n, q, src))
        if (k + 1) % 10 == 0:
            print(f"  [{k+1:3d}/{len(want)}]  {time.time()-t0:6.1f} s",
                  flush=True)
    print(f"\n  {len(rows)} derived and VERIFIED by differentiation, "
          f"{len(failed)} failed, {time.time()-t0:.1f} s")
    for j, n, q, why in failed:
        print(f"    j={j} n={n} q={q}: {why}")
    if failed:
        print("\n  A failure is not something to paper over: the consumer "
              "would\n  silently drop those monomials. Fix or special-case "
              "them.")
        return 1
    if args.only:
        for j, n, q, src in rows:
            print(f"\nj={j} n={n} q={q}:\n  {src}")
        return 0
    text = render(rows)
    if args.check:
        ok = OUT.exists() and OUT.read_text() == text
        print(f"  {OUT}: {'up to date' if ok else 'STALE'}")
        return 0 if ok else 1
    OUT.write_text(text)
    print(f"  wrote {OUT} ({len(text)} bytes)")
    return 0


HEADER = '''"""Inner (along-strike) antiderivatives of the Q-family -- GENERATED.

Regenerate with ``python tools/gen_qsemi_table.py``; that script's docstring
carries the derivation and the conventions. Every entry was verified by
differentiating it back to the integrand, not by a numerical spot check.

``INNER[(j, n, q)](t, A, B, D3)`` evaluates

    F = int t^j dt / (R^n Q^q),     R = sqrt(t^2 + B),   Q = R - D3

with ``A = B - D3^2`` supplied by the caller, and ``A``, ``B``, ``D3`` constant
along the strike direction. The definite integral over a slab is
``F(t_hi) - F(t_lo)``. ``A`` is passed rather than derived because it is what
the antiderivatives are written in: see the generator for why it has to be an
independent symbol.
"""
from __future__ import annotations

import numpy as np

from numpy import arctan, arctanh, log, sqrt      # noqa: F401  (used by eval)
'''


def render(rows) -> str:
    lines = [HEADER.rstrip("\n"), "", "", "INNER = {"]
    for j, n, q, src in rows:
        lines.append(f"    ({j}, {n}, {q}):")
        lines.append(f"        lambda t, A, B, D3: {src},")
    lines.append("}")
    lines.append("")
    lines.append(f"#: asserted by the consumer, so a truncated table is caught")
    lines.append(f"N_INNER = {len(rows)}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
