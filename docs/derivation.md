# Cortez-consistent mollified Mindlin: what the potential route delivers

A record of step 7 of the plan, including a negative result, because a route
that was tried and measured is worth more written down than re-attempted.

## The question

The vendored Mindlin kernel mollifies by the **literal lift** — substitute
`r → sqrt(r² + ε²)` in the classical potentials — plus an explicit Cortez blob
on the direct part. Classical Mindlin satisfies `t₃ₖ = 0` on `z = 0` identically;
the lifted kernel satisfies it only to O(ε²), measured 4.5e-4 at ν = 0.25 and
1.97e-2 at ν = 0.49 (ε = 0.1).

The hope was that mollifying **correctly** — convolving the source with the blob
rather than substituting in the radicals — would make the free-surface condition
**exact**, since each point-force Mindlin solution satisfies it and convolution
is linear.

## What is true

**The four generator identities.** With `R = sqrt(|d|² + ε²)`, `Q = R − d₃`:

```
r        * φ = R                                  (no counter-term)
1/r      * φ = 1/R   + (ε²/2)/R³                  (known Cortez)
log(r−z) * φ = log Q − (ε²/2)/(R Q)               NEW
d_a/(r−z)* φ = d_a/Q + (ε²/2) d_a/(R Q²)          NEW
```

Verified against 3-D quadrature of the convolution to 7e-9 … 1e-5, while the
literal lift is ≥ 1024× worse. Gated in `tests/gates/mhs/verify_blob_identities.py`.

**The unified rule reproduces the published Cortez Kelvin kernel.** For

```
f * φ = (1 − ε² ∂/∂(ε²)) [f]_ε  +  (ε²/2) [½ ∇² f]_ε
```

applied to Kelvin's Papkovich–Neuber potentials, all nine entries of
`G - C₁[(3−4ν)δ/R + d_i d_j/R³ + 2(1−ν)ε²δ/R³]` simplify to **exactly zero**
symbolically, and to 1.9e-16 numerically. The literal lift is missing precisely
`−ε²/(8πμR³)`, which *is* `C₁·2(1−ν)ε²/R³`: the blob term.

**Apostol's potentials are biharmonic in the source variable** — all of them,
all three force directions, `∇⁴_src f = 0` to 1e-16 or better. The mollification
is a *source* convolution, so that is the variable that matters.

## What is not true

**The rule is exact only for HARMONIC `f`.** For merely biharmonic `f` it is
O(ε⁴). Measured on Apostol's complete potentials, each verified biharmonic,
against the source convolution at two quadrature resolutions so the residual is
the rule's and not the measurement's:

| potential | ∇⁴_src | ε=0.2 | ε=0.1 | order |
|---|---|---|---|---|
| vertical `b_z` | 5.6e-16 | 1.27e-4 | 8.21e-6 | 3.95 |
| vertical `β` | 2.1e-16 | 7.09e-4 | 4.53e-5 | 3.97 |
| horiz-x `b_z` | 8.7e-19 | 4.23e-5 | 2.73e-6 | 3.95 |
| horiz-x `β` | 4.3e-19 | 5.91e-6 | 3.64e-7 | 4.02 |

`1/r₂` alone, being harmonic, comes out exact (1.5e-11, pure quadrature). The
`r` case is exact too, which is what misled the derivation: generalising from
one trivially-biharmonic example to all biharmonic functions was the error.

**So the free-surface traction stays O(ε²)**, with a better constant:

| ν | literal lift (order) | corrected (order) | gain |
|---|---|---|---|
| 0.25 | 1.97, 1.99, 2.00 | 1.75, 1.94, **1.98** | 2.6–3.1× |
| 0.49 | 1.98, 1.99, 2.00 | 4.18, 2.26, **1.57** | 9–50× |

The potentials improve from O(ε²) to O(ε⁴), but the traction takes **two
derivatives** of them, and differentiating an error whose spatial scale is ε
twice costs ε²: O(ε⁴)/ε² = O(ε²). The arithmetic is consistent and the
conclusion is not escapable by a better implementation of this route.

## Two wrong turns, recorded so they are not repeated

**The lift-agnostic-symbols shortcut does not work.** Carrying `R₁`, `R₂` as
symbols with the rules `∂R/∂x = x/R` makes *first* derivatives lift-agnostic,
which is true and tempting. But the Laplacian needs `Σ ∂²R/∂x_i²`, which brings
in `Σ x_i² = R² − ε²` — a relation that differs between lifted and unlifted. So
that route computes `∇²[f]_ε`, not `[∇² f]_ε`, which is exactly the O(ε²) error
in the counter-term and hence O(ε⁴) in the result. The correct implementation
computes `∇²` on the unlifted function and lifts the result, by substituting
`S → S + ε²` under every non-integer power of `S = x² + y² + (z∓z₀)²`.

**A fourth derivative by nested finite differences measures its own
cancellation.** Nested 5-point second-derivative stencils divide by `h⁴`; at
`h = 2e-3` that read 7e-4 on a quantity whose true value is 1e-16. Exact algebra
needs symbolic differentiation.

## Why this is the obstruction, not an implementation gap

The moment expansion of a convolution terminates for biharmonic `f` only if the
blob's fourth and higher moments are finite. The Cortez blob has an **algebraic
tail**, `φ_ε ~ 15ε⁴/(8π r⁷)`, so `∫ |w|⁴ φ dw ~ ∫ r⁻¹ dr` diverges
logarithmically. No finite-order rule can be exact for non-harmonic `f` against
such a blob. A terminating rule would need a compactly supported blob — not the
one the published mollified kernels are built on.

This is the same obstruction the free-surface trilemma names: `φ_ε` has
algebraic tails, so some blob mass sits above `z = 0` and its image lands inside
the body.

## What is foreclosed, and what is not

Foreclosed: **an exact free-surface condition via the potential-convolution
rule.**

Not foreclosed: the identity `G^{ε,HS} = G^{0,HS} * φ_ε` remains true by
definition, and the exact convolution — if it has a closed form at all — is
still whatever it is. What has been measured is that *this* way of evaluating it
is O(ε⁴) on the potentials and O(ε²) on the traction.

## Bearing on the rest of the plan

The honest claim for a half-space kernel built this way is now:

> O(ε⁴) potentials and an O(ε²) free-surface residual with a constant 2.6–50×
> better than the literal lift

rather than "exact free-surface condition". Anything written for publication
should say that.

Step 8 — analytic triangle integration — is **unaffected in value**. Its case was
never the free-surface condition: it is that closed-form integration removes the
`n_quad ≈ 8 L/ε` budget (measured in
`tests/gates/oracle/verify_vertical_fault.py`) and makes the near-singular
surface-breaking case reachable. Integrating an O(ε⁴) kernel analytically is
worth the same as integrating an exact one; only the headline sentence changes.

## Decision: ship the literal lift

Made on measurement, and the measurement corrected the earlier impression.

**The 2.6–50× figures above are a DIAGNOSTIC, not a user-visible gain.** The
free-surface traction residual is a quantity that should be zero, so a ratio
against it flatters any improvement. What a caller actually reads is the
Green's function. Measured against the EXACT mollified kernel — the source
convolution of classical Mindlin, which is the definition — at a buried point
source:

| observer | ν | ε | literal lift | corrected | gain |
|---|---|---|---|---|---|
| surface | 0.25 | 0.20 | 3.46e-3 | 6.85e-4 | 5.1× |
| surface | 0.25 | 0.10 | 8.70e-4 | 1.69e-4 | 5.2× |
| surface | 0.49 | 0.20 | 3.64e-3 | 1.30e-3 | 2.8× |
| surface | 0.49 | 0.10 | 9.14e-4 | 3.22e-4 | 2.8× |
| near-field | 0.25 | 0.20 | 1.87e-2 | 6.70e-3 | 2.8× |
| near-field | 0.25 | 0.10 | 4.92e-3 | 2.01e-3 | 2.4× |
| near-field | 0.49 | 0.20 | 2.16e-2 | 1.14e-2 | 1.9× |
| near-field | 0.49 | 0.10 | 5.67e-3 | 3.43e-3 | 1.7× |

**1.7× to 5.2×, and both columns are O(ε²)** — the errors fall by four per ε
halving in each. The cost, counted as symbolic operations in the assembled
kernel: **1,986 → 43,333, a factor of 21.8.**

Three reasons that is the wrong trade:

1. **21.8× the arithmetic for ≤ 5.2× the accuracy**, with no change of order.
2. Because both are O(ε²), a 5× gain is worth about what reducing ε by 2.2×
   buys. Under quadrature that costs ~5× more points (n_quad ∝ L/ε per
   direction); under the closed form of step 8 it costs nothing. Either way it
   is cheaper than 21.8×.
3. **The lift's accuracy is not the limiting term for any realistic use.** At
   ε = 0.1 it is 8.7e-4 on the surface — two orders below the box truncation and
   observational error that dominated the equivalent full-space budget. Buying
   headroom nobody is using is not an improvement.

So the shipped kernel is the literal lift, and this derivation stays
documentation. The corrected form remains useful as a *reference*: it is
genuinely more accurate, and a gate that ever needs a better-than-lift
comparison can build it from the recipe above.

Worth naming the general lesson, since it nearly went the other way: the
improvement was measured first against the quantity it was designed to improve
(a residual that should be zero) and only later against the quantity a caller
reads. The first framing said 50×, the second says 1.7×. The second is the one
that decides.


---

# Step 8b: the Q-family rationalises, and what is left is two quadratics

Recorded before the work, because the first finding changes what the work IS.

## The Q-family has no transcendentals

The image kernel's Q-family carries ``1/Q^q`` with ``Q = R2 - D3``, which looks
like a new transcendental family needing its own primitives. It is not. Since

    R2^2 = D1^2 + D2^2 + D3^2 + eps^2      so      R2^2 - D3^2 = W

with ``W = D1^2 + D2^2 + eps^2``, the denominator rationalises:

    1/Q = (R2 + D3) / W                             (residual exactly 0)

so ``1/Q^q = (R2 + D3)^q / W^q``, and EVEN powers of ``R2`` then collapse
through ``R2^2 = W + D3^2``. Only a single factor of ``R2`` survives. The whole
Q-family is therefore

    poly(D1, D2, D3, eps) / W^k   +   R2 * poly(...) / W^k

-- rational, with no ``log Q`` and no ``1/Q`` anywhere. (``log Q`` appears in
Apostol's potentials but not in ``dG/d src`` or ``d2G/d obs d src``, which is
all the slip kernels need: the measured Q powers there are {0, 1, 2, 3} with no
log.)

## What W is, and why the geometry decides the difficulty

``D1, D2`` are the HORIZONTAL components of ``D``, so ``W`` is the mollified
squared distance from the observer to the VERTICAL LINE through the source
point. The primitive needed is

    int_T  xi1^a xi2^b / (W^k R^n) dS,      n odd

a product of the existing hierarchy's denominator with powers of a SECOND
quadratic. How hard that is depends entirely on the element's orientation:

| element | W in the element's frame | consequence |
|---|---|---|
| horizontal | ``xi1^2 + xi2^2 + eps^2`` -- the SAME radial quadratic as ``R``, at a different height | the existing hierarchy, evaluated twice. No new primitives. |
| vertical | one in-plane direction is vertical, and ``W`` does not depend on it | that integral is elementary |
| general dip | two genuinely different quadratics | the new primitive family |

So the roadmap's rung 1 / rung 2 split survives, and the "two quadratic forms"
risk it flagged is real -- but it is now a RATIONAL two-quadratic integral over
a polygon rather than an unknown transcendental class. Euler substitution on
the conic gives log, arctan and algebraic terms; no dilogarithm barrier.

## What changed about WHY to do it

8b was motivated as the capability fix for on-fault stress near a surface trace.
**That motivation is gone**: the quadrature budget law (``q_gauss_orders``)
closed the accuracy gap -- 1e-15 at every collocation point, and 1.1e-8 at
0.03 h below a trace at eps/h = 0.003, against O(1) before.

What remains is COST, and it is large. On a realistic batch the Q-family
quadrature is about 98% of the image kernel's runtime (12.4 ms per obs/source
pair against the closed-form R-family's 0.23 ms), and the adaptive rule raised
it 4.1x over the flat order it replaced. A closed-form Q-family would remove
essentially all of it.

So 8b is now a performance project with a bounded, rational target -- a better
position than when it was a capability project with an unknown function class,
but no longer urgent. The accuracy it was meant to buy is already banked.
