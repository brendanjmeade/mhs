"""Every tolerance, threshold and ceiling in one module, which imports nothing.

A convention written twice is a bug, so a number lives here and nowhere else.
Two kinds live here and they are not interchangeable:

* **User-facing**, which a caller may legitimately rebind at the top of a script
  (``MATRIX_MAX_BYTES`` is the only one). It is a resource limit, not an
  acceptance criterion.
* **Gate-only**, the ``*_PARITY*`` tolerances and the import budget. These are
  reachable from no public argument, because a run that could move its own pass
  threshold could declare its own success.
"""

# ----------------------------------------------------------------- memory ----
# A stress matrix is 9 * 3 * 8 = 216 bytes per obs/source pair, and parts=True
# is two of them: 432. At 10k obs x 10k sources that is 21.6 / 43.2 GB, which is
# the target case for an inversion rather than a pathological corner.
#
# The ceiling REFUSES rather than trying, because a 21.6 GB allocation on a
# 36 GB machine does not fail -- it swaps, and presents as a hang with no
# diagnosis. The refusal message carries the arithmetic that makes the next step
# obvious, and `out=` plus input slicing are the documented escapes.
MATRIX_MAX_BYTES = 8 * 1024**3
# Fraction of total RAM the default ceiling is additionally capped at, for a
# machine smaller than the number above.
MATRIX_RAM_FRACTION = 0.5

# ------------------------------------------------------------ gate-only ------
# `import mhs` must not pay for sympy. Measured: importing the vendored
# oracle's mindlin_kernels costs 14.25 s, because it builds and lambdifies ~90
# symbolic matrices at module scope.
#
# The budget is RELATIVE to `import numba`, measured in the same subprocess,
# because mhs's own import runs every @njit decorator (compilation defers to
# first call) and numba's import is itself ~1 s and drifts between releases. An
# absolute budget would be a measurement of the machine and of numba's release
# notes; this one is a measurement of mhs.
IMPORT_BUDGET_OVER_NUMBA_S = 1.0

# Entrywise parity of a float32 `out=` against the float64 computation. The
# kernels always compute in float64 registers and the store downcasts, so this
# is a storage tolerance, not a second arithmetic.
OUT_FLOAT32_PARITY = 1.0e-6

#: Quadrature budget for the image Q-FAMILY (the R-family is closed form).
#:
#:     n_quad ~ IMAGE_Q_BUDGET_C * L / sqrt(delta^2 + eps^2)
#:
#: with ``delta`` the observer's distance to the IMAGE triangle. This is the
#: SAME law the direct term obeys (n_quad ~ 8 L / eps, gated in
#: oracle/verify_vertical_fault clause [b]) with the same constant: the direct
#: observer sits ON its element, so delta = 0 and the scale is eps.
#:
#: C measured 5.8 .. 8.4 for 1e-9 relative, over delta/h in {0.02 .. 0.33} and
#: eps/h in {0.01 .. 0.1}. The default carries headroom over the MAXIMUM, not
#: the mean, because starving this rule is silent -- the first version of this
#: file set a FLAT 16, chosen from a buried element where the error is
#: eps-independent, and that left an on-fault P1/P2 collocation point at 7e-4
#: and a readout 0.03 h below a surface trace at O(1).
IMAGE_Q_BUDGET_C = 10.0
#: Floor: below this the rule is cheap anyway and the law over-trims far field.
IMAGE_Q_GAUSS_MIN = 8
#: Ceiling. The law asks for ~400 at delta/h = 0.02, eps/h = 0.01; past this
#: the Q-family needs a closed form rather than more points, and the cost is
#: quadratic. Hitting it is reported rather than silently accepted.
IMAGE_Q_GAUSS_MAX = 192
#: Working-set budget for one Q-family block, in bytes. The 1329 records share
#: only 236 distinct monomials, so the family is evaluated as ONE contraction
#: against a monomial basis -- which means holding that basis, and the small
#: fixed set of cached powers, for a whole chunk of observers at once. That
#: working set grows as n_obs * n_quad, so the observer axis is chunked against
#: this budget rather than left to the caller: n_quad reaches IMAGE_Q_GAUSS_MAX
#: exactly where observers cluster near a surface trace, which is the one
#: configuration where both factors are large together.
IMAGE_Q_BLOCK_BYTES = 64 << 20
#: Budget for the column blocks IN FLIGHT when `mhs.parallel.by_source` builds
#: a matrix across processes. Workers return their own block and the parent
#: writes it in, so the peak is the output plus whatever has been computed but
#: not yet consumed. Sizing the chunk COUNT against this bounds that second
#: term instead of letting it scale with the worker count -- with one chunk per
#: worker it would be the whole output a second time.
PARALLEL_INFLIGHT_BYTES = 512 << 20

# ---------------------------------------------------------- shared nodes ----
#: Weld tolerance for the continuous layout's shared-node table, as a fraction
#: of the mesh's shortest edge. A tiling reaches one geometric vertex by
#: different arithmetic paths (``i*h + h`` against ``(i+1)*h``), so the last
#: bits differ and an exact comparison is simply wrong; the welded count is
#: the same at 1e-12, 1e-9 and 1e-6 h on a lattice disc, so the value is not
#: delicate. 1e-9 is far under any mesh feature and far over round-off.
WELD_TOL_OVER_H = 1.0e-9

# ------------------------------------------------------------------ fold ----
#: THE FOLD. mhs's image is Mindlin's, "lifted" to the smeared source (S -> S
#: + eps^2 under the non-integer powers), which is right to O(eps^2) for a
#: BURIED source. A source whose smeared slip reaches above z = 0 is a
#: different problem: continued above the surface, the lifted formula puts
#: its image point and its Boussinesq branch line INSIDE the body, O(1/eps)
#: within eps of the trace. The regularised problem must instead take the
#: smeared slip FOLDED back into the body (the part above z = 0 replaced by
#: its mirror image: mass-preserving, the even extension, exact in antiplane
#: strain). Its image is the mirror element's own full-space field plus the
#: Boussinesq field of the normal traction the symmetric pair leaves on z = 0
#: (``mhs.kernels.fold``). A source is folded when its shallowest vertex is
#: within this many eps of the surface: beyond 3 eps the blob's algebraic
#: tails carry (eps/depth)^4 of its mass above z = 0 and the lift's error is
#: O(eps^2/depth^2) -- measured 1.9e-2 of the image at 12 eps, 1.6e-3 at 40.
FOLD_SOURCE_DEPTH_EPS = 3.0
#: Receivers deeper than this many eps keep the lift for a folded source.
#: Measured on a vertical strike-slip fault against the antiplane solution at
#: eps = 0.1 h: folding only receivers within 10 eps (one element height)
#: gives the same slip as folding every receiver -- collocation 0.977-0.980,
#: Galerkin 0.984-0.987 of exact -- because the lift's far-field error is
#: O(eps/D) on an entry that is itself O(1/D^2).
FOLD_BAND_DEPTH_EPS = 10.0
#: ... and so do receivers whose surface projection lies farther than this
#: many eps from the source's crease (the correction there is O(eps/D) of a
#: 1/D^2 entry: 1e-4 of the row at 60 eps, measured on fault B of the
#: on-fault study at 1.5 km = 60 eps).
FOLD_REACH_EPS = 60.0
#: The Boussinesq term is a surface integral of the pair's normal traction
#: against the point-load stress, on a polar grid centred under the receiver:
#: geometric radial panels from eps/2 (plus breakpoints at the receiver's own
#: depth when that is under eps/2, where the kernel's width is the depth),
#: Gauss points per panel below; the singular part is subtracted at the
#: receiver's foot point and restored by Love's closed-form uniform-disk field.
FOLD_N_RADIAL = 6
FOLD_N_ANGULAR = 4
#: The source's sharp surface features -- the smeared crease line along its
#: surface edge (a line of vertical Burgers vector, 1/rho for rho > eps) and
#: its surface vertices -- lie in the disk around its surface vertices of
#: radius (their spread) + this many eps.
FOLD_CREASE_MARGIN_EPS = 5.0
#: Angular spacing, in eps, where the circle crosses that disk (the crease
#: profile has width eps) ...
FOLD_FINE_SPACING_EPS = 0.25
#: ... and in the flanks out to four disk radii, where the crease field decays
#: as 1/rho; FOLD_N_COARSE points around the rest of the circle.
FOLD_MEDIUM_SPACING_EPS = 2.0
FOLD_N_COARSE = 24
#: Radial sub-panel widths, in eps, across the crease annulus and its flanks.
FOLD_FINE_SUBPANEL_EPS = 6.0
FOLD_MEDIUM_SUBPANEL_EPS = 16.0
#: The grid ends at R1 = (distance to the crease) + 4 (disk radius) + this
#: many source diameters, beyond which the pair's surface load is < 0.2 % of
#: its near value (it decays as 1/R^3). Doubling the whole grid moves a
#: straddling element's correction by 5.5e-3.
FOLD_R1_MARGIN_DIAM = 8.0
