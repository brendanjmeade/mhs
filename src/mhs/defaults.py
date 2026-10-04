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
