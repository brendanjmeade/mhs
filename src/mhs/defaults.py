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
