"""Kernel identifiers. The arithmetic lives in the sibling modules.

Each kernel is its own module so numba's ``cache=True`` invalidation tracks the
file that holds the arithmetic: editing the half-space mathematics in
``mindlin.py`` must not discard ``moments.py``'s compiled artifact. The same
split sets the CI cache key.
"""

#: The per-triangle blocks an assembler can ask for.
KERNELS = ("disp", "stress", "eigenstress")
