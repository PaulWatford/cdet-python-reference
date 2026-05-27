# 2-site Hubbard dimer

Two-site Hubbard chain with hopping $t$ between sites and on-site
interaction $U$. 16-state Hilbert space.

This is the smallest nontrivial multi-site test for CDet: it exercises
the site-labelled Wick determinant and the site-agnostic CDet recursion,
but the Hilbert space is small enough that arbitrary-precision
references (mpmath dps=50) are tractable through $n=4$.

## Files

- `derivation.md`: bonding/antibonding band structure, vertex sums.
- `green_function.py`: `G0_2site` via the diagonal band basis.
- `exact_diagonalization.py`: 16x16 ED reference for arbitrary $U$.
- `exact_taylor.py`: symbolic Taylor expansion of $G(\tau; U)$ via mpmath.
- `wick_determinant.py`: site-labelled $D_V$.
- `cdet_recursion.py`: site-agnostic CDet recursion.

## Tests

`test_dimer_cdet.py` (n=0..2), `test_dimer_cdet_higher.py` (n=3),
`test_dimer_cdet_n4.py` (n=4 via simplex integration).
