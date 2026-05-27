# 4-site pi-flux square plaquette

Four sites at the corners of a square with gauge-fixed hopping signs
chosen so that the product of hoppings around the plaquette equals
$-1$ (the pi-flux condition). 256-state many-body Hilbert space.

The free single-particle Hamiltonian is built explicitly and
diagonalised numerically. The free Green's function is constructed
via the spectral sum over the four single-particle eigenstates.

## Files

- `derivation.md`: Hamiltonian with gauge signs, spectral-sum
  construction, half-filling structure.
- `green_function.py`: `G0_4site_piflux`, `G0_4site_piflux_at_zero_minus`,
  `G0_4site_piflux_smart`.
- `exact_diagonalization.py`: 256-dim ED via Jordan-Wigner.
- `wick_determinant.py`: $D_V$ with 4-site vertex labelling.
- `cdet_recursion.py`: recursion (site-agnostic).

## Tests

`test_piflux_cdet.py` (n=0..2), `test_piflux_cdet_n3.py` (n=3 via
simplex integration).
