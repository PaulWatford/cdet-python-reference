# 6-site honeycomb hexagonal ring

Six sites in a ring with nearest-neighbour hopping, no flux.
4096-state many-body Hilbert space.

The free single-particle Hamiltonian is built explicitly and
diagonalised numerically (`numpy.linalg.eigh`). The free Green's
function is constructed via the spectral sum

$$G_0(i, j; \tau) = \sum_n \langle i | n \rangle \langle n | j \rangle
                    \, g_{\text{band}}(\varepsilon_n - \mu;\, \tau)$$

over the six single-particle eigenstates.

The ring's $Z_6$ rotational symmetry is preserved exactly by the CDet
output (verified to machine precision in the audit suite).

## Files

- `derivation.md`: Hamiltonian, spectral-sum construction, vertex
  structure for CDet.
- `green_function.py`: `G0_hexring`, `G0_hexring_at_zero_minus`.
- `exact_diagonalization.py`: 4096-dim ED via Jordan-Wigner with
  sparse Hamiltonian construction.
- `wick_determinant.py`: $D_V$ with 6-site vertex labelling.
- `cdet_recursion.py`: Rossi CDet recursion (site-agnostic).

## Tests

`test_hexring_cdet.py` (n=0..2; n=3 not currently included due to
runtime cost, $6^3 = 216$ site configurations per simplex order).
