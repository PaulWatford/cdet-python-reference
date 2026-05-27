# 4-site pi-flux square plaquette: CDet setup

Four sites at the corners of a square with gauge-fixed hopping signs
chosen so that the product of hopping amplitudes around the plaquette
equals $-1$ (the pi-flux condition). The Hubbard Hamiltonian is

$$H = -t \sum_{\langle i j \rangle, \sigma} s_{ij}\, c^\dagger_{i\sigma} c_{j\sigma}
    + U \sum_i n_{i\uparrow} n_{i\downarrow}
    - \mu \sum_{i\sigma} n_{i\sigma}$$

where $s_{ij} \in \{+1, -1\}$ are the gauge signs.

## Free Green's function

The free single-particle Hamiltonian is the $4 \times 4$ real symmetric
matrix with the gauge-signed hopping signs and the chemical potential
on the diagonal. The implementation (`green_function.py`):

1. Build $H_{\text{kin}} = -t M$ where $M$ is the gauge-signed
   hopping matrix.
2. Diagonalise with `numpy.linalg.eigh`, obtaining four eigenvalues
   $\{\varepsilon_n\}$ and eigenvectors.
3. Construct the imaginary-time Green's function via the spectral sum

$$G_0(i, j; \tau) = \sum_n \langle i | n \rangle \langle n | j \rangle
                    \, g_{\text{band}}(\varepsilon_n - \mu;\, \tau)$$

with $g_{\text{band}}$ the standard per-band imaginary-time Green's
function (anti-periodic in $\tau$).

Verification in the `__main__` block of `green_function.py` checks
anti-periodicity, the $G_{ij} = G_{ji}$ symmetry, and the half-filling
density $G_0(i, i; 0^-) = 0.5$ per spin at $\mu = 0$.

## Vertex structure for CDet

CDet on this system organises diagrams by vertex configurations
$V = \{(i_k, \tau_k)\}_{k=1}^n$ over the 4 plaquette sites and the
ordered-times simplex on $[0, \beta]^n$. See `cdet_recursion.py` and
`wick_determinant.py` for the algorithm.

## Half-filling structure

At half-filling and $\mu = 0$ the diagonal entries of $G_0$ are $0.5$
per spin by particle-hole symmetry. The ED reference in
`exact_diagonalization.py` reproduces this with the 256-state
many-body Hilbert space.

## Verification ladder

1. free Green's function checks (in `green_function.py`)
2. ED reference via Jordan-Wigner gives the $256 \times 256$ many-body $H$
3. Wick determinant $D_V$ verified against $n = 0$ and $n = 1$ cases
4. CDet recursion verified for vertex permutation invariance
5. end-to-end audit at $n = 0, 1, 2$ against ED finite differences
6. extension to $n = 3$ via simplex integration (separate test file)

`tests/test_piflux_cdet.py` exercises steps 1-5 and
`tests/test_piflux_cdet_n3.py` extends to $n = 3$.
