# 6-site hexagonal ring: CDet setup

Six sites in a ring with nearest-neighbour hopping, each carrying spin
$\uparrow, \downarrow$. The Hubbard Hamiltonian is

$$H = -t \sum_{\langle i j \rangle, \sigma} c^\dagger_{i\sigma} c_{j\sigma}
    + U \sum_i n_{i\uparrow} n_{i\downarrow}
    - \mu \sum_{i\sigma} n_{i\sigma}$$

with the sum over the six nearest-neighbour bonds of the ring
(0-1, 1-2, 2-3, 3-4, 4-5, 5-0).

## Free Green's function

The free part of $H$ (the $U = 0$ piece) acts independently on each
spin sector. For one spin, the single-particle Hamiltonian is a real
symmetric $6 \times 6$ matrix encoding the hopping topology and the
chemical potential.

Implementation (`green_function.py`):

1. Build the kinetic part of the single-particle Hamiltonian
   $H_{\text{kin}} = -t M$ where $M$ is the ring adjacency matrix.
2. Diagonalise numerically with `numpy.linalg.eigh`, obtaining six
   eigenvalues $\{\varepsilon_n\}$ and eigenvectors $\{|n\rangle\}$.
3. Construct the imaginary-time Green's function as the spectral sum

$$G_0(i, j; \tau) = \sum_n \langle i | n \rangle \langle n | j \rangle
                    \, g_{\text{band}}(\varepsilon_n - \mu;\, \tau)$$

where $g_{\text{band}}(\xi; \tau)$ is the standard imaginary-time
Green's function for a single particle with effective single-particle
energy $\xi$, anti-periodic with period $2\beta$.

Implementation lives in `green_function.py`. Verification (in the
`__main__` block of that file) covers:

- diagonal site symmetry at $\mu = 0$,
- correct ring-distance pattern of off-diagonal entries,
- reduction to the atom Green's function in the $t \to 0$ limit,
- agreement with the matrix-exponential construction
  $G_0 = -(I - n_F) e^{-H_{\text{kin}} \tau}$ across a grid of
  $(\beta, \mu, t, \tau)$ values to $< 10^{-12}$.

## Vertex structure for CDet

The CDet recursion for the connected Green's function organises
diagrams by vertex configurations $V = \{(i_k, \tau_k)\}_{k=1}^n$ where
each $(i_k, \tau_k)$ is a vertex site and imaginary time. The vertex
times are integrated over the ordered simplex
$0 \leq \tau_1 \leq \tau_2 \leq \cdots \leq \tau_n \leq \beta$ and the
vertex sites range over the 6 ring sites. See `cdet_recursion.py` for
the algorithm and `wick_determinant.py` for the per-configuration
Wick determinant $D_V$.

## Hilbert space dimension

Six sites with two spins each give 12 spin-orbitals and Hilbert space
dimension $2^{12} = 4096$. The exact diagonalisation reference
(`exact_diagonalization.py`) builds the full $4096 \times 4096$
many-body Hamiltonian via Jordan-Wigner with a sparse representation
and uses `scipy.sparse.linalg.eigsh` for the lowest-energy block.

## Verification ladder

1. free Green's function checks (in `green_function.py`)
2. ED reference via Jordan-Wigner builds the $4096 \times 4096$ many-body $H$
3. Wick determinant $D_V$ verified against the $n = 0$ and $n = 1$ cases
4. CDet recursion $C_V$ verified via vertex permutation invariance and
   site-relabelling consistency
5. end-to-end audit at $n = 0, 1, 2$ comparing CDet output to finite
   differences of $G(U)$ from ED

`tests/test_hexring_cdet.py` exercises steps 1-5.
