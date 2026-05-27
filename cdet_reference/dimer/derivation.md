# 2-site Hubbard dimer: CDet setup

Two sites with nearest-neighbour hopping $t$ and on-site interaction
$U$. The Hamiltonian is

$$H = -t \sum_\sigma (c^\dagger_{0\sigma} c_{1\sigma} + \text{h.c.})
    + U \sum_i n_{i\uparrow} n_{i\downarrow}
    - \mu \sum_{i\sigma} n_{i\sigma}.$$

## Free Green's function

The free single-particle Hamiltonian is the $2 \times 2$ matrix

$$H_{\text{kin}}^{(0)} = \begin{pmatrix} 0 & -t \\ -t & 0 \end{pmatrix}$$

(chemical potential handled separately when assembling the Green's
function). Implementation in `green_function.py`:

1. Diagonalise $H_{\text{kin}}^{(0)}$ with `numpy.linalg.eigh`.
2. Construct $G_0$ via the spectral sum

$$G_0(i, j; \tau) = \sum_n \langle i | n \rangle \langle n | j \rangle
                    \, g_{\text{band}}(\varepsilon_n - \mu;\, \tau)$$

over the two single-particle eigenstates.

Verification (`__main__` block of `green_function.py`):
- $t \to 0$ limit reduces to two decoupled atom Green's functions
- site-swap symmetry $G_0(0, 0) = G_0(1, 1)$ and $G_0(0, 1) = G_0(1, 0)$
- anti-periodicity $G_0(\tau - \beta) = -G_0(\tau)$

## Vertex structure for CDet

CDet on the dimer organises diagrams by vertex configurations
$V = \{(i_k, \tau_k)\}_{k=1}^n$ with $i_k \in \{0, 1\}$ and
$\tau_k \in [0, \beta]$ (ordered). The Wick determinant carries the
$(-1)^{|V|}$ sign and the site labelling of vertices.

## Verification ladder

The dimer's small Hilbert space (16 states) supports arbitrary-precision
references via mpmath (`exact_taylor.py`, `dps = 50`). This makes the
dimer the highest-precision benchmark in the suite, with audits
running to $n = 4$ in $U$.

1. free $G_0$ checks (in `green_function.py`)
2. ED reference (`exact_diagonalization.py`), 16x16 dense
3. mpmath Taylor reference (`exact_taylor.py`), symbolic, dps=50
4. Wick determinant verified against $n = 0, 1$ cases
5. CDet recursion verified against ED finite differences
6. end-to-end audit at $n = 0..4$ against mpmath Taylor coefficients

`tests/test_dimer_cdet*.py` exercise steps 1-6.
