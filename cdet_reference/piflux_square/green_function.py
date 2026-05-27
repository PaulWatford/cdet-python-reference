"""
Free Green's function on the 4-site square plaquette
with pi-flux.

Four sites at the corners of a square. The hopping signs are chosen so
that the product of hoppings around the plaquette equals -1 (the
pi-flux gauge condition).

The free single-particle Hamiltonian H_kin = -t * M is built explicitly
as a 4x4 matrix and diagonalised numerically with `numpy.linalg.eigh`.
The imaginary-time Green's function is constructed via the spectral sum

    G_0(i, j; tau) = sum_n <i|n><n|j>  g_band(eps_n - mu; tau)

over the four single-particle eigenstates.
"""
import numpy as np
from functools import lru_cache

from cdet_reference.atom.green_function import G0_atom, n_F


N_SITES = 4

# Hopping matrix: -t factor applied at H_kin time.
# Gauge signs chosen so product around the plaquette = -1 (pi-flux).
_HOP = np.array(
    [[ 0,  1,  0, -1],
     [ 1,  0,  1,  0],
     [ 0,  1,  0,  1],
     [-1,  0,  1,  0]],
    dtype=float,
)


@lru_cache(maxsize=None)
def _H_kin(t):
    """Free single-particle Hamiltonian (kinetic only). 4x4 matrix."""
    return -t * _HOP


@lru_cache(maxsize=None)
def _spectral_data(t):
    """Numerical eigendecomposition of H_kin.

    Returns (eigvals, eigvecs) with eigvecs[:, n] the n-th eigenvector.
    """
    return np.linalg.eigh(_H_kin(t))


def _g_band(tau, beta, xi):
    """Imaginary-time band Green's function with effective exponent xi = eps - mu."""
    return G0_atom(tau, beta, mu=-xi)


def G0_4site_piflux(i, j, tau, beta, mu, t):
    """Free Green's function G_0(i, j; tau) on the 4-site pi-flux plaquette.

    Built from the spectral sum over the eigenstates of H_kin.
    """
    eigvals, eigvecs = _spectral_data(t)
    total = 0.0
    for n in range(N_SITES):
        amp = eigvecs[i, n] * eigvecs[j, n]
        total += amp * _g_band(tau, beta, eigvals[n] - mu)
    return total


def G0_4site_piflux_at_zero_minus(i, j, beta, mu, t):
    """G_0(i, j; 0^-) on the 4-site plaquette via the spectral sum."""
    eigvals, eigvecs = _spectral_data(t)
    total = 0.0
    for n in range(N_SITES):
        amp = eigvecs[i, n] * eigvecs[j, n]
        total += amp * n_F(eigvals[n] - mu, beta)
    return total


def G0_4site_piflux_smart(i, j, ti, tj, beta, mu, t):
    """Equal-time / equal-site convention helper.

    Returns G_0(i, j; ti - tj) for the off-diagonal case, and
    G_0(i, j; 0^-) when (i, ti) == (j, tj).
    """
    if i == j and ti == tj:
        return G0_4site_piflux_at_zero_minus(i, j, beta, mu, t)
    return G0_4site_piflux(i, j, ti - tj, beta, mu, t)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== green_function pi-flux 4-site sanity checks ===\n")

    beta = 4.0
    mu = 0.0
    t = 1.0

    # Spectrum at t=1: should produce 4 real eigenvalues
    eigvals, _ = _spectral_data(t)
    print("Test 1: spectrum of H_kin at t=1, mu=0")
    for n in range(N_SITES):
        print(f"  eps_{n} = {eigvals[n]:+.6f}")
    print()

    # Anti-periodicity
    print("Test 2: Anti-periodicity G_0(tau - beta) = -G_0(tau)")
    worst = 0.0
    for tau in [0.5, 1.5, 3.0]:
        for i in range(N_SITES):
            for j in range(N_SITES):
                g_pos = G0_4site_piflux(i, j, tau, beta, mu, t)
                g_neg = G0_4site_piflux(i, j, tau - beta, beta, mu, t)
                diff = abs(g_pos + g_neg)
                worst = max(worst, diff)
    print(f"  worst |G(tau) + G(tau-beta)| over checks: {worst:.2e}")
    print()

    # Site permutation symmetry (G_ij is symmetric in i, j)
    print("Test 3: G_ij = G_ji")
    worst = 0.0
    for tau in [0.3, 1.7]:
        for i in range(N_SITES):
            for j in range(N_SITES):
                gij = G0_4site_piflux(i, j, tau, beta, mu, t)
                gji = G0_4site_piflux(j, i, tau, beta, mu, t)
                worst = max(worst, abs(gij - gji))
    print(f"  worst |G_ij - G_ji|: {worst:.2e}")
    print()

    # Density at zero minus
    print("Test 4: G_0(i, i; 0^-) at half-filling should equal 0.5 per spin")
    for i in range(N_SITES):
        n_i = G0_4site_piflux_at_zero_minus(i, i, beta, mu=0.0, t=t)
        print(f"  G_0({i},{i}; 0^-) = {n_i:.10f}")
