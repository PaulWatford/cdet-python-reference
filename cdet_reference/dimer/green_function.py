"""
Free Green's function on the 2-site Hubbard dimer.

The free single-particle Hamiltonian is the 2x2 matrix

    H_kin = [[ -mu,  -t ],
            [ -t,   -mu ]]

acting on the site basis {0, 1} for each spin separately. It is
diagonalised numerically with `numpy.linalg.eigh`, and the imaginary-time
Green's function is constructed via the spectral sum

    G_0(i, j; tau) = sum_n <i|n><n|j>  g_band(eps_n - mu; tau)

where g_band is the per-band imaginary-time Green's function for an
effective single-particle energy.
"""
import numpy as np
from functools import lru_cache

from cdet_reference.atom.green_function import G0_atom, n_F


N_SITES = 2


@lru_cache(maxsize=None)
def _H_kin_zero_mu(t):
    """Free kinetic part of the dimer Hamiltonian (chemical potential separated)."""
    return np.array([[0.0, -t], [-t, 0.0]], dtype=float)


@lru_cache(maxsize=None)
def _spectral_data(t):
    """Numerical diagonalisation of the free kinetic Hamiltonian.

    Returns (eigvals, eigvecs) where eigvecs[:, n] is the n-th eigenvector
    and eigvals[n] the n-th eigenvalue. The chemical potential is added
    later when the Green's function is evaluated, so this depends only on t.
    """
    return np.linalg.eigh(_H_kin_zero_mu(t))


def _g_band(tau, beta, xi):
    """Imaginary-time Green's function for a single band with effective
    single-particle energy xi = eps - mu. Anti-periodic in tau with period 2*beta.
    """
    return G0_atom(tau, beta, mu=-xi)


def G0_2site(i, j, tau, beta, mu, t):
    """Free Green's function G_0(i, j; tau) for the 2-site dimer.

    Built from the spectral sum over the eigenstates of the free kinetic
    Hamiltonian. Anti-periodic in tau.
    """
    eigvals, eigvecs = _spectral_data(t)
    total = 0.0
    for n in range(N_SITES):
        amp = eigvecs[i, n] * eigvecs[j, n]
        total += amp * _g_band(tau, beta, eigvals[n] - mu)
    return total


def G0_2site_at_zero_minus(i, j, beta, mu, t):
    """G_0(i, j; 0^-) for the 2-site dimer via the spectral sum."""
    eigvals, eigvecs = _spectral_data(t)
    total = 0.0
    for n in range(N_SITES):
        amp = eigvecs[i, n] * eigvecs[j, n]
        total += amp * n_F(eigvals[n] - mu, beta)
    return total


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== green_function dimer sanity checks ===\n")

    beta = 5.0
    mu = 0.3
    t = 1.0

    # At t = 0 the model reduces to two decoupled single-site (atom) Green's functions.
    print("Test 1: At t=0, G_hop should be 0 and G_loc should match G_atom")
    for tau in [0.1, 0.5, 1.0, 2.0, 4.5]:
        g_loc = G0_2site(0, 0, tau, beta, mu, t=0.0)
        g_hop = G0_2site(0, 1, tau, beta, mu, t=0.0)
        g_atom = G0_atom(tau, beta, mu)
        print(f"  tau={tau}: G_loc={g_loc:.10f}, G_atom={g_atom:.10f}, "
              f"diff={abs(g_loc - g_atom):.2e}, G_hop={g_hop:.2e}")
    print()

    print("Test 2: With t=1, G_hop is nonzero")
    for tau in [0.1, 0.5, 1.0, 2.0, 4.5]:
        g_loc = G0_2site(0, 0, tau, beta, mu, t)
        g_hop = G0_2site(0, 1, tau, beta, mu, t)
        print(f"  tau={tau}: G_loc={g_loc:.10f}, G_hop={g_hop:.10f}")
    print()

    print("Test 3: Site swap symmetry G_0(0,0) = G_0(1,1) and G_0(0,1) = G_0(1,0)")
    for tau in [0.5, 1.5, 3.7]:
        g00 = G0_2site(0, 0, tau, beta, mu, t)
        g11 = G0_2site(1, 1, tau, beta, mu, t)
        g01 = G0_2site(0, 1, tau, beta, mu, t)
        g10 = G0_2site(1, 0, tau, beta, mu, t)
        print(f"  tau={tau}: |G_00 - G_11| = {abs(g00 - g11):.2e}, "
              f"|G_01 - G_10| = {abs(g01 - g10):.2e}")
    print()

    print("Test 4: Antiperiodicity G_0(tau - beta) = -G_0(tau)")
    for tau in [0.5, 1.5, 3.7]:
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            g_pos = G0_2site(i, j, tau, beta, mu, t)
            g_neg = G0_2site(i, j, tau - beta, beta, mu, t)
            print(f"  tau={tau}, (i,j)=({i},{j}): G={g_pos:.6f}, "
                  f"-G(tau-beta)={-g_neg:.6f}, diff={abs(g_pos + g_neg):.2e}")
    print()

    print("Test 5: G_0(i,j; 0^-) values")
    n_loc = G0_2site_at_zero_minus(0, 0, beta, mu, t)
    n_hop = G0_2site_at_zero_minus(0, 1, beta, mu, t)
    print(f"  G_loc(0^-) = {n_loc:.10f}")
    print(f"  G_hop(0^-) = {n_hop:.10f}")
    print(f"  G_hop(0^-) at t=0 = {G0_2site_at_zero_minus(0, 1, beta, mu, t=0.0):.2e}")
