"""
Free Green's function on the 6-site hexagonal ring.

Six sites in a ring with nearest-neighbour hopping. The free
single-particle Hamiltonian is built explicitly as a 6x6 matrix and
diagonalised numerically with `numpy.linalg.eigh`. The imaginary-time
Green's function is constructed via the spectral sum

    G_0(i, j; tau) = sum_n <i|n><n|j>  g_band(eps_n - mu; tau)

over the six single-particle eigenstates.
"""
import numpy as np
from functools import lru_cache

from cdet_reference.atom.green_function import G0_atom, n_F


N_SITES = 6


@lru_cache(maxsize=None)
def _H_kin(t):
    """Free single-particle Hamiltonian for the 6-site ring (6x6 matrix)."""
    H = np.zeros((N_SITES, N_SITES), dtype=float)
    for i in range(N_SITES):
        j = (i + 1) % N_SITES
        H[i, j] = -t
        H[j, i] = -t
    return H


@lru_cache(maxsize=None)
def _spectral_data(t):
    """Numerical eigendecomposition of H_kin.

    Returns (eigvals, eigvecs) with eigvecs[:, n] the n-th eigenvector.
    """
    return np.linalg.eigh(_H_kin(t))


def _g_band(tau, beta, xi):
    """Imaginary-time band Green's function with effective exponent xi = eps - mu."""
    return G0_atom(tau, beta, mu=-xi)


def G0_hexring(i, j, tau, beta, mu, t):
    """Free Green's function G_0(i, j; tau) on the 6-site ring.

    Built from the spectral sum over the eigenstates of H_kin.
    """
    eigvals, eigvecs = _spectral_data(t)
    total = 0.0
    for n in range(N_SITES):
        amp = eigvecs[i, n] * eigvecs[j, n]
        total += amp * _g_band(tau, beta, eigvals[n] - mu)
    return total


def G0_hexring_at_zero_minus(i, j, beta, mu, t):
    """G_0(i, j; 0^-) on the 6-site ring via the spectral sum."""
    eigvals, eigvecs = _spectral_data(t)
    total = 0.0
    for n in range(N_SITES):
        amp = eigvecs[i, n] * eigvecs[j, n]
        total += amp * n_F(eigvals[n] - mu, beta)
    return total


# =============================================================================
# Sanity checks
# =============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("  Free Green's function on the 6-site hexagonal ring")
    print("=" * 72)
    print()

    # Spectrum
    eigvals, _ = _spectral_data(t=1.0)
    print("Check 1: spectrum of H_kin at t=1")
    for n in range(N_SITES):
        print(f"  eps_{n} = {eigvals[n]:+.6f}")
    print()

    # G_0(i, i; tau) is site-independent at mu = 0 by translational symmetry
    print("Check 2: site-symmetry of G_0(i, i; tau) at mu = 0")
    beta_v, mu_v, t_v = 2.0, 0.0, 1.0
    for tau in [0.3, 0.7, 1.5]:
        g_vals = [G0_hexring(i, i, tau, beta_v, mu_v, t_v) for i in range(N_SITES)]
        spread = max(g_vals) - min(g_vals)
        print(f"  tau={tau}: spread across i={spread:.2e} (should be ~0)")
    print()

    # G_0(i, j) at half-filling: values by ring distance
    print("Check 3: G_0(0, j) by ring distance (beta=2, mu=0.3, t=1, tau=1):")
    beta_v, mu_v, t_v, tau_v = 2.0, 0.3, 1.0, 1.0
    for j in range(N_SITES):
        d = min(abs(j), N_SITES - abs(j))
        g = G0_hexring(0, j, tau_v, beta_v, mu_v, t_v)
        print(f"  G_0(0, {j}) [ring-distance {d}] = {g:+.8f}")
    print()

    # t -> 0 reduces to the atom Green's function
    print("Check 4: t -> 0 limit reduces to atom G_0")
    beta_v, mu_v = 2.0, 0.3
    for tau in [0.3, 1.0, 1.5]:
        g_diag = G0_hexring(0, 0, tau, beta_v, mu_v, t=0.0)
        g_off = G0_hexring(0, 1, tau, beta_v, mu_v, t=0.0)
        g_atom = G0_atom(tau, beta_v, mu_v)
        print(f"  tau={tau}: G_0(0,0)={g_diag:+.8f}, atom={g_atom:+.8f}, "
              f"diff={abs(g_diag - g_atom):.2e}")
        print(f"            G_0(0,1)={g_off:+.2e} (should be 0)")
    print()

    # Verification against direct matrix-exponential construction
    print("Check 5: spectral sum agrees with direct matrix-exponential form")
    from scipy.linalg import expm
    I_mat = np.eye(N_SITES)

    def G0_via_matexp(i, j, tau, beta, mu, t):
        H_sp = _H_kin(t) - mu * I_mat
        nF_mat = np.linalg.inv(I_mat + expm(beta * H_sp))
        if tau > 0:
            return -((I_mat - nF_mat) @ expm(-H_sp * tau))[i, j]
        elif tau < 0:
            return (nF_mat @ expm(-H_sp * tau))[i, j]
        else:
            return -(I_mat - nF_mat)[i, j]

    n_pass, n_total, max_diff = 0, 0, 0.0
    import itertools
    for beta in [1.5, 2.0, 3.0]:
        for mu in [0.0, 0.3, -0.5]:
            for t in [0.5, 1.0, 1.5]:
                for tau in [0.2, 0.7, 1.3]:
                    if tau >= beta:
                        continue
                    for (i, j) in itertools.product(range(N_SITES), range(N_SITES)):
                        g1 = G0_hexring(i, j, tau, beta, mu, t)
                        g2 = G0_via_matexp(i, j, tau, beta, mu, t)
                        diff = abs(g1 - g2)
                        max_diff = max(max_diff, diff)
                        n_total += 1
                        if diff < 1e-12:
                            n_pass += 1
    print(f"  {n_pass}/{n_total} checks pass at < 1e-12, max diff = {max_diff:.2e}")
