"""
Exact single-particle spectral function A(k, omega) on the 6-site
honeycomb hexring Hubbard model.

This file implements the EXACT reference (from ED) for benchmarking
CDet-derived spectral functions later.

Physics:
  The retarded Green's function in k-space is

      G_R(k, omega) = (1/Z) sum_{m,n} (e^{-beta*E_m} + e^{-beta*E_n}) *
                          |<n|c_ksigma|m>|^2 / (omega - (E_n - E_m) + ieta)

  giving the spectral function

      A(k, omega) = -Im G_R(k, omega + i0+) / pi
              = (1/Z) sum_{m,n} (e^{-beta*E_m} + e^{-beta*E_n}) *
                          |<n|c_ksigma|m>|^2 delta(omega - (E_n - E_m))

  (with a Lorentzian replacing delta for finite broadening eta).

  c_k = (1/sqrt(N)) sum_j e^{2*pi*i*k*j/N} c_j  (Fourier annihilation)

The hexring has 4 distinct band momenta in the |lambda| sense:
  k=0 (Gamma):       eps_k = -2t  (lowest single-particle band)
  k=1, k=5:      eps_k = -t   (K, K' Dirac doublet, lower)
  k=2, k=4:      eps_k = +t   (Dirac doublet, upper)
  k=3 (M):       eps_k = +2t  (highest band)

This is the "unfolded" graphene-like band structure on 6 sites.

sum rule: integral_{-inf}^{inf} A(k, omega) domega = <{c_ksigma, c+_ksigma}> = 1 (for spin-1/2 fermions)
"""
import numpy as np
from scipy.sparse import csr_matrix
from functools import lru_cache


N_SITES = 6
DIM = 4096


# Fourier transformation: c_ksigma = (1/sqrt(N)) sum_j omega_6^{kj} c_jsigma
# where omega_6 = e^{2*pi*i/6}
_omega6 = np.exp(2j * np.pi / N_SITES)


@lru_cache(maxsize=1)
def fourier_matrix():
    """NxN unitary Fourier matrix U[k, j] = omega_6^{k*j} / sqrt(N).
    
    c_k = sum_j U[k, j] c_j (annihilation operator transformation).
    """
    U = np.zeros((N_SITES, N_SITES), dtype=complex)
    for k in range(N_SITES):
        for j in range(N_SITES):
            U[k, j] = _omega6**(k * j) / np.sqrt(N_SITES)
    return U


def band_energy(k, t=1.0):
    """Single-particle band energy eps_k = -2t cos(2pi k / 6).
    
    H_kin = -t Sum c+_i c_j (i,j neighbors) diagonalizes to
        H_kin = sum_k eps_k c+_k c_k  with eps_k = -2t cos(2pi k / 6).
    """
    return -2.0 * t * np.cos(2 * np.pi * k / N_SITES)


def c_k_operator(k, spin, ops):
    """Construct c_ksigma as a sparse 4096x4096 matrix.
    
    c_ksigma = (1/sqrt(N)) sum_j omega_6^{k*j} c_jsigma
    
    Args:
        k: momentum index in {0..5}
        spin: 'up' or 'dn'
        ops: dict from build_hexring_operators()
    
    Returns:
        Sparse complex matrix (4096, 4096).
    """
    U = fourier_matrix()
    # Build dense complex result, then sparsify
    c_k_dense = np.zeros((DIM, DIM), dtype=complex)
    for j in range(N_SITES):
        c_k_dense = c_k_dense + U[k, j] * ops[f'c_{spin}_{j}'].toarray()
    return csr_matrix(c_k_dense)


def cdag_k_operator(k, spin, ops):
    """Construct c+_ksigma as a sparse 4096x4096 matrix.
    
    c+_ksigma = (1/sqrt(N)) sum_j omega_6^{-k*j} c+_jsigma = (c_ksigma)+
    """
    return c_k_operator(k, spin, ops).conj().T.tocsr()


def spectral_function_exact(
    k, omega_range, beta, mu, t, U,
    spin='up', eta=0.05, eigh_cache=None, ops=None
):
    """
    Exact spectral function A(k, omega) via the Lehmann representation.
    
    A(k, omega) = (1/Z) sum_{m,n} (e^{-beta*E_m} + e^{-beta*E_n}) |<n|c_ksigma|m>|^2 
              * L(omega - (E_n - E_m), eta)
    
    where L(x, eta) = (1/pi) * eta/(x^2 + eta^2) is a Lorentzian of width eta.
    
    Args:
        k: momentum index in {0..5}
        omega_range: array of omega values to evaluate
        beta, mu, t, U: Hubbard parameters
        spin: 'up' or 'dn'
        eta: Lorentzian broadening (real frequency smearing). The exact
            A(k, omega) is a sum of delta functions at the eigenvalue
            differences; eta replaces each delta with a Lorentzian of
            half-width-at-half-max eta so the result is plottable.
            Typical values 0.02-0.1 in units of t.
        eigh_cache: optional precomputed (eigvals, eigvecs)
        ops: optional precomputed operators dict
    
    Returns:
        A(k, omega) array of same shape as omega_range. Real, non-negative,
        with sum rule integral A(k, omega) domega = 1 (with slight loss to broadening tails
        when omega_range is finite).
    """
    if ops is None:
        from cdet_reference.hexring.exact_diagonalization import build_hexring_operators
        ops = build_hexring_operators()
    
    if eigh_cache is None:
        from cdet_reference.hexring.exact_diagonalization import build_H_hexring
        H = build_H_hexring(beta, mu, t, U, ops=ops).toarray()
        eigvals, eigvecs = np.linalg.eigh(H)
    else:
        eigvals, eigvecs = eigh_cache
    
    # Build c_k operator
    c_k = c_k_operator(k, spin, ops).toarray()
    
    # Transform c_k to eigenbasis: c_k_eig[m, n] = <m|c_k|n>
    # In eigenbasis of H: c_k_eig = V+ * c_k * V where V = eigvecs
    c_k_eig = eigvecs.conj().T @ c_k @ eigvecs
    
    # Boltzmann weights, shifted by e_min for stability
    e_min = eigvals.min()
    boltz = np.exp(-beta * (eigvals - e_min))
    Z = boltz.sum()
    
    # sum over all (m, n) pairs
    # |<n|c_k|m>|^2 * (e^{-beta*E_m} + e^{-beta*E_n}) at energy difference E_n - E_m
    # 
    # In our convention, c_k acts as: c_k|m> = sum_n |n> <n|c_k|m>
    # so <n|c_k|m> = c_k_eig[n, m].
    
    A = np.zeros_like(omega_range, dtype=float)
    for m in range(DIM):
        if boltz[m] < 1e-15:
            continue  # negligible contribution
        for n in range(DIM):
            matrix_elem_sq = abs(c_k_eig[n, m])**2
            if matrix_elem_sq < 1e-15:
                continue
            weight = (boltz[m] + boltz[n]) / Z
            dE = eigvals[m] - eigvals[n]  # Mahan convention: pole at omega = E_m - E_n where N_m > N_n
            # Add Lorentzian-broadened delta function
            A = A + weight * matrix_elem_sq * eta / (np.pi * ((omega_range - dE)**2 + eta**2))
    
    return A


def density_of_states(omega_range, beta, mu, t, U, spin='up',
                       eta=0.05, eigh_cache=None, ops=None):
    """Total density of states D(omega) = (1/N) sum_k A(k, omega) (averaged over k)."""
    if ops is None:
        from cdet_reference.hexring.exact_diagonalization import build_hexring_operators
        ops = build_hexring_operators()
    
    if eigh_cache is None:
        from cdet_reference.hexring.exact_diagonalization import build_H_hexring
        H = build_H_hexring(beta, mu, t, U, ops=ops).toarray()
        eigh_cache = np.linalg.eigh(H)
    
    D = np.zeros_like(omega_range, dtype=float)
    for k in range(N_SITES):
        A_k = spectral_function_exact(k, omega_range, beta, mu, t, U,
                                        spin=spin, eta=eta,
                                        eigh_cache=eigh_cache, ops=ops)
        D = D + A_k / N_SITES
    return D


if __name__ == "__main__":
    import time
    print("=" * 72)
    print("  Exact spectral function A(k, omega) on the 6-site hexring")
    print("=" * 72)
    print()
    
    # Sanity: free band positions
    print("Single-particle band energies eps_k = -2t cos(2pi k/6) at t=1:")
    for k in range(N_SITES):
        eps_k = band_energy(k, t=1.0)
        print(f"  k={k}: eps_k = {eps_k:+.3f}")
    print()
    print("Expected: 4 distinct bands at eps in {-2, -1, +1, +2}")
    print("  k=0: -2 (Gamma, lowest)")
    print("  k=1, 5: -1 (Dirac, lower)  <- K, K' valleys")
    print("  k=2, 4: +1 (Dirac, upper)")
    print("  k=3: +2 (M, highest)")
    print()
    
    # Sanity: Fourier matrix unitary
    U = fourier_matrix()
    err = np.linalg.norm(U @ U.conj().T - np.eye(N_SITES))
    print(f"Fourier matrix unitarity: ||U*U+ - I|| = {err:.2e}")
    print()
    
    # Sanity: free U=0 spectral function: should be sharp peaks at band energies
    print("Sanity check 1: A(k=0, omega) at U=0, beta=8, should peak at omega = -2 (Gamma band)")
    from cdet_reference.hexring.exact_diagonalization import build_hexring_operators, build_H_hexring
    ops = build_hexring_operators()
    
    beta = 8.0  # low temperature for sharp peaks
    mu = 0.0    # half-filling
    t = 1.0
    
    t0 = time.time()
    H = build_H_hexring(beta, mu, t, 0.0, ops=ops).toarray()
    eigvals, eigvecs = np.linalg.eigh(H)
    print(f"  ED for U=0 ({time.time()-t0:.1f}s)")
    print(f"  ED ground state energy: {eigvals[0]:.6f} (expected -8t = -8)")
    print()
    
    omega = np.linspace(-4, 4, 401)
    t0 = time.time()
    A_k0 = spectral_function_exact(0, omega, beta, mu, t, 0.0, eta=0.05,
                                    eigh_cache=(eigvals, eigvecs), ops=ops)
    print(f"  Compute A(k=0, omega) ({time.time()-t0:.1f}s)")
    
    # Locate peaks
    peak_idx = np.argmax(A_k0)
    peak_omega = omega[peak_idx]
    peak_A = A_k0[peak_idx]
    print(f"  Peak at omega = {peak_omega:+.3f}, A = {peak_A:.3f}")
    
    # sum rule check
    dw = omega[1] - omega[0]
    integral = np.sum(A_k0) * dw
    print(f"  integral A(k=0, omega) domega = {integral:.4f} (expected 1.0)")
    print()
    
    # Check 2: A(k=1, omega) should peak at omega = -1 (K Dirac point)
    print("Sanity check 2: A(k=1, omega) at U=0, should peak at omega = -1 (K Dirac)")
    A_k1 = spectral_function_exact(1, omega, beta, mu, t, 0.0, eta=0.05,
                                    eigh_cache=(eigvals, eigvecs), ops=ops)
    peak_idx = np.argmax(A_k1)
    peak_omega = omega[peak_idx]
    peak_A = A_k1[peak_idx]
    print(f"  Peak at omega = {peak_omega:+.3f}, A = {peak_A:.3f}")
    integral = np.sum(A_k1) * dw
    print(f"  integral A(k=1, omega) domega = {integral:.4f}")
    print()
    
    # Check 3: A(k=3, omega) should peak at omega = +2 (M point)
    print("Sanity check 3: A(k=3, omega) at U=0, should peak at omega = +2 (M point)")
    A_k3 = spectral_function_exact(3, omega, beta, mu, t, 0.0, eta=0.05,
                                    eigh_cache=(eigvals, eigvecs), ops=ops)
    peak_idx = np.argmax(A_k3)
    peak_omega = omega[peak_idx]
    peak_A = A_k3[peak_idx]
    print(f"  Peak at omega = {peak_omega:+.3f}, A = {peak_A:.3f}")
    integral = np.sum(A_k3) * dw
    print(f"  integral A(k=3, omega) domega = {integral:.4f}")
    print()
    
    # Check 4: density of states (total)
    print("Sanity check 4: total DOS D(omega) at U=0, should have peaks at omega in {-2,-1,1,2}")
    D = density_of_states(omega, beta, mu, t, 0.0, eta=0.05,
                            eigh_cache=(eigvals, eigvecs), ops=ops)
    print(f"  integral D(omega) domega = {np.sum(D) * dw:.4f}")
    print()
    print("  D(omega) at expected peak positions:")
    for omega_test in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        idx = np.argmin(abs(omega - omega_test))
        print(f"    omega = {omega_test:+.1f}: D = {D[idx]:.4f}")
    print()
    print("(For U=0 at low T, weights should be 1/6 (Gamma), 2/6 (Dirac lower),")
    print(" 2/6 (Dirac upper), 1/6 (M) when integrating in omega-window around peak)")
