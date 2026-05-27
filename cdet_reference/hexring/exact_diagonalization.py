"""
Exact diagonalization reference for the 6-site hexagonal ring Hubbard model.

The Hilbert space has dimension 2^12 = 4096 (6 sites x 2 spins).

Bit encoding: bit (2i + sigma) corresponds to (site i in {0..5}, spin sigma in {0=up, 1=dn}).

This module provides:
    build_hexring_operators(), sparse c, c+, n operators in 4096-dim space
    build_H_hexring(beta, mu, t, U)        , sparse Hamiltonian
    G_exact_hexring(i, j, tau, ...)         , exact Green's function
    partition_function_hexring(...)
"""
import numpy as np
from functools import lru_cache
from scipy.sparse import csr_matrix
from scipy.linalg import expm


DIM = 4096
N_SITES = 6
N_SPIN_ORBITALS = 12


def jordan_wigner_sign(state, bit_idx):
    """Return (-1)^p where p is the number of 1-bits below bit_idx."""
    mask = (1 << bit_idx) - 1
    return (-1) ** bin(state & mask).count('1')


def spin_orbital_index(site, spin):
    """site in {0..5}, spin in {'up', 'dn'}."""
    if spin == 'up':
        return 2 * site
    elif spin == 'dn':
        return 2 * site + 1
    else:
        raise ValueError(f"spin must be 'up' or 'dn', got {spin}")


def _single_orbital_c_dag_sparse(bit_idx):
    """c+_alpha as sparse matrix (4096 x 4096)."""
    rows, cols, vals = [], [], []
    for n in range(DIM):
        if not (n & (1 << bit_idx)):
            new_state = n | (1 << bit_idx)
            sign = jordan_wigner_sign(n, bit_idx)
            rows.append(new_state)
            cols.append(n)
            vals.append(sign)
    return csr_matrix((vals, (rows, cols)), shape=(DIM, DIM), dtype=float)


@lru_cache(maxsize=1)
def build_hexring_operators():
    """Build all c, c+, n operators as sparse matrices. Cached.
    
    Returns dict with keys 'c_up_0', 'cdag_up_0', 'n_up_0', etc.
    """
    ops = {}
    for site in range(N_SITES):
        for spin in ['up', 'dn']:
            bit = spin_orbital_index(site, spin)
            cdag = _single_orbital_c_dag_sparse(bit)
            c = cdag.T.tocsr()  # c is the transpose of c+
            n = (cdag @ c).tocsr()
            ops[f'cdag_{spin}_{site}'] = cdag
            ops[f'c_{spin}_{site}'] = c
            ops[f'n_{spin}_{site}'] = n
    return ops


# Edges of the hexagon: (0,1), (1,2), (2,3), (3,4), (4,5), (5,0)
HEX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]


def build_H_hexring(beta, mu, t, U, ops=None):
    """Sparse Hamiltonian for the 6-site hexring Hubbard model.
    
    H = -t Sum_<ij>sigma (c+_isigma c_jsigma + h.c.) + U sum_i n_iup n_idn - mu sum_isigma n_isigma
    """
    if ops is None:
        ops = build_hexring_operators()
    
    H = csr_matrix((DIM, DIM), dtype=float)
    
    # Hopping
    for i, j in HEX_EDGES:
        for spin in ['up', 'dn']:
            cdag_i = ops[f'cdag_{spin}_{i}']
            c_j = ops[f'c_{spin}_{j}']
            cdag_j = ops[f'cdag_{spin}_{j}']
            c_i = ops[f'c_{spin}_{i}']
            H = H - t * (cdag_i @ c_j + cdag_j @ c_i)
    
    # On-site interaction
    for site in range(N_SITES):
        H = H + U * (ops[f'n_up_{site}'] @ ops[f'n_dn_{site}'])
    
    # Chemical potential
    for site in range(N_SITES):
        H = H - mu * (ops[f'n_up_{site}'] + ops[f'n_dn_{site}'])
    
    return H


def partition_function_hexring(beta, mu, t, U):
    """Z = Tr[exp(-beta*H)] via full ED."""
    H = build_H_hexring(beta, mu, t, U).toarray()
    eigvals = np.linalg.eigvalsh(H)
    e_min = eigvals.min()
    Z = float(np.sum(np.exp(-beta * (eigvals - e_min)))) * np.exp(-beta * e_min)
    return Z


def G_exact_hexring(i, j, tau, beta, mu, t, U, spin='up', H=None, ops=None,
                     eigh_cache=None):
    """
    Compute G_exact(i, j; tau) = -<T c_isigma(tau) c+_jsigma(0)> for the 6-site hexring
    Hubbard model.
    
    For 0 < tau < beta:
        G(tau) = -(1/Z) Tr[e^{-(beta-tau)H} c_isigma e^{-tauH} c+_jsigma]
    
    Args:
        eigh_cache: optional precomputed (eigvals, eigvecs). Avoids re-diagonalising.
    """
    if ops is None:
        ops = build_hexring_operators()
    if H is None:
        H = build_H_hexring(beta, mu, t, U, ops=ops)
    
    # Convert to dense if sparse for eigh
    if hasattr(H, 'toarray'):
        H_dense = H.toarray()
    else:
        H_dense = H
    
    if eigh_cache is not None:
        eigvals, eigvecs = eigh_cache
    else:
        eigvals, eigvecs = np.linalg.eigh(H_dense)
    
    c_i = ops[f'c_{spin}_{i}'].toarray()
    cdag_j = ops[f'cdag_{spin}_{j}'].toarray()
    
    c_eig = eigvecs.T @ c_i @ eigvecs
    cdag_eig = eigvecs.T @ cdag_j @ eigvecs
    
    e_min = eigvals.min()
    
    if tau > 0:
        if tau >= beta:
            raise ValueError(f"tau = {tau} out of range (beta = {beta})")
        exp1 = np.exp(-(beta - tau) * (eigvals - e_min))
        exp2 = np.exp(-tau * (eigvals - e_min))
        numerator = np.einsum('k,kl,l,lk->', exp1, c_eig, exp2, cdag_eig)
        Z = np.sum(np.exp(-beta * (eigvals - e_min)))
        return -float(numerator / Z)
    elif tau < 0:
        return -G_exact_hexring(i, j, tau + beta, beta, mu, t, U, spin, H, ops,
                                  eigh_cache=eigh_cache)
    else:
        return G_exact_hexring(i, j, 1e-15 * beta, beta, mu, t, U, spin, H, ops,
                                 eigh_cache=eigh_cache)


# =============================================================================
# Sanity checks
# =============================================================================
if __name__ == "__main__":
    import time
    
    print("=" * 72)
    print("  Exact 6-site hexring Hubbard ED reference")
    print("=" * 72)
    print()
    
    # Check 1: Hilbert space and operators
    t0 = time.time()
    ops = build_hexring_operators()
    print(f"Check 1: built 36 operators in 4096-dim space ({time.time()-t0:.1f}s)")
    print()
    
    # Check 2: anticommutation
    print("Check 2: anticommutation {c, c+} = delta I")
    pairs = [(0, 0, 'up', 'up'), (0, 0, 'up', 'dn'), (0, 1, 'up', 'up'),
             (2, 3, 'dn', 'up'), (5, 5, 'dn', 'dn')]
    for i, j, si, sj in pairs:
        c_i = ops[f'c_{si}_{i}']
        cdag_j = ops[f'cdag_{sj}_{j}']
        anticomm = (c_i @ cdag_j + cdag_j @ c_i).toarray()
        expected = (1.0 if (i == j and si == sj) else 0.0) * np.eye(DIM)
        err = np.linalg.norm(anticomm - expected)
        print(f"  {{c_{si}_{i}, cdag_{sj}_{j}}}: error = {err:.2e}")
    print()
    
    # Check 3: U=0 GS energy at half-filling (mu=0)
    print("Check 3: U=0 ground state energy at half-filling (mu=0)")
    print("  Single-particle spectrum: {-2t (x2), -t (x4), -t (x4), +t (x4), +t (x4), +2t (x2)}")
    print("  At half-filling (6 electrons), fill the 6 lowest spin-orbitals:")
    print("    2 at -2t  + 4 at -t  = -4t - 4t = -8t")
    
    t0 = time.time()
    H = build_H_hexring(beta=4.0, mu=0.0, t=1.0, U=0.0, ops=ops)
    H_dense = H.toarray()
    eigvals_u0 = np.linalg.eigvalsh(H_dense)
    e_gs_numerical = eigvals_u0[0]
    print(f"  ED GS energy: {e_gs_numerical:.10f} ({time.time()-t0:.1f}s)")
    print(f"  Expected -8t: {-8.0:.10f}")
    print(f"  Difference:    {abs(e_gs_numerical + 8):.2e}")
    print()
    
    # Check 4: partition function matches free-fermion formula
    print("Check 4: free-fermion partition function at U=0, mu=0")
    print("  Z_free = Pi_alpha (1 + e^{-beta*eps_alpha}) over 12 spin-orbitals")
    print("        = (1+e^{2betat})^2*(1+e^{betat})^4*(1+e^{-betat})^4*(1+e^{-2betat})^2")
    
    beta_v, t_v = 4.0, 1.0
    Z_ed = float(np.sum(np.exp(-beta_v * (eigvals_u0 - eigvals_u0.min()))) * 
                 np.exp(-beta_v * eigvals_u0.min()))
    
    # Expected via product formula
    Z_expected = (
        (1 + np.exp(2*beta_v*t_v))**2 *
        (1 + np.exp(beta_v*t_v))**4 *
        (1 + np.exp(-beta_v*t_v))**4 *
        (1 + np.exp(-2*beta_v*t_v))**2
    )
    print(f"  Z (ED):       {Z_ed:.6e}")
    print(f"  Z (expected): {Z_expected:.6e}")
    print(f"  Relative diff: {abs(Z_ed - Z_expected) / Z_expected:.2e}")
    print()
    
    # Check 5: U=0 Green's function = free G_0 on hexring
    print("Check 5: U=0 G_exact matches free G_0 on hexring")
    from cdet_reference.hexring.green_function import G0_hexring
    
    beta_v, mu_v, t_v = 2.0, 0.3, 1.0
    H = build_H_hexring(beta_v, mu_v, t_v, 0.0, ops=ops)
    H_dense = H.toarray()
    eigh_cache = np.linalg.eigh(H_dense)
    
    test_cases = [(0, 0, 0.3), (0, 0, 1.5), (0, 1, 0.7), (0, 2, 1.0),
                  (0, 3, 1.2), (3, 5, 0.5)]
    
    print(f"  beta={beta_v}, mu={mu_v}, t={t_v}, U=0")
    for (i, j, tau) in test_cases:
        g_ed = G_exact_hexring(i, j, tau, beta_v, mu_v, t_v, 0.0, ops=ops,
                                 eigh_cache=eigh_cache)
        g_free = G0_hexring(i, j, tau, beta_v, mu_v, t_v)
        diff = abs(g_ed - g_free)
        print(f"  G({i},{j};tau={tau}): ED={g_ed:+.10f}, free={g_free:+.10f}, diff={diff:.2e}")
    print()
    
    # Check 6: G_exact smooth in U
    print("Check 6: G_exact smooth in U")
    beta_v, mu_v, t_v = 2.0, 0.0, 1.0
    for U_v in [0.0, 0.01, 0.1, 0.5]:
        H = build_H_hexring(beta_v, mu_v, t_v, U_v, ops=ops)
        H_dense = H.toarray()
        eigh_cache = np.linalg.eigh(H_dense)
        g = G_exact_hexring(0, 0, 1.0, beta_v, mu_v, t_v, U_v, ops=ops,
                              eigh_cache=eigh_cache)
        print(f"  G(0,0;tau=1; U={U_v}) = {g:+.10f}")
