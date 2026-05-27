"""
Exact diagonalization reference for the 4-site pi-flux square Hubbard model.

The Hilbert space has dimension 2^8 = 256 (4 sites x 2 spins, each site
can be 0, up, dn, or updn).

This module provides:
    build_4site_operators(), c, c+, n operators in the 256-dim space
    build_H_piflux(beta, mu, t, U), Hamiltonian matrix
    G_exact_4site_piflux(i, j, tau, beta, mu, t, U), exact Green's function
    partition_function_4site_piflux(beta, mu, t, U)
    
Convention: basis ordering is determined by the bit-encoding of state
(b_7 b_6 b_5 b_4 b_3 b_2 b_1 b_0) where:
    b_0 = site 0, up
    b_1 = site 0, dn
    b_2 = site 1, up
    b_3 = site 1, dn
    b_4 = site 2, up
    b_5 = site 2, dn
    b_6 = site 3, up
    b_7 = site 3, dn

So spin sigma on site i is encoded by bit 2i + (0 if up, 1 if dn).

Jordan-Wigner string ordering: the fermion operators anticommute, and 
the sign on c+_alpha acting on a state with bit-pattern n depends on the
parity of the bits BELOW alpha: 
    c+_alpha |n> = (-1)^{p_alpha(n)} |n + (1<<alpha)>  if n & (1<<alpha) == 0
where p_alpha(n) is the number of 1-bits in n[0..alpha-1].

This is the same convention used in exact_2site.py, just extended to 4 sites.

"""
import numpy as np
from functools import lru_cache
from scipy.linalg import expm


DIM = 256  # 2^8 = 256
N_SITES = 4
N_SPIN_ORBITALS = 8  # 4 sites x 2 spins


def spin_orbital_index(site, spin):
    """Return the bit index for site in {0,1,2,3} and spin in {'up', 'dn'}."""
    if spin == 'up':
        return 2 * site
    elif spin == 'dn':
        return 2 * site + 1
    else:
        raise ValueError(f"spin must be 'up' or 'dn', got {spin}")


def jordan_wigner_sign(state, bit_idx):
    """Return (-1)^p where p is the number of occupied orbitals below bit_idx."""
    mask = (1 << bit_idx) - 1
    p = bin(state & mask).count('1')
    return (-1) ** p


def single_orbital_c_dag(bit_idx):
    """
    Construct the matrix for c+_alpha (creation on orbital alpha = bit_idx)
    in the 256-dim basis with Jordan-Wigner sign factors.
    """
    op = np.zeros((DIM, DIM), dtype=float)
    for n in range(DIM):
        if not (n & (1 << bit_idx)):  # orbital alpha empty
            new_state = n | (1 << bit_idx)
            sign = jordan_wigner_sign(n, bit_idx)
            op[new_state, n] = sign
    return op


def single_orbital_c(bit_idx):
    """Construct c_alpha (annihilation on orbital alpha)."""
    op = np.zeros((DIM, DIM), dtype=float)
    for n in range(DIM):
        if n & (1 << bit_idx):  # orbital alpha occupied
            new_state = n ^ (1 << bit_idx)  # remove the bit
            sign = jordan_wigner_sign(n, bit_idx)
            op[new_state, n] = sign
    return op


@lru_cache(maxsize=1)
def build_4site_operators():
    """Build all c, c+, n operators for 4 sites x 2 spins. Cached."""
    ops = {}
    for site in range(N_SITES):
        for spin in ['up', 'dn']:
            idx = spin_orbital_index(site, spin)
            c = single_orbital_c(idx)
            cdag = single_orbital_c_dag(idx)
            ops[f'c_{spin}_{site}'] = c
            ops[f'cdag_{spin}_{site}'] = cdag
            ops[f'n_{spin}_{site}'] = cdag @ c
    return ops


def build_H_piflux(beta, mu, t, U, ops=None):
    """
    Build the Hamiltonian for the 4-site pi-flux square Hubbard model.
    
    H = -t Sum_<ij>sigma s_ij c+_isigma c_jsigma + U sum_i n_iup n_idn - mu sum_isigma n_isigma
    
    where s_ij is the gauge phase: +1 for edges (0,1), (1,2), (2,3); -1 for (3,0).
    
    Note: beta is unused in H itself; included for API symmetry with the
    Green's function calls.
    """
    if ops is None:
        ops = build_4site_operators()
    
    H = np.zeros((DIM, DIM), dtype=float)
    
    # Hopping: edges (0,1), (1,2), (2,3) with sign +1; edge (3,0) with sign -1
    edges = [(0, 1, +1), (1, 2, +1), (2, 3, +1), (3, 0, -1)]
    for i, j, sign in edges:
        for spin in ['up', 'dn']:
            ci = ops[f'c_{spin}_{i}']
            cj = ops[f'c_{spin}_{j}']
            cdi = ops[f'cdag_{spin}_{i}']
            cdj = ops[f'cdag_{spin}_{j}']
            H += -t * sign * (cdi @ cj + cdj @ ci)
    
    # On-site interaction U nup ndn
    for site in range(N_SITES):
        nup = ops[f'n_up_{site}']
        ndn = ops[f'n_dn_{site}']
        H += U * (nup @ ndn)
    
    # Chemical potential
    for site in range(N_SITES):
        H -= mu * (ops[f'n_up_{site}'] + ops[f'n_dn_{site}'])
    
    return H


def partition_function_4site_piflux(beta, mu, t, U):
    """Z = Tr[exp(-beta*H)]."""
    H = build_H_piflux(beta, mu, t, U)
    eigvals = np.linalg.eigvalsh(H)
    # Stable: subtract minimum
    e_min = eigvals.min()
    Z = float(np.sum(np.exp(-beta * (eigvals - e_min)))) * np.exp(-beta * e_min)
    return Z


def G_exact_4site_piflux(i, j, tau, beta, mu, t, U, spin='up', H=None, ops=None):
    """
    Compute G_exact(i, j; tau) = -<T c_isigma(tau) c+_jsigma(0)> for the full interacting
    4-site Hubbard model with pi-flux.
    
    For 0 < tau < beta:
        G(tau) = -(1/Z) Tr[e^{-(beta-tau)H} c_isigma e^{-tauH} c+_jsigma]
    """
    if ops is None:
        ops = build_4site_operators()
    if H is None:
        H = build_H_piflux(beta, mu, t, U, ops=ops)
    
    c_i = ops[f'c_{spin}_{i}']
    cdag_j = ops[f'cdag_{spin}_{j}']
    
    # Diagonalize once for stability
    eigvals, eigvecs = np.linalg.eigh(H)
    e_min = eigvals.min()
    
    # exp(-(beta-tau)H) in eigenbasis: diag(exp(-(beta-tau)(e_k - e_min)))
    # exp(-tau H): diag(exp(-tau(e_k - e_min)))
    
    if tau > 0:
        if tau >= beta:
            raise ValueError(f"tau = {tau} out of range (beta = {beta})")
        
        # Numerator: Tr[e^{-(beta-tau)H} c_i e^{-tauH} c+_j]
        # = Tr[U exp(-(beta-tau)D) U+ c_i U exp(-tauD) U+ c+_j]
        # = Tr[exp(-(beta-tau)D) (U+ c_i U) exp(-tauD) (U+ c+_j U)]
        c_eig = eigvecs.T @ c_i @ eigvecs
        cdag_eig = eigvecs.T @ cdag_j @ eigvecs
        
        # exp factors (with e_min offset for numerical stability)
        exp1 = np.exp(-(beta - tau) * (eigvals - e_min))  # for first exp
        exp2 = np.exp(-tau * (eigvals - e_min))           # for second exp
        
        # Tr[diag(exp1) * c_eig * diag(exp2) * cdag_eig]
        # = sum_{k,l} exp1[k] * c_eig[k,l] * exp2[l] * cdag_eig[l,k]
        numerator = np.einsum('k,kl,l,lk->', exp1, c_eig, exp2, cdag_eig)
        
        # Z (with same offset)
        Z = np.sum(np.exp(-beta * (eigvals - e_min)))
        
        return -float(numerator / Z)
    
    elif tau < 0:
        # G(tau) for tau < 0: G(tau + beta) = -G(tau), so compute G(tau + beta) and negate
        return -G_exact_4site_piflux(i, j, tau + beta, beta, mu, t, U, spin, H, ops)
    
    else:
        # tau = 0: by convention treat as 0+ (limit from above)
        return G_exact_4site_piflux(i, j, 1e-15 * beta, beta, mu, t, U, spin, H, ops)


# ============================================================================
# Sanity checks
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Exact 4-site pi-flux Hubbard ED reference")
    print("=" * 70)
    print()

    # Check 1: Hilbert space dimension
    ops = build_4site_operators()
    print(f"Check 1: operators built. Hilbert space dim = {DIM}")
    print()

    # Check 2: anticommutators
    print("Check 2: anticommutation relations {c_i, c+_j} = delta_ij delta_{sigma,sigma'} I")
    test_pairs = [(0, 0, 'up', 'up'), (0, 0, 'up', 'dn'), (0, 1, 'up', 'up'), (1, 2, 'dn', 'up')]
    for i, j, si, sj in test_pairs:
        c_i = ops[f'c_{si}_{i}']
        cdag_j = ops[f'cdag_{sj}_{j}']
        anticomm = c_i @ cdag_j + cdag_j @ c_i
        expected = (1.0 if (i == j and si == sj) else 0.0) * np.eye(DIM)
        err = np.linalg.norm(anticomm - expected)
        print(f"  {{c_{si}_{i}, cdag_{sj}_{j}}}: error = {err:.2e}")
    print()

    # Check 3: U=0 ground state energy at half-filling (mu=0)
    print("Check 3: ground state energy at U=0, mu=0 (free fermions)")
    print("  Expected: 4 spinless states in lower band (-sqrt(2)*t each), 4 in upper band (+sqrt(2)*t).")
    print("  At half-filling, all 4 lower-band states (2 spatial x 2 spin) are filled.")
    print("  GS energy = 4 * (-sqrt(2) t) = -4*sqrt(2) t")
    H = build_H_piflux(beta=4.0, mu=0.0, t=1.0, U=0.0, ops=ops)
    eigvals = np.linalg.eigvalsh(H)
    print(f"  GS energy (numerical): {eigvals[0]:.10f}")
    print(f"  Expected -4*sqrt(2):         {-4*np.sqrt(2):.10f}")
    print(f"  Difference:            {abs(eigvals[0] - (-4*np.sqrt(2))):.2e}")
    print()

    # Check 4: at U=0, free-fermion partition function check
    print("Check 4: free-fermion partition function at U=0, mu=0")
    print("  Z = Pi_alpha (1 + e^{-beta epsilon_alpha}) over all 8 spin-orbitals")
    print("    With 4 spin-orbitals at -sqrt(2)t and 4 at +sqrt(2)t:")
    print("    Z = (1 + e^{beta sqrt(2) t})^4 (1 + e^{-beta sqrt(2) t})^4 at mu=0")
    print()
    # 4 spatial states x 2 spin states = 8 spin-orbitals total.
    # Spatial spectrum: 2 states at -sqrt(2)t (lower band), 2 at +sqrt(2)t (upper band).
    # With 2 spin copies of each: 4 spin-orbitals at -sqrt(2)t, 4 at +sqrt(2)t.
    # So Z = (1 + e^{beta sqrt(2) t})^4 (1 + e^{-beta sqrt(2) t})^4 at mu=0.
    beta_v, mu_v, t_v, U_v = 4.0, 0.0, 1.0, 0.0
    Z_numerical = partition_function_4site_piflux(beta_v, mu_v, t_v, U_v)
    Z_expected = (1 + np.exp(beta_v * np.sqrt(2) * t_v))**4 * (1 + np.exp(-beta_v * np.sqrt(2) * t_v))**4
    print(f"  Z (numerical):  {Z_numerical:.10e}")
    print(f"  Z (expected):   {Z_expected:.10e}")
    print(f"  Ratio:          {Z_numerical / Z_expected:.10e}")
    print(f"  Difference:     {abs(Z_numerical - Z_expected):.2e}")
    print()

    # Check 5: U=0 Green's function should equal free Green's function
    print("Check 5: U=0 Green's function = free G_0 on the pi-flux square")
    import sys
    from cdet_reference.piflux_square.green_function import G0_4site_piflux
    
    beta_v, mu_v, t_v, U_v = 2.0, 0.3, 1.0, 0.0
    test_cases = [(0,0,0.3), (0,0,1.5), (0,1,0.7), (0,2,1.0), (1,3,1.2), (2,3,0.5)]
    print(f"  beta={beta_v}, mu={mu_v}, t={t_v}, U={U_v}")
    for (i, j, tau) in test_cases:
        g_ed = G_exact_4site_piflux(i, j, tau, beta_v, mu_v, t_v, U_v, ops=ops)
        g_free = G0_4site_piflux(i, j, tau, beta_v, mu_v, t_v)
        diff = abs(g_ed - g_free)
        print(f"  G({i},{j};tau={tau}): ED = {g_ed:+.10f}, free = {g_free:+.10f}, diff = {diff:.2e}")
    print()

    # Check 6: interacting Green's function at small U is U=0 + O(U)
    print("Check 6: G_exact at U=0 vs small U (smoothness check)")
    beta_v, mu_v, t_v = 2.0, 0.0, 1.0
    H_cache = None
    for U_v in [0.0, 0.01, 0.1]:
        g = G_exact_4site_piflux(0, 0, 1.0, beta_v, mu_v, t_v, U_v, ops=ops)
        print(f"  G(0,0;tau=1; U={U_v}) = {g:+.10f}")
    print()
    print("Should be smooth, confirms the ED reference is well-conditioned.")
