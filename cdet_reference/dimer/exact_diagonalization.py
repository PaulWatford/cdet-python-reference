"""
Exact diagonalization of the 2-site Hubbard model.

Independent of the free Green's function code. Builds the full 16-state
Hilbert space matrix for H = -t sum_sigma (c_0sigma+ c_1sigma + h.c.) + U sum_i n_iup n_idn
- mu sum_i (n_iup + n_idn), then computes everything via matrix exponentials.

This is the ground truth for verifying CDet on the 2-site model.

Basis convention:
  4 single-site states per site: |0>, |up>, |dn>, |updn>
  16 total states for 2 sites
  
  Site state index encoding: occupation patterns
    state 0: (n_up, n_dn) = (0,0)
    state 1: (1,0)  = |up>
    state 2: (0,1)  = |dn>
    state 3: (1,1)  = |up,dn>  = c_up+ c_dn+ |0>  (convention: dn outside)
  
  16 = 4x4 product basis. State index = 4*site_0_state + site_1_state.

"""
import numpy as np
from scipy.linalg import expm


# Basis convention for a SINGLE site (4-state Fock space).
# The c_up, c_dn operators on a single site:
#   c_up takes state with n_up=1 to corresponding n_up=0 state
#   c_dn takes state with n_dn=1 to corresponding n_dn=0 state
# With convention |up,dn> = c_dn+ c_up+ |0> (i.e. dn is the outer creation):
#   c_up |up> = |0>
#   c_up |up,dn> = c_up c_dn+ c_up+ |0> = -c_dn+ c_up c_up+ |0> = -c_dn+ |0> = -|dn>
#   c_dn |dn> = |0>
#   c_dn |up,dn> = c_dn c_dn+ c_up+ |0> = (1 - c_dn+ c_dn) c_up+ |0> = c_up+ |0> = |up>

def single_site_c_up():
    """4x4 matrix for c_up on a single site in basis (|0>, |up>, |dn>, |updn>)."""
    M = np.zeros((4, 4))
    M[0, 1] = 1.0   # c_up |up> = |0>
    M[2, 3] = -1.0  # c_up |updn> = -|dn>
    return M


def single_site_c_dn():
    """4x4 matrix for c_dn on a single site."""
    M = np.zeros((4, 4))
    M[0, 2] = 1.0   # c_dn |dn> = |0>
    M[1, 3] = 1.0   # c_dn |updn> = |up>
    return M


def single_site_n_up():
    """4x4 matrix for n_up."""
    return np.diag([0.0, 1.0, 0.0, 1.0])


def single_site_n_dn():
    """4x4 matrix for n_dn."""
    return np.diag([0.0, 0.0, 1.0, 1.0])


def fermion_parity_single_site():
    """(-1)^(n_up + n_dn) on a single site, used for Jordan-Wigner string."""
    # n_total parity: 0 -> +1, 1 -> -1, 1 -> -1, 2 -> +1
    return np.diag([1.0, -1.0, -1.0, 1.0])


def build_2site_operators():
    """Build all 2-site operators as 16x16 matrices.
    
    The Jordan-Wigner ordering: operators at site 1 must include a
    fermion-parity string from site 0.
    
    Returns a dict with keys:
      'c_up_0', 'c_up_1', 'c_dn_0', 'c_dn_1' , annihilation ops
      'cdag_up_0', 'cdag_up_1', 'cdag_dn_0', 'cdag_dn_1' , creation ops
      'n_up_0', 'n_up_1', 'n_dn_0', 'n_dn_1' , number ops
    """
    I4 = np.eye(4)
    cu = single_site_c_up()
    cd = single_site_c_dn()
    nu = single_site_n_up()
    nd = single_site_n_dn()
    P = fermion_parity_single_site()  # Jordan-Wigner string at site 0
    
    # Site-0 operators: act on site 0 sub-Hilbert, identity on site 1
    # Kronecker: state index = 4*i0 + i1, so op_site0 = kron(op, I)
    ops = {}
    ops['c_up_0'] = np.kron(cu, I4)
    ops['c_dn_0'] = np.kron(cd, I4)
    ops['n_up_0'] = np.kron(nu, I4)
    ops['n_dn_0'] = np.kron(nd, I4)
    
    # Site-1 operators: parity string from site 0 x op at site 1
    ops['c_up_1'] = np.kron(P, cu)
    ops['c_dn_1'] = np.kron(P, cd)
    ops['n_up_1'] = np.kron(I4, nu)
    ops['n_dn_1'] = np.kron(I4, nd)
    
    # Creation operators are transposes (real matrices)
    ops['cdag_up_0'] = ops['c_up_0'].T
    ops['cdag_dn_0'] = ops['c_dn_0'].T
    ops['cdag_up_1'] = ops['c_up_1'].T
    ops['cdag_dn_1'] = ops['c_dn_1'].T
    
    return ops


def build_H(beta, mu, t, U, ops=None):
    """Build the 2-site Hubbard Hamiltonian as a 16x16 matrix.
    
    H = -t sum_sigma (c_{0sigma}+ c_{1sigma} + h.c.) + U sum_i n_iup n_idn - mu sum_i (n_iup + n_idn)
    """
    if ops is None:
        ops = build_2site_operators()
    
    # Hopping: -t (c_0+ c_1 + h.c.) for each spin
    H_hop = -t * (
        ops['cdag_up_0'] @ ops['c_up_1'] + ops['cdag_up_1'] @ ops['c_up_0']
      + ops['cdag_dn_0'] @ ops['c_dn_1'] + ops['cdag_dn_1'] @ ops['c_dn_0']
    )
    
    # On-site U
    H_U = U * (ops['n_up_0'] @ ops['n_dn_0'] + ops['n_up_1'] @ ops['n_dn_1'])
    
    # Chemical potential
    H_mu = -mu * (ops['n_up_0'] + ops['n_dn_0'] + ops['n_up_1'] + ops['n_dn_1'])
    
    return H_hop + H_U + H_mu


def partition_function(beta, mu, t, U):
    """Z = Tr exp(-beta H) via matrix exponential."""
    H = build_H(beta, mu, t, U)
    return np.trace(expm(-beta * H))


def G_exact_2site(i, j, tau, beta, mu, t, U, spin='up'):
    """Exact interacting Green's function for the 2-site Hubbard model.
    
        G^sigma(i, j; tau) = -<T_tau c_{i,sigma}(tau) c_{j,sigma}+(0)>
        
    For 0 < tau < beta:
        G(tau) = -(1/Z) Tr[ exp(-(beta-tau) H) c_{i,sigma} exp(-tau H) c_{j,sigma}+ ]
    """
    ops = build_2site_operators()
    H = build_H(beta, mu, t, U, ops)
    
    Z = np.trace(expm(-beta * H))
    
    # Pick the right c operators
    c_op = ops[f'c_{spin}_{i}']
    cdag_op = ops[f'cdag_{spin}_{j}']
    
    # Fold tau into (0, beta) with antiperiodic sign
    sign = 1.0
    while tau >= beta:
        tau -= beta
        sign *= -1.0
    while tau < 0:
        tau += beta
        sign *= -1.0
    
    # G(tau) = -(1/Z) Tr[exp(-(beta-tau)H) c exp(-tauH) c+]
    e_late = expm(-(beta - tau) * H)
    e_early = expm(-tau * H)
    
    inner = e_late @ c_op @ e_early @ cdag_op
    return sign * (-np.trace(inner) / Z)


def density_2site(beta, mu, t, U, site=0, spin='up'):
    """Exact <n_{i,sigma}> for the 2-site Hubbard model."""
    ops = build_2site_operators()
    H = build_H(beta, mu, t, U, ops)
    Z = np.trace(expm(-beta * H))
    
    n_op = ops[f'n_{spin}_{site}']
    return np.trace(expm(-beta * H) @ n_op) / Z


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== exact_2site.py sanity checks ===\n")
    
    # Test 1: at t=U=0, density per spin should be n_F(-mu)
    print("Test 1: At t=U=0, density should be n_F(-mu)")
    import math
    beta = 5.0
    mu = 0.3
    n = density_2site(beta, mu, t=0.0, U=0.0)
    n_expected = 1.0 / (1.0 + math.exp(beta * mu))  # = n_F(-mu) at beta*(-mu) = ...
    # Actually n_F(xi) with xi = -mu. n_F(-mu) = 1/(1 + e^{-beta*mu}).
    n_expected = 1.0 / (1.0 + math.exp(-beta*mu))
    print(f"  <n_up> at t=U=0: {n:.10f}, expected = n_F(-mu) = {n_expected:.10f}, diff = {abs(n-n_expected):.2e}")
    print()
    
    # Test 2: at U=0 (any t), G(tau) should equal the free G_0
    print("Test 2: At U=0, G_exact_2site = G_0 (free)")
    from cdet_reference.dimer.green_function import G0_2site
    beta = 5.0
    mu = 0.3
    t = 1.0
    for tau in [0.5, 1.5, 3.0]:
        for (i, j) in [(0, 0), (0, 1), (1, 1)]:
            g_ex = G_exact_2site(i, j, tau, beta, mu, t, U=0.0)
            g_0 = G0_2site(i, j, tau, beta, mu, t)
            print(f"  tau={tau}, (i,j)=({i},{j}): G_ex={g_ex:.10f}, G_0={g_0:.10f}, diff={abs(g_ex-g_0):.2e}")
    print()
    
    # Test 3: half-filling at mu=U/2 gives density = 0.5
    print("Test 3: At half-filling mu=U/2, density per spin should be 0.5")
    for U in [0.5, 1.0, 2.0, 4.0]:
        n = density_2site(beta=2.0, mu=U/2, t=1.0, U=U)
        print(f"  U={U}: <n_{{0,up}}> = {n:.10f}, expected 0.5, diff = {abs(n-0.5):.2e}")
    print()
    
    # Test 4: ground state energy at half-filling, large beta
    print("Test 4: Ground state energy at half-filling, large beta")
    # 2-site Hubbard at half-filling has analytic GS energy:
    #   E_0 = U/2 - sqrt((U/2)^2 + 4t^2)
    # This is the lowest singlet, derived from the 2-electron Sz=0 sector.
    for U_val in [0.0, 1.0, 4.0]:
        t_val = 1.0
        beta_val = 50.0
        # Compute lowest eigenvalue from H minus the half-filling chemical potential offset
        ops = build_2site_operators()
        H = build_H(beta_val, mu=U_val/2, t=t_val, U=U_val, ops=ops)
        eigvals = np.linalg.eigvalsh(H)
        E_0_computed = eigvals.min()
        # At half-filling (2 e on 2 sites, n=1 per site), the relevant eigenvalues
        # come from the N=2 sector. With mu=U/2, the chemical-potential offset gives
        # E = E_Hubbard - mu*N = E_Hubbard - U (for N=2)
        # Standard result: E_singlet (Hubbard, no mu) = U/2 - sqrt((U/2)^2 + 4t^2)
        # With mu=U/2 subtracted: E_singlet - U = -U/2 - sqrt((U/2)^2 + 4t^2)
        E_0_analytic = -U_val/2 - math.sqrt((U_val/2)**2 + 4*t_val**2)
        print(f"  U={U_val}, t={t_val}: E_0(computed)={E_0_computed:.10f}, E_0(analytic)={E_0_analytic:.10f}, diff={abs(E_0_computed - E_0_analytic):.2e}")
