"""
Wick determinant D_V for the 2-site Hubbard model.

Vertices are now (site, time) pairs: V = [(i_1, tau_1), ..., (i_n, tau_n)].
External operators are (site, time, spin), but since H_int = U n_up n_dn
treats both spins symmetrically per vertex, we compute the spin-up
correlator and the spin-dn contribution factors separately.

D_V(x_out, x_in) = (-1)^|V| * det(M_up) * det(M_dn)

where (for spin-up correlator):

  M_up: (|V|+1) x (|V|+1)
    row 0  <- x_out = (i_out, tau_out)
    rows 1..n <- V[k] = (i_k, tau_k)
    col 0  <- x_in = (i_in, tau_in)
    cols 1..n <- V[k] = (i_k, tau_k)
    M_up[a, b] = G_0(site_a, site_b; tau_a - tau_b)
  
  M_dn: |V| x |V|
    rows and cols both indexed by V
    M_dn[a, b] = G_0(site_a, site_b; tau_a - tau_b)
    diagonal: G_0(i, i; 0^-) = local density n_F(eps_B) + n_F(eps_A)) / 2

D_V(empty) = (-1)^|V| * det(A)^2 where A is the |V|x|V| matrix above for spin-dn
        (equal to M_dn here since no external operators for either spin).

See two_site_derivation.md for the full derivation.

"""
import numpy as np
from cdet_reference.dimer.green_function import G0_2site, G0_2site_at_zero_minus


def _build_matrix_2site(rows, cols, beta, mu, t):
    """Build matrix M[i,j] = G_0(rows[i].site, cols[j].site; rows[i].tau - cols[j].tau).
    
    rows, cols are lists of (site, tau) tuples.
    Equal (site, tau): use 0^- convention via G0_2site_at_zero_minus.
    """
    n_rows = len(rows)
    n_cols = len(cols)
    M = np.zeros((n_rows, n_cols))
    for a, (ia, ta) in enumerate(rows):
        for b, (ib, tb) in enumerate(cols):
            if ia == ib and ta == tb:
                # Equal-time, equal-site: 0^- convention
                M[a, b] = G0_2site_at_zero_minus(ia, ib, beta, mu, t)
            else:
                M[a, b] = G0_2site(ia, ib, ta - tb, beta, mu, t)
    return M


def D_corr_2site(V, x_out, x_in, beta, mu, t):
    """D_V(x_out, x_in) for the 2-site Hubbard model.
    
    Args:
      V:        list of internal vertices, each (site, tau) tuple; n = len(V) (can be 0)
      x_out:    (site, tau) for the c_up external operator
      x_in:     (site, tau) for the c_up+ external operator
      beta, mu, t: Hubbard parameters
    
    Returns:
      D_V(x_out, x_in) = (-1)^|V| * det(M_up) * det(M_dn)
    """
    V = list(V)
    n = len(V)
    sign = (-1)**n
    
    # M_up: (n+1) x (n+1)
    rows_up = [x_out] + V
    cols_up = [x_in] + V
    M_up = _build_matrix_2site(rows_up, cols_up, beta, mu, t)
    det_up = np.linalg.det(M_up)
    
    # M_dn: n x n
    if n == 0:
        det_dn = 1.0
    else:
        M_dn = _build_matrix_2site(V, V, beta, mu, t)
        det_dn = np.linalg.det(M_dn)
    
    return sign * det_up * det_dn


def D_vac_2site(V, beta, mu, t):
    """D_V(empty), vacuum diagram sum at V for the 2-site model.
    
    Same as the atom case but with site-indexed G_0:
      D_V(empty) = (-1)^|V| * det(A)^2
    """
    V = list(V)
    n = len(V)
    sign = (-1)**n
    
    if n == 0:
        return 1.0
    
    A = _build_matrix_2site(V, V, beta, mu, t)
    det_A = np.linalg.det(A)
    
    return sign * det_A * det_A


# ---------------------------------------------------------------------------
# Sanity checks: at t=0 (decoupled atoms) should reduce to atom CDet
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from cdet_reference.atom.wick_determinant import D_corr, D_vac
    
    print("=== wick_determinant_2site.py sanity checks ===\n")
    
    beta = 5.0
    mu = 0.3
    
    # Test 1: at t=0, putting all vertices on site 0 should match atom CDet
    print("Test 1: At t=0, all vertices on site 0, should match atom")
    print()
    
    # n=0
    x_out = (0, 1.5)
    x_in = (0, 0.5)
    d_2s = D_corr_2site([], x_out, x_in, beta, mu, t=0.0)
    d_atom = D_corr([], 1.5, 0.5, beta, mu)
    print(f"  n=0: D_2site = {d_2s:.12f}, D_atom = {d_atom:.12f}, diff = {abs(d_2s-d_atom):.2e}")
    
    # n=1
    V_2s = [(0, 1.0)]
    V_atom = [1.0]
    d_2s_n1 = D_corr_2site(V_2s, x_out, x_in, beta, mu, t=0.0)
    d_atom_n1 = D_corr(V_atom, 1.5, 0.5, beta, mu)
    print(f"  n=1: D_2site = {d_2s_n1:.12f}, D_atom = {d_atom_n1:.12f}, diff = {abs(d_2s_n1-d_atom_n1):.2e}")
    
    dv_2s = D_vac_2site(V_2s, beta, mu, t=0.0)
    dv_atom = D_vac(V_atom, beta, mu)
    print(f"  n=1 vac: D_2site = {dv_2s:.12f}, D_atom = {dv_atom:.12f}, diff = {abs(dv_2s-dv_atom):.2e}")
    
    # n=2: both on site 0
    V_2s = [(0, 1.0), (0, 2.5)]
    V_atom = [1.0, 2.5]
    d_2s_n2 = D_corr_2site(V_2s, x_out, x_in, beta, mu, t=0.0)
    d_atom_n2 = D_corr(V_atom, 1.5, 0.5, beta, mu)
    print(f"  n=2: D_2site = {d_2s_n2:.12f}, D_atom = {d_atom_n2:.12f}, diff = {abs(d_2s_n2-d_atom_n2):.2e}")
    
    dv_2s_n2 = D_vac_2site(V_2s, beta, mu, t=0.0)
    dv_atom_n2 = D_vac(V_atom, beta, mu)
    print(f"  n=2 vac: D_2site = {dv_2s_n2:.12f}, D_atom = {dv_atom_n2:.12f}, diff = {abs(dv_2s_n2-dv_atom_n2):.2e}")
    print()
    
    # Test 2: at t=0, putting one vertex on site 1 disconnects from the external operators at site 0
    # The result should reflect: site 1 vertex contributes a vacuum-bubble-like factor
    # while site 0 vertex contributes to the connected propagator
    print("Test 2: At t=0, t=0 means sites are decoupled. A vertex on the OTHER site")
    print("        gives a factorized contribution.")
    print()
    
    V_2s = [(1, 1.0)]
    d_2s_n1_oppsite = D_corr_2site(V_2s, x_out, x_in, beta, mu, t=0.0)
    # At t=0, site 0 and site 1 are decoupled. The vertex on site 1 doesn't
    # connect to the external operators on site 0 at all. So D_V should equal
    # D_empty(x_out, x_in)_site0 x D_{(1,tau_1)}(empty)_site1
    #   = G_0_atom(tau_out - tau_in) x D_empty(vac at site 1 with one vertex)
    d_zero = D_corr([], 1.5, 0.5, beta, mu)  # G_0 between externals
    d_vac_atom_at_site1 = D_vac([1.0], beta, mu)  # vacuum bubble at site 1
    expected = d_zero * d_vac_atom_at_site1
    print(f"  V=[(1, 1.0)], externals on site 0:")
    print(f"    D_2site (computed): {d_2s_n1_oppsite:.12f}")
    print(f"    G_0 * D_vac_atom:   {expected:.12f}")
    print(f"    diff:               {abs(d_2s_n1_oppsite - expected):.2e}")
    print()
    
    # Test 3: with t>0, things should differ from the atom
    print("Test 3: At t=1, D_V depends on t (no longer matches atom)")
    t = 1.0
    V_2s = [(0, 1.0)]
    d_2s_n1_t1 = D_corr_2site(V_2s, x_out, x_in, beta, mu, t=t)
    print(f"  At t=0: D = {d_2s_n1:.10f}")
    print(f"  At t=1: D = {d_2s_n1_t1:.10f}")
    print(f"  Different: {abs(d_2s_n1 - d_2s_n1_t1):.2e}")
