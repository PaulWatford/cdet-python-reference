"""
Wick determinant D_V for the 4-site pi-flux square Hubbard model.

Vertices are (site, time) pairs with site in {0, 1, 2, 3} and time in (0, beta).

The structure is identical to wick_determinant_2site.py, just with the
4-site Green's function as the underlying propagator.

D_V(x_out, x_in) = (-1)^|V| * det(M_up) * det(M_dn)

where:
  M_up: (|V|+1) x (|V|+1) matrix
    row 0 <- x_out, rows 1..n <- V[k]
    col 0 <- x_in,  cols 1..n <- V[k]
    M_up[a, b] = G_0(site_a, site_b; tau_a - tau_b)
  
  M_dn: |V| x |V| matrix (just the internal vertices)
    M_dn[a, b] = G_0(site_a, site_b; tau_a - tau_b)
    Equal-time, equal-site uses the 0^- convention.

D_V(empty) = (-1)^|V| * det(A)^2 where A = M_dn (vacuum diagram).

KEY pi-FLUX STRUCTURAL FEATURE:
  G_0(i, j; tau) = 0 for (i, j) = (0, 2) or (1, 3) (opposite-corner pairs)
  This means many entries in M_up and M_dn are exactly zero, which can
  collapse det(M) in nontrivial ways.

"""
import numpy as np
from cdet_reference.piflux_square.green_function import G0_4site_piflux, G0_4site_piflux_at_zero_minus


def _build_matrix_4site(rows, cols, beta, mu, t):
    """Build M[a, b] = G_0(rows[a].site, cols[b].site; rows[a].tau - cols[b].tau).
    
    rows, cols: lists of (site, tau) tuples.
    Equal (site, tau) -> 0^- convention.
    """
    n_rows = len(rows)
    n_cols = len(cols)
    M = np.zeros((n_rows, n_cols))
    for a, (ia, ta) in enumerate(rows):
        for b, (ib, tb) in enumerate(cols):
            if ia == ib and ta == tb:
                M[a, b] = G0_4site_piflux_at_zero_minus(ia, ib, beta, mu, t)
            else:
                M[a, b] = G0_4site_piflux(ia, ib, ta - tb, beta, mu, t)
    return M


def D_corr_4site_piflux(V, x_out, x_in, beta, mu, t):
    """D_V(x_out, x_in) for the 4-site pi-flux Hubbard model.
    
    Args:
      V: list of internal vertices, each (site, tau) with site in {0,1,2,3}, tau in (0,beta).
      x_out: (site, tau) for the c_up external operator
      x_in:  (site, tau) for the c_up+ external operator
      beta, mu, t: model parameters
    
    Returns:
      D_V(x_out, x_in) = (-1)^|V| * det(M_up) * det(M_dn)
    """
    V = list(V)
    n = len(V)
    sign = (-1) ** n
    
    rows_up = [x_out] + V
    cols_up = [x_in] + V
    M_up = _build_matrix_4site(rows_up, cols_up, beta, mu, t)
    det_up = np.linalg.det(M_up)
    
    if n == 0:
        det_dn = 1.0
    else:
        M_dn = _build_matrix_4site(V, V, beta, mu, t)
        det_dn = np.linalg.det(M_dn)
    
    return float(sign * det_up * det_dn)


def D_vac_4site_piflux(V, beta, mu, t):
    """D_V(empty) for the 4-site pi-flux Hubbard model.
    
    Vacuum diagram sum at vertex set V (no external operators).
    """
    V = list(V)
    n = len(V)
    sign = (-1) ** n
    
    if n == 0:
        return 1.0
    
    A = _build_matrix_4site(V, V, beta, mu, t)
    det_A = np.linalg.det(A)
    
    return float(sign * det_A * det_A)


# ============================================================================
# Sanity checks
# ============================================================================
if __name__ == "__main__":
    import sys
    from cdet_reference.dimer.wick_determinant import D_corr_2site, D_vac_2site
    from cdet_reference.atom.wick_determinant import D_corr, D_vac
    
    print("=" * 70)
    print("  wick_determinant_4site_piflux.py, sanity checks")
    print("=" * 70)
    print()
    
    beta = 3.0
    mu = 0.3
    t = 0.0  # for t=0 reduction tests
    
    # Test 1: at t = 0, all vertices on site 0, should match atom
    print("Test 1: t=0 + all vertices on site 0 -> reduces to atom CDet")
    print()
    
    x_out_4 = (0, 1.5)
    x_in_4 = (0, 0.5)
    x_out_a = 1.5
    x_in_a = 0.5
    
    # n=0
    d_4 = D_corr_4site_piflux([], x_out_4, x_in_4, beta, mu, t=0.0)
    d_a = D_corr([], x_out_a, x_in_a, beta, mu)
    print(f"  n=0: D_4site = {d_4:.12f}, D_atom = {d_a:.12f}, diff = {abs(d_4-d_a):.2e}")
    
    # n=1
    V_4 = [(0, 1.0)]
    V_a = [1.0]
    d_4 = D_corr_4site_piflux(V_4, x_out_4, x_in_4, beta, mu, t=0.0)
    d_a = D_corr(V_a, x_out_a, x_in_a, beta, mu)
    print(f"  n=1: D_4site = {d_4:.12f}, D_atom = {d_a:.12f}, diff = {abs(d_4-d_a):.2e}")
    
    dv_4 = D_vac_4site_piflux(V_4, beta, mu, t=0.0)
    dv_a = D_vac(V_a, beta, mu)
    print(f"  n=1 vac: D_4site = {dv_4:.12f}, D_atom = {dv_a:.12f}, diff = {abs(dv_4-dv_a):.2e}")
    
    # n=2
    V_4 = [(0, 1.0), (0, 2.0)]
    V_a = [1.0, 2.0]
    d_4 = D_corr_4site_piflux(V_4, x_out_4, x_in_4, beta, mu, t=0.0)
    d_a = D_corr(V_a, x_out_a, x_in_a, beta, mu)
    print(f"  n=2: D_4site = {d_4:.12f}, D_atom = {d_a:.12f}, diff = {abs(d_4-d_a):.2e}")
    
    dv_4 = D_vac_4site_piflux(V_4, beta, mu, t=0.0)
    dv_a = D_vac(V_a, beta, mu)
    print(f"  n=2 vac: D_4site = {dv_4:.12f}, D_atom = {dv_a:.12f}, diff = {abs(dv_4-dv_a):.2e}")
    
    # n=3
    V_4 = [(0, 0.7), (0, 1.5), (0, 2.3)]
    V_a = [0.7, 1.5, 2.3]
    d_4 = D_corr_4site_piflux(V_4, x_out_4, x_in_4, beta, mu, t=0.0)
    d_a = D_corr(V_a, x_out_a, x_in_a, beta, mu)
    print(f"  n=3: D_4site = {d_4:.12f}, D_atom = {d_a:.12f}, diff = {abs(d_4-d_a):.2e}")
    print()
    
    # Test 2: at t = 0, vertex on opposite site disconnects
    print("Test 2: t=0, vertex on isolated site -> product of independent factors")
    print()
    
    # externals on site 0, single vertex on site 2 (isolated since t=0)
    V_4 = [(2, 1.0)]
    d_4 = D_corr_4site_piflux(V_4, x_out_4, x_in_4, beta, mu, t=0.0)
    # Should equal D_empty(externals on 0) * D_vac(vertex on site 2)
    # D_empty(externals at site 0) = G_0_atom(tau_out - tau_in)
    d_zero = D_corr([], x_out_a, x_in_a, beta, mu)
    # D_vac(vertex on site 2 at t=0): vacuum bubble at site 2 = atom vacuum bubble
    dv_atom_at_2 = D_vac([1.0], beta, mu)
    expected = d_zero * dv_atom_at_2
    print(f"  V=[(2, 1.0)], externals at site 0:")
    print(f"    D_4site (computed):  {d_4:.12f}")
    print(f"    G_0_atom * D_vac:    {expected:.12f}")
    print(f"    diff:                {abs(d_4 - expected):.2e}")
    print()
    
    # Test 3: at t = 0, 4-site reduction matches 2-site reduction
    print("Test 3: t=0, vertex on site 0 only -> should also match 2-site result")
    V_4 = [(0, 1.0)]
    V_2 = [(0, 1.0)]
    x_out_2 = (0, 1.5)
    x_in_2 = (0, 0.5)
    d_4 = D_corr_4site_piflux(V_4, x_out_4, x_in_4, beta, mu, t=0.0)
    d_2 = D_corr_2site(V_2, x_out_2, x_in_2, beta, mu, t=0.0)
    print(f"  D_4site = {d_4:.12f}")
    print(f"  D_2site = {d_2:.12f}")
    print(f"  diff =    {abs(d_4 - d_2):.2e}")
    print()
    
    # Test 4: at finite t, D_V depends on hopping
    print("Test 4: At t = 1, D_V depends on t and pi-flux structure")
    t = 1.0
    V_4 = [(0, 1.0)]
    d_4_t0 = D_corr_4site_piflux(V_4, x_out_4, x_in_4, beta, mu, t=0.0)
    d_4_t1 = D_corr_4site_piflux(V_4, x_out_4, x_in_4, beta, mu, t=1.0)
    print(f"  t=0: D = {d_4_t0:.10f}")
    print(f"  t=1: D = {d_4_t1:.10f}")
    print(f"  Different: {abs(d_4_t0 - d_4_t1):.2e}")
    print()
    
    # Test 5: pi-flux structural feature
    # If both externals are at opposite-corner sites (0 and 2), then
    # the n=0 term D_empty(x_out=2, x_in=0) = G_0(0, 2; tau_out - tau_in) = 0
    print("Test 5: opposite-corner externals -> D_empty = 0 (pi-flux structural)")
    x_out_opp = (2, 1.5)  # opposite corner of x_in=(0, 0.5)
    x_in_opp = (0, 0.5)
    d_opp_n0 = D_corr_4site_piflux([], x_out_opp, x_in_opp, beta, mu, t=1.0)
    print(f"  D_empty(x_out=(2,1.5), x_in=(0,0.5)) at t=1: {d_opp_n0:.3e}  (should be 0)")
    
    # With a vertex inserted, the opposite-corner external is not necessarily
    # zero. The pi-flux gauge phase determines the specific cancellations.
    V_opp = [(0, 1.0)]
    d_opp_n1 = D_corr_4site_piflux(V_opp, x_out_opp, x_in_opp, beta, mu, t=1.0)
    print(f"  D_(0,1.0)(x_out=(2,1.5), x_in=(0,0.5)) at t=1: {d_opp_n1:.6e}")
    # This need not be zero: the path through the vertex on site 0 can
    # contribute. But specific paths through opposite-corner pairs in
    # the Wick matrix are zero.
    print()
    
    # Test 6: Permutation invariance of V (vertex ordering shouldn't matter)
    print("Test 6: D_V invariant under vertex ordering")
    V_a = [(0, 1.0), (1, 1.5), (3, 2.0)]
    V_b = [(1, 1.5), (3, 2.0), (0, 1.0)]
    V_c = [(3, 2.0), (0, 1.0), (1, 1.5)]
    d_a = D_corr_4site_piflux(V_a, x_out_4, x_in_4, beta, mu, t=1.0)
    d_b = D_corr_4site_piflux(V_b, x_out_4, x_in_4, beta, mu, t=1.0)
    d_c = D_corr_4site_piflux(V_c, x_out_4, x_in_4, beta, mu, t=1.0)
    print(f"  V = {V_a}: D = {d_a:.12f}")
    print(f"  V = {V_b}: D = {d_b:.12f}")
    print(f"  V = {V_c}: D = {d_c:.12f}")
    max_diff = max(abs(d_a - d_b), abs(d_b - d_c), abs(d_a - d_c))
    print(f"  Max diff: {max_diff:.2e}")
    print()
