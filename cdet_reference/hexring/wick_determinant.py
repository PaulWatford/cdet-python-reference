"""
Wick determinant D_V for the 6-site hexagonal ring Hubbard model.

Vertices are (site, time) with site in {0,1,2,3,4,5} and time in (0, beta).

The construction mirrors the smaller lattices: a (|V|+1)x(|V|+1) matrix
for the contracted spin and a |V|x|V| matrix for the spectator spin.
"""
import numpy as np
from cdet_reference.hexring.green_function import G0_hexring, G0_hexring_at_zero_minus


def _build_matrix_hexring(rows, cols, beta, mu, t):
    """Build M[a, b] = G_0(rows[a].site, cols[b].site; rows[a].tau - cols[b].tau)."""
    n_rows = len(rows)
    n_cols = len(cols)
    M = np.zeros((n_rows, n_cols))
    for a, (ia, ta) in enumerate(rows):
        for b, (ib, tb) in enumerate(cols):
            if ia == ib and ta == tb:
                M[a, b] = G0_hexring_at_zero_minus(ia, ib, beta, mu, t)
            else:
                M[a, b] = G0_hexring(ia, ib, ta - tb, beta, mu, t)
    return M


def D_corr_hexring(V, x_out, x_in, beta, mu, t):
    """D_V(x_out, x_in) = (-1)^|V| * det(M_up) * det(M_dn) for hexring.
    
    V: list of (site, tau) tuples (site in {0..5}, tau in (0, beta))
    x_out, x_in: (site, tau) tuples
    """
    V = list(V)
    n = len(V)
    sign = (-1) ** n
    
    rows_up = [x_out] + V
    cols_up = [x_in] + V
    M_up = _build_matrix_hexring(rows_up, cols_up, beta, mu, t)
    det_up = np.linalg.det(M_up)
    
    if n == 0:
        det_dn = 1.0
    else:
        M_dn = _build_matrix_hexring(V, V, beta, mu, t)
        det_dn = np.linalg.det(M_dn)
    
    return float(sign * det_up * det_dn)


def D_vac_hexring(V, beta, mu, t):
    """D_V(empty) = (-1)^|V| * det(A)^2 for hexring."""
    V = list(V)
    n = len(V)
    sign = (-1) ** n
    
    if n == 0:
        return 1.0
    
    A = _build_matrix_hexring(V, V, beta, mu, t)
    det_A = np.linalg.det(A)
    
    return float(sign * det_A * det_A)


# =============================================================================
# Sanity checks
# =============================================================================
if __name__ == "__main__":
    import sys
    from cdet_reference.atom.wick_determinant import D_corr, D_vac
    
    print("=" * 72)
    print("  Wick determinant on 6-site hexring, sanity checks")
    print("=" * 72)
    print()
    
    beta, mu = 3.0, 0.3
    
    # Test 1: t=0 + all vertices on site 0 -> matches atom
    print("Test 1: t=0 + all vertices on site 0 -> matches atom CDet")
    x_out_h = (0, 1.5)
    x_in_h = (0, 0.5)
    
    for V_atom in [[], [1.0], [1.0, 2.0], [0.7, 1.5, 2.3]]:
        V_h = [(0, tau) for tau in V_atom]
        d_a = D_corr(V_atom, 1.5, 0.5, beta, mu)
        d_h = D_corr_hexring(V_h, x_out_h, x_in_h, beta, mu, t=0.0)
        diff = abs(d_a - d_h)
        print(f"  n={len(V_atom)}: D_atom={d_a:+.12f}, D_hex={d_h:+.12f}, diff={diff:.2e}")
        
        dv_a = D_vac(V_atom, beta, mu)
        dv_h = D_vac_hexring(V_h, beta, mu, t=0.0)
        diff_v = abs(dv_a - dv_h)
        if V_atom:
            print(f"        vac: D_atom={dv_a:+.12f}, D_hex={dv_h:+.12f}, diff={diff_v:.2e}")
    print()
    
    # Test 2: At t=0, vertex on isolated site -> factorizes
    print("Test 2: t=0, single vertex on site 3 (opposite of externals at site 0)")
    V_iso = [(3, 1.0)]
    d_iso = D_corr_hexring(V_iso, x_out_h, x_in_h, beta, mu, t=0.0)
    # Expected: D_empty(externals at 0) x D_vac(vertex at 3)
    d_zero = D_corr([], 1.5, 0.5, beta, mu)
    dv_at_3 = D_vac([1.0], beta, mu)
    expected = d_zero * dv_at_3
    print(f"  D_hex (V=[(3, 1.0)]): {d_iso:+.12f}")
    print(f"  D_empty x D_vac (atom):   {expected:+.12f}")
    print(f"  diff:                  {abs(d_iso - expected):.2e}")
    print()
    
    # Test 3: At finite t, t-dependence
    print("Test 3: At t=1, D_V differs from t=0")
    V = [(0, 1.0)]
    d_t0 = D_corr_hexring(V, x_out_h, x_in_h, beta, mu, t=0.0)
    d_t1 = D_corr_hexring(V, x_out_h, x_in_h, beta, mu, t=1.0)
    print(f"  V=[(0, 1.0)] at t=0: D={d_t0:+.10f}")
    print(f"  V=[(0, 1.0)] at t=1: D={d_t1:+.10f}")
    print(f"  Different: {abs(d_t0 - d_t1):.2e}")
    print()
    
    # Test 4: Permutation invariance
    print("Test 4: D_V invariant under vertex ordering")
    V_a = [(0, 0.5), (2, 1.5), (4, 2.5)]
    V_b = [(2, 1.5), (4, 2.5), (0, 0.5)]
    V_c = [(4, 2.5), (0, 0.5), (2, 1.5)]
    d_a = D_corr_hexring(V_a, x_out_h, x_in_h, beta, mu, t=1.0)
    d_b = D_corr_hexring(V_b, x_out_h, x_in_h, beta, mu, t=1.0)
    d_c = D_corr_hexring(V_c, x_out_h, x_in_h, beta, mu, t=1.0)
    print(f"  V = {V_a}: D = {d_a:+.12f}")
    print(f"  V = {V_b}: D = {d_b:+.12f}")
    print(f"  V = {V_c}: D = {d_c:+.12f}")
    print(f"  Max diff: {max(abs(d_a-d_b), abs(d_b-d_c), abs(d_a-d_c)):.2e}")
