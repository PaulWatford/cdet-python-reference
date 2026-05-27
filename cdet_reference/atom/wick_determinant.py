"""
D_V quantities for the CDet recursion on the Hubbard atom.

Implements:
  D_V(x_out, x_in) = (-1)^|V| * det(M_up) * det(M_dn)
  D_V(empty)           = (-1)^|V| * det(A)^2

where the matrices are built from the free Green's function G_0 of the
Hubbard atom (see wick_derivation.md for the full derivation
and sign-pinning argument).

For the Hubbard atom, vertices are imaginary times tau in [0, beta]. There is
no spatial index. Equal-time diagonal entries G_0(tau_i - tau_i) use the
0^- convention: G_0(0^-) = n_F(-mu) = <n>_0.

The convention is fixed by demanding that the CDet recursion (in
cdet_recursion.py) produces Taylor coefficients of G_exact_atom in U
that match the analytic formula to machine precision.

"""
import numpy as np
import sys
from cdet_reference.atom.green_function import G0_atom, G0_atom_at_zero_minus


def _build_matrix(rows_tau, cols_tau, beta, mu):
    """Build a matrix M with M[i,j] = G_0(rows_tau[i] - cols_tau[j]).
    
    Equal-row-column entries (i.e., when rows_tau[i] == cols_tau[j] EXACTLY)
    use the 0^- convention.
    
    Note: G0_atom returns G_0(0^-) at exactly tau=0, so no special handling
    needed as long as the input times are passed unchanged.
    """
    n_rows = len(rows_tau)
    n_cols = len(cols_tau)
    M = np.zeros((n_rows, n_cols))
    for i, t_r in enumerate(rows_tau):
        for j, t_c in enumerate(cols_tau):
            dtau = t_r - t_c
            if dtau == 0.0:
                # Equal-time: use G_0(0^-) = n_0
                M[i, j] = G0_atom_at_zero_minus(beta, mu)
            else:
                M[i, j] = G0_atom(dtau, beta, mu)
    return M


def D_corr(V, tau_out, tau_in, beta, mu):
    """Compute D_V(x_out, x_in) for the correlator (two-point function).
    
    Args:
      V:        list or array of internal vertex times (length n, can be empty)
      tau_out:  external time of the c_up operator
      tau_in:   external time of the c_up+ operator
      beta, mu: temperature inverse and chemical potential
    
    Returns:
      D_V(x_out, x_in) = (-1)^|V| * det(M_up) * det(M_dn)
      
    Where:
      M_up is (|V|+1) x (|V|+1)
        row 0 <- tau_out, rows 1..n <- V
        col 0 <- tau_in,  cols 1..n <- V
        entries G_0(tau^row - tau^col)
      
      M_dn is |V| x |V|
        rows and cols indexed by V
        entries G_0(tau_i - tau_j); diagonal = G_0(0^-)
    
    Note: U^|V| is NOT included here; multiply externally when assembling
    the full perturbative coefficient.
    """
    V = list(V)
    n = len(V)
    # The (-1)^n absorbs the (-U)^n from expanding exp(-U H_int) in the
    # interaction picture. The U^n factor itself is kept outside D_V.
    sign = (-1)**n
    
    # M_up: (n+1) x (n+1)
    rows_up = [tau_out] + V
    cols_up = [tau_in] + V
    M_up = _build_matrix(rows_up, cols_up, beta, mu)
    det_up = np.linalg.det(M_up)
    
    # M_dn: n x n  (empty if n=0)
    if n == 0:
        det_dn = 1.0  # determinant of 0x0 matrix is 1 by convention
    else:
        M_dn = _build_matrix(V, V, beta, mu)
        det_dn = np.linalg.det(M_dn)
    
    return sign * det_up * det_dn


def D_vac(V, beta, mu):
    """Compute D_V(empty), the vacuum diagram sum at vertex set V.
    
    Args:
      V:        list of vertex times (length n)
      beta, mu: temperature inverse and chemical potential
    
    Returns:
      D_V(empty) = (-1)^|V| * det(A)^2
      
    Where A is the |V|x|V| matrix with A[i,j] = G_0(tau_i - tau_j).
    Both spins contribute the same determinant (no external operators),
    so it's squared.
    """
    V = list(V)
    n = len(V)
    # Same sign convention as D_corr (see comment there).
    sign = (-1)**n
    
    if n == 0:
        return 1.0  # No vertices = no vacuum diagram = 1 (empty product)
    
    A = _build_matrix(V, V, beta, mu)
    det_A = np.linalg.det(A)
    
    return sign * det_A * det_A


# ===========================================================================
# Sanity checks (the full audit lives in tests/test_atom_wick_determinant.py)
# ===========================================================================

if __name__ == "__main__":
    print("=== wick_determinant.py basic sanity checks ===\n")
    
    beta = 5.0
    mu = 0.3
    
    # n=0: D_empty(x_out, x_in) should equal G_0(tau_out - tau_in)
    tau_out, tau_in = 1.5, 0.5
    d0 = D_corr([], tau_out, tau_in, beta, mu)
    g0 = G0_atom(tau_out - tau_in, beta, mu)
    print(f"n=0 correlator: D_empty = {d0:.10f}, G_0 = {g0:.10f}, diff = {abs(d0-g0):.2e}")
    
    # n=0 vacuum: D_empty(empty) = 1
    d0v = D_vac([], beta, mu)
    print(f"n=0 vacuum:     D_empty(empty) = {d0v}, expected 1.0")
    print()
    
    # n=1: explicit formula
    tau_1 = 1.0
    g0_oi = G0_atom(tau_out - tau_in, beta, mu)
    g0_o1 = G0_atom(tau_out - tau_1, beta, mu)
    g0_1i = G0_atom(tau_1 - tau_in, beta, mu)
    n_0 = G0_atom_at_zero_minus(beta, mu)
    
    # D_{tau_1}(x_out, x_in) = (-1)^1 * det(M_up) * det(M_dn)
    # M_up = [[G_0(tau_out-tau_in), G_0(tau_out-tau_1)], [G_0(tau_1-tau_in), G_0(0^-)]]
    # det M_up = G_0(out-in)*n_0 - G_0(out-1)*G_0(1-in)
    # M_dn = [[n_0]], det = n_0
    # D = -1 * [G_0(out-in)*n_0 - G_0(out-1)*G_0(1-in)] * n_0
    #   = -n_0^2 * G_0(out-in) + n_0 * G_0(out-1)*G_0(1-in)
    expected_D = -(n_0**2) * g0_oi + n_0 * g0_o1 * g0_1i
    computed_D = D_corr([tau_1], tau_out, tau_in, beta, mu)
    print(f"n=1 correlator: D_tau1   = {computed_D:.10f}")
    print(f"                expected = {expected_D:.10f}")
    print(f"                diff     = {abs(computed_D - expected_D):.2e}")
    
    # D_{tau_1}(empty) = -1 * n_0^2
    expected_Dv = -(n_0**2)
    computed_Dv = D_vac([tau_1], beta, mu)
    print(f"n=1 vacuum:     D_tau1(empty) = {computed_Dv:.10f}, expected = {expected_Dv:.10f}, diff = {abs(computed_Dv - expected_Dv):.2e}")
    print()
    
    # n=1 connected (the Rossi formula): C_tau1 = D_tau1 - G_0 * D_tau1(empty)
    C_n1 = computed_D - g0_oi * computed_Dv
    expected_C = n_0 * g0_o1 * g0_1i
    print(f"n=1 connected (manual recursion): C_tau1 = {C_n1:.10f}")
    print(f"                expected         = +n_0*G_0*G_0 = {expected_C:.10f}")
    print(f"                diff             = {abs(C_n1 - expected_C):.2e}")
