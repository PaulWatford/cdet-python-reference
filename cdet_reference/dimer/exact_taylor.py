"""
High-precision Taylor coefficients of G_exact_2site(U).

The standard finite-difference approach using float64 loses precision rapidly:
  - d^2G/dU^2 accurate to ~1e-7
  - d^3G/dU^3 accurate to ~1e-5
  - d^4G/dU^4 accurate to ~1e-3

For verifying CDet at n=3 and n=4, we need better. We use mpmath to compute
G_exact_2site at multiple U values with ~50-digit precision, then apply
high-order finite-difference stencils.

This is computationally expensive (matrix exp on 16x16 at high precision)
but produces clean reference values for CDet end-to-end tests.

"""
import mpmath
import numpy as np


def _single_site_c_up_mp():
    """4x4 c_up matrix as mpmath matrix."""
    M = mpmath.zeros(4, 4)
    M[0, 1] = 1
    M[2, 3] = -1
    return M


def _single_site_c_dn_mp():
    M = mpmath.zeros(4, 4)
    M[0, 2] = 1
    M[1, 3] = 1
    return M


def _single_site_n_up_mp():
    return mpmath.diag([0, 1, 0, 1])


def _single_site_n_dn_mp():
    return mpmath.diag([0, 0, 1, 1])


def _parity_mp():
    return mpmath.diag([1, -1, -1, 1])


def _identity4_mp():
    return mpmath.eye(4)


def _kron_mp(A, B):
    """Kronecker product of two mpmath matrices."""
    rA, cA = A.rows, A.cols
    rB, cB = B.rows, B.cols
    out = mpmath.zeros(rA * rB, cA * cB)
    for i in range(rA):
        for j in range(cA):
            for k in range(rB):
                for l in range(cB):
                    out[i*rB + k, j*cB + l] = A[i, j] * B[k, l]
    return out


def _build_2site_ops_mp():
    """Build the 16x16 mpmath operators (same as exact_2site.py but in mpmath)."""
    I4 = _identity4_mp()
    cu = _single_site_c_up_mp()
    cd = _single_site_c_dn_mp()
    nu = _single_site_n_up_mp()
    nd = _single_site_n_dn_mp()
    P = _parity_mp()
    
    ops = {}
    ops['c_up_0'] = _kron_mp(cu, I4)
    ops['c_dn_0'] = _kron_mp(cd, I4)
    ops['n_up_0'] = _kron_mp(nu, I4)
    ops['n_dn_0'] = _kron_mp(nd, I4)
    ops['c_up_1'] = _kron_mp(P, cu)
    ops['c_dn_1'] = _kron_mp(P, cd)
    ops['n_up_1'] = _kron_mp(I4, nu)
    ops['n_dn_1'] = _kron_mp(I4, nd)
    
    # Daggers: just transpose (real matrices)
    for k in list(ops.keys()):
        if k.startswith('c_'):
            ops[f'cdag_{k[2:]}'] = ops[k].T
    
    return ops


def _build_H_mp(mu, t, U, ops=None):
    """Build the 2-site Hubbard H as 16x16 mpmath matrix."""
    if ops is None:
        ops = _build_2site_ops_mp()
    
    mu_mp = mpmath.mpf(mu)
    t_mp = mpmath.mpf(t)
    U_mp = mpmath.mpf(U)
    
    H_hop = -t_mp * (
        ops['cdag_up_0'] * ops['c_up_1'] + ops['cdag_up_1'] * ops['c_up_0']
      + ops['cdag_dn_0'] * ops['c_dn_1'] + ops['cdag_dn_1'] * ops['c_dn_0']
    )
    H_U = U_mp * (ops['n_up_0'] * ops['n_dn_0'] + ops['n_up_1'] * ops['n_dn_1'])
    H_mu = -mu_mp * (ops['n_up_0'] + ops['n_dn_0'] + ops['n_up_1'] + ops['n_dn_1'])
    
    return H_hop + H_U + H_mu


def G_exact_2site_mp(i_out, i_in, tau, beta, mu, t, U, dps=50):
    """High-precision G_exact_2site using mpmath.
    
    Args:
      i_out, i_in: site indices for out/in
      tau, beta, mu, t, U: parameters
      dps: decimal places of precision (default 50)
    
    Returns:
      mpmath.mpf, the Green's function value with ~dps digits.
    
    Why dps=50: we take fourth-order finite differences to extract c_n
    Taylor coefficients up to n=4. Each derivative roughly halves the
    available precision; starting at 50 digits leaves ~30 digits in c_4,
    safely above the ~16-digit float64 floor of any downstream consumer.
    """
    mpmath.mp.dps = dps
    
    tau_mp = mpmath.mpf(tau)
    beta_mp = mpmath.mpf(beta)
    
    # Fold tau into (0, beta) with antiperiodic sign
    sign = mpmath.mpf(1)
    while tau_mp >= beta_mp:
        tau_mp -= beta_mp
        sign = -sign
    while tau_mp < 0:
        tau_mp += beta_mp
        sign = -sign
    
    ops = _build_2site_ops_mp()
    H = _build_H_mp(mu, t, U, ops)
    
    Z_mat = mpmath.expm(-beta_mp * H)
    # Compute trace
    Z_val = mpmath.mpf(0)
    for k in range(16):
        Z_val += Z_mat[k, k]
    
    c = ops[f'c_up_{i_out}']
    cdag = ops[f'cdag_up_{i_in}']
    
    e_late = mpmath.expm(-(beta_mp - tau_mp) * H)
    e_early = mpmath.expm(-tau_mp * H)
    
    inner = e_late * c * e_early * cdag
    tr = mpmath.mpf(0)
    for k in range(16):
        tr += inner[k, k]
    
    return sign * (-tr / Z_val)


# ---------------------------------------------------------------------------
# High-precision Taylor coefficients via central differences
# ---------------------------------------------------------------------------

# Central-difference stencils:
# For derivatives of order n, we use 2n+1 or more evaluation points.
# The error scales as O(h^p) for a (2k+1)-point stencil with appropriate weights.
# At high precision (dps=50), we can afford VERY small h (like 1e-6) and still
# have the truncation error dominate.

# 5-point stencil for 2nd derivative: f''(0) = (-f(2h) + 16f(h) - 30f(0) + 16f(-h) - f(-2h)) / (12 h^2)
# 7-point stencil for 3rd derivative: standard
# 9-point stencil for 4th derivative: standard

def taylor_coeff(func, n, h='1e-8', dps=50):
    """Compute the n-th Taylor coefficient c_n = f^(n)(0) / n! using high-precision finite differences.
    
    Uses central-difference stencils with 2n+3 points and step h.
    """
    mpmath.mp.dps = dps
    h = mpmath.mpf(h)
    
    if n == 0:
        return func(0)
    
    if n == 1:
        # 5-point stencil for f'(0): (-f(2h) + 8f(h) - 8f(-h) + f(-2h)) / (12h)
        return (-func(2*h) + 8*func(h) - 8*func(-h) + func(-2*h)) / (12 * h)
    
    if n == 2:
        # 5-point stencil for f''(0): (-f(2h) + 16f(h) - 30f(0) + 16f(-h) - f(-2h)) / (12 h^2), then /2
        f_2h = func(2*h)
        f_h = func(h)
        f_0 = func(0)
        f_mh = func(-h)
        f_m2h = func(-2*h)
        f_pp = (-f_2h + 16*f_h - 30*f_0 + 16*f_mh - f_m2h) / (12 * h * h)
        return f_pp / 2  # c_2 = f''(0) / 2!
    
    if n == 3:
        # 7-point central-difference stencil for f'''(0):
        # f'''(0) ~= (-f(3h) + 8f(2h) - 13f(h) + 13f(-h) - 8f(-2h) + f(-3h)) / (8 h^3)
        # (Verified: gives +6 for f(x)=x^3 as expected.)
        f_3h = func(3*h)
        f_2h = func(2*h)
        f_h = func(h)
        f_mh = func(-h)
        f_m2h = func(-2*h)
        f_m3h = func(-3*h)
        f_ppp = (-f_3h + 8*f_2h - 13*f_h + 13*f_mh - 8*f_m2h + f_m3h) / (8 * h * h * h)
        return f_ppp / 6  # c_3 = f'''(0) / 3!
    
    if n == 4:
        # 7-point stencil for f''''(0):
        # f''''(0) ~= (-f(3h) + 12f(2h) - 39f(h) + 56f(0) - 39f(-h) + 12f(-2h) - f(-3h)) / (6 h^4)
        f_3h = func(3*h)
        f_2h = func(2*h)
        f_h = func(h)
        f_0 = func(0)
        f_mh = func(-h)
        f_m2h = func(-2*h)
        f_m3h = func(-3*h)
        f_pppp = (-f_3h + 12*f_2h - 39*f_h + 56*f_0 - 39*f_mh + 12*f_m2h - f_m3h) / (6 * h**4)
        return f_pppp / 24  # c_4 = f''''(0) / 4!
    
    raise NotImplementedError(f"Stencil for n={n} not implemented")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    print("=== exact_2site_taylor.py sanity checks ===\n")
    
    # Test: Taylor coefficients should match exact_2site (float64) at low orders
    beta = 2.0
    mu = 0.3
    t = 1.0
    tau = 1.0
    i_out, i_in = 0, 0
    
    # We need to verify the mpmath ED gives the same result as float64 ED
    print(f"Test 1: mpmath ED reproduces float64 ED")
    from cdet_reference.dimer.exact_diagonalization import G_exact_2site
    
    t_start = time.time()
    g_float = G_exact_2site(i_out, i_in, tau, beta, mu, t, U=0.5)
    g_mp = G_exact_2site_mp(i_out, i_in, tau, beta, mu, t, U=0.5, dps=30)
    elapsed = time.time() - t_start
    print(f"  G_exact_2site (float64): {g_float}")
    print(f"  G_exact_2site_mp:         {g_mp}")
    print(f"  diff: {abs(g_float - float(g_mp)):.2e}")
    print(f"  (mpmath took {elapsed:.1f} s for one evaluation)")
    print()
    
    # Test 2: extract c_0 (= G_0 at U=0) and verify against the free Green's function
    print(f"Test 2: c_0 = G(U=0) matches free Green's function")
    from cdet_reference.dimer.green_function import G0_2site
    
    func = lambda U: G_exact_2site_mp(i_out, i_in, tau, beta, mu, t, U=U, dps=30)
    
    t_start = time.time()
    c0 = taylor_coeff(func, n=0, dps=30)
    g0_free = G0_2site(i_out, i_in, tau, beta, mu, t)
    print(f"  c_0 via mpmath:   {c0}")
    print(f"  G_0 (free):       {g0_free}")
    print(f"  diff: {abs(float(c0) - g0_free):.2e}")
    print(f"  ({time.time()-t_start:.1f} s)")
    print()
    
    # Test 3: c_1 via mpmath should match the c_1 we computed in audit_cdet_2site
    print(f"Test 3: c_1 at (beta,mu,t,tau) = (2, 0.3, 1, 1)")
    t_start = time.time()
    c1 = taylor_coeff(func, n=1, h='1e-8', dps=30)
    print(f"  c_1 = {c1}")
    print(f"  ({time.time()-t_start:.1f} s, 4 mpmath evaluations)")
