"""
End-to-end audit of CDet on the 4-site pi-flux Hubbard model.

Verification ladder:
  Check A: n=0, C_empty = G_0_free (direct match)
  Check B: n=1 dG/dU|_{U=0} via finite-diff on ED matches CDet integration
  Check C: n=2 d^2G/dU^2|_{U=0} via 5-point stencil matches CDet dblquad

Reference: ED via matrix exponential at the relevant U values, take finite
differences to extract Taylor coefficients in U. The float64 finite-difference
ceiling is ~1e-6, which is sufficient for the n=0..2 verification here.
For higher orders (n >= 3) we'd switch to mpmath at high precision.

"""
import sys
import time
import numpy as np

from cdet_reference.piflux_square.green_function import G0_4site_piflux
from cdet_reference.piflux_square.exact_diagonalization import G_exact_4site_piflux, build_4site_operators
from cdet_reference.piflux_square.cdet_recursion import C_V_4site_piflux
from cdet_reference.quadrature.quadrature_nested import integrate_1d_with_kinks, integrate_simplex_3d


N_PASS = 0
N_FAIL = 0
ABS_FLOOR = 1e-15


def report(name, ok, details=""):
    global N_PASS, N_FAIL
    flag = "PASS" if ok else "FAIL"
    if ok:
        N_PASS += 1
    else:
        N_FAIL += 1
    print(f"  [{flag}] {name}")
    if details:
        print(f"         {details}")


def compare(a, b):
    scale = max(abs(a), abs(b))
    if scale < ABS_FLOOR:
        return 0.0
    return abs(a - b) / scale


# ============================================================================
# Setup: parameters
# ============================================================================
beta = 2.0
mu = 0.3
t = 1.0
tau = 1.0
i_out, i_in = 0, 0   # both externals at site 0
x_out = (i_out, tau)
x_in = (i_in, 0.0)
fixed_kinks = [tau]

# For the audit, we also need an "internal" vertex to integrate over.
# In CDet, the c_n coefficient is computed by integrating C_V over tau_1, ..., tau_n
# in (0, beta)^n and summing over all 4^n site configurations, then dividing by n!.
# Or equivalently via simplex integration over tau_1 < tau_2 < ... < tau_n (no factorial).

print("=" * 72)
print("  4-site pi-flux Hubbard CDet audit (n=0, 1, 2)")
print("=" * 72)
print()
print(f"Parameters: beta={beta}, mu={mu}, t={t}, tau={tau}, (i_out, i_in)=({i_out}, {i_in})")
print()

# Pre-build ED operators for speed
print("Building 256-dim operators...")
ops = build_4site_operators()
print()


# ============================================================================
# Check A: n=0: C_empty(x_out, x_in) = G_0(i_out, i_in; tau_out - tau_in)
# ============================================================================
print("Check A: n=0, C_empty = G_0 (free Green's function)")

c_0_cdet = C_V_4site_piflux([], x_out, x_in, beta, mu, t)
g_0_free = G0_4site_piflux(i_out, i_in, tau - 0.0, beta, mu, t)
diff = compare(c_0_cdet, g_0_free)
report(f"A. C_empty = {c_0_cdet:.12f}, G_0 = {g_0_free:.12f}",
       diff < 1e-12, f"rel = {diff:.2e}")
print()


# ============================================================================
# Check B: n=1 dG/dU via 3-point stencil vs CDet integration
# ============================================================================
print("Check B: n=1, dG/dU|_{U=0} via finite diff vs CDet integration")
print("  Reference: 3-point finite-diff on ED G_exact at U = +/-h")
print()

h = 1e-4
g_plus = G_exact_4site_piflux(i_out, i_in, tau, beta, mu, t, +h, ops=ops)
g_minus = G_exact_4site_piflux(i_out, i_in, tau, beta, mu, t, -h, ops=ops)
dG_dU_fd = (g_plus - g_minus) / (2 * h)
print(f"  ED finite-diff dG/dU|_{{U=0}} ~= {dG_dU_fd:.10f}")

# CDet: c_1 = - integral_0^beta dtau_1 sum_{i_1=0}^3 C_{V=[(i_1, tau_1)]}(x_out, x_in)
# The sign convention: in our setup, the n-th Taylor coefficient of G in U is
#   c_n = (1/n!) integral_{[0,beta]^n} dtau_1 ... dtau_n sum_{sites} C_V(x_out, x_in)
# where the sum over sites is over all 4^n site configurations.
# 
# (No extra minus sign: the (-U)^n factor in perturbation theory has been
# absorbed into D_V's definition.)

t0 = time.time()
c_1_cdet = 0.0
for site_1 in range(4):
    integrand = lambda tau_1, site_1=site_1: C_V_4site_piflux(
        [(site_1, tau_1)], x_out, x_in, beta, mu, t
    )
    contrib = integrate_1d_with_kinks(integrand, 10, beta, fixed_kinks)
    print(f"    site_1={site_1}: contrib = {contrib:+.10f}")
    c_1_cdet += contrib
elapsed_b = time.time() - t0
print(f"  CDet n=1 (sum over 4 sites, simplex = whole interval): {c_1_cdet:.10f}")
print(f"  ({elapsed_b:.1f}s)")

diff = compare(c_1_cdet, dG_dU_fd)
report(f"B. c_1(CDet) = {c_1_cdet:.10f}, c_1(ED-FD) = {dG_dU_fd:.10f}",
       diff < 1e-4, f"rel = {diff:.2e} (limited by 3-pt finite-diff precision)")
print()


# ============================================================================
# Check C: n=2 d^2G/dU^2 via 5-point stencil vs CDet integration
# ============================================================================
print("Check C: n=2, d^2G/dU^2|_{U=0} via 5-point stencil vs CDet")
print()

h = 1e-3
g_2p = G_exact_4site_piflux(i_out, i_in, tau, beta, mu, t, +2*h, ops=ops)
g_p  = G_exact_4site_piflux(i_out, i_in, tau, beta, mu, t, +h, ops=ops)
g_0  = G_exact_4site_piflux(i_out, i_in, tau, beta, mu, t, 0.0, ops=ops)
g_m  = G_exact_4site_piflux(i_out, i_in, tau, beta, mu, t, -h, ops=ops)
g_2m = G_exact_4site_piflux(i_out, i_in, tau, beta, mu, t, -2*h, ops=ops)

# 5-point stencil for f''(0):  [-f(2h) + 16f(h) - 30f(0) + 16f(-h) - f(-2h)] / (12 h^2)
d2G_dU2_fd = (-g_2p + 16*g_p - 30*g_0 + 16*g_m - g_2m) / (12 * h**2)
c_2_ed = d2G_dU2_fd / 2.0  # c_2 = (1/2!) d^2G/dU^2
print(f"  ED finite-diff d^2G/dU^2|_{{U=0}} ~= {d2G_dU2_fd:.8f}")
print(f"  ED finite-diff c_2 = d^2G/dU^2/2 ~= {c_2_ed:.8f}")
print()

# CDet: c_2 via simplex integration summing over 16 site configs
print("  CDet n=2 via simplex integration (16 site configs)...")
from cdet_reference.quadrature.quadrature_nested import integrate_simplex_3d  # Already imported

# 2D simplex integrator
def integrate_simplex_2d(f, n_per_piece, beta, fixed_kinks):
    """int int f(tau_1, tau_2) over 0 <= tau_1 <= tau_2 <= beta."""
    def inner_t1(t2):
        ak = [k for k in fixed_kinks if 0 < k < t2]
        if t2 < 1e-15:
            return 0.0
        return integrate_1d_with_kinks(lambda t1: f(t1, t2), n_per_piece, t2, ak)
    return integrate_1d_with_kinks(inner_t1, n_per_piece, beta, fixed_kinks)


t0 = time.time()
c_2_cdet = 0.0
for site_1 in range(4):
    for site_2 in range(4):
        integrand = (
            lambda tau_1, tau_2, s1=site_1, s2=site_2:
            C_V_4site_piflux([(s1, tau_1), (s2, tau_2)], x_out, x_in, beta, mu, t)
        )
        contrib = integrate_simplex_2d(integrand, 8, beta, fixed_kinks)
        c_2_cdet += contrib

elapsed_c = time.time() - t0
print(f"  CDet n=2 (sum 16 site configs, simplex): c_2 = {c_2_cdet:.10f}")
print(f"  ({elapsed_c:.1f}s)")

diff = compare(c_2_cdet, c_2_ed)
report(f"C. c_2(CDet) = {c_2_cdet:.10f}, c_2(ED-FD) = {c_2_ed:.10f}",
       diff < 1e-4, f"rel = {diff:.2e} (limited by 5-pt finite-diff precision)")
print()


# ============================================================================
# Summary
# ============================================================================
print("=" * 72)
print(f"  Summary: {N_PASS} PASS, {N_FAIL} FAIL")
print("=" * 72)
