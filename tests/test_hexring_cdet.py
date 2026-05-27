"""
End-to-end audit of CDet on the 6-site hexring Hubbard model at orders
n = 0, 1, 2.

Reference: 4096-dim ED via matrix exponential. Taylor coefficients via
finite difference on G_exact(U). Same approach as the pi-flux audit.

Expected runtime: dominated by 6-7 ED eigh's at ~15s each = ~1-2 min for n=2.
"""
import sys
import time
import numpy as np

from cdet_reference.hexring.green_function import G0_hexring
from cdet_reference.hexring.exact_diagonalization import G_exact_hexring, build_hexring_operators, build_H_hexring
from cdet_reference.hexring.cdet_recursion import C_V_hexring
from cdet_reference.quadrature.quadrature_nested import integrate_1d_with_kinks


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


def integrate_simplex_2d(f, n_per_piece, beta, fixed_kinks):
    """int int f(tau_1, tau_2) over 0 <= tau_1 <= tau_2 <= beta."""
    def inner_t1(t2):
        ak = [k for k in fixed_kinks if 0 < k < t2]
        if t2 < 1e-15:
            return 0.0
        return integrate_1d_with_kinks(lambda t1: f(t1, t2), n_per_piece, t2, ak)
    return integrate_1d_with_kinks(inner_t1, n_per_piece, beta, fixed_kinks)


# =============================================================================
# Setup
# =============================================================================
beta = 2.0
mu = 0.3
t = 1.0
tau = 1.0
i_out, i_in = 0, 0
x_out = (i_out, tau)
x_in = (i_in, 0.0)
fixed_kinks = [tau]


print("=" * 72)
print("  6-site hexring Hubbard CDet audit (n=0, 1, 2)")
print("=" * 72)
print()
print(f"Parameters: beta={beta}, mu={mu}, t={t}, tau={tau}, externals at site {i_out}")
print()

# Pre-build operators
t0 = time.time()
ops = build_hexring_operators()
print(f"Built operators in {time.time()-t0:.1f}s")
print()


# =============================================================================
# Check A: n=0: C_empty = G_0(i_out, i_in; tau_out - tau_in)
# =============================================================================
print("Check A: n=0, C_empty = G_0 (free Green's function)")

c_0_cdet = C_V_hexring([], x_out, x_in, beta, mu, t)
g_0_free = G0_hexring(i_out, i_in, tau - 0.0, beta, mu, t)
diff = compare(c_0_cdet, g_0_free)
report(f"A. C_empty = {c_0_cdet:.12f}, G_0 = {g_0_free:.12f}",
       diff < 1e-12, f"rel = {diff:.2e}")
print()


# =============================================================================
# Check B: n=1 dG/dU via 3-point stencil vs CDet integration
# =============================================================================
print("Check B: n=1, dG/dU|_{U=0} via finite diff vs CDet integration")
print("  Reference: 3-pt FD on ED G_exact at U = +/-h (need 2 ED eighs)")
print()

h = 1e-4
t_ref0 = time.time()

# Each call to G_exact_hexring does an eigh. Cache by U value.
# Need U = +h and U = -h. Pre-eigh both.
H_plus = build_H_hexring(beta, mu, t, +h, ops=ops).toarray()
H_minus = build_H_hexring(beta, mu, t, -h, ops=ops).toarray()
eigh_plus = np.linalg.eigh(H_plus)
eigh_minus = np.linalg.eigh(H_minus)
print(f"  ED eigh for U=+/-h done ({time.time()-t_ref0:.1f}s)")

g_plus = G_exact_hexring(i_out, i_in, tau, beta, mu, t, +h, ops=ops, eigh_cache=eigh_plus)
g_minus = G_exact_hexring(i_out, i_in, tau, beta, mu, t, -h, ops=ops, eigh_cache=eigh_minus)
dG_dU_fd = (g_plus - g_minus) / (2 * h)
print(f"  ED finite-diff dG/dU|_{{U=0}} ~= {dG_dU_fd:.10f}")
print()

t_cdet0 = time.time()
c_1_cdet = 0.0
site_contribs_n1 = {}
for site_1 in range(6):
    integrand = lambda tau_1, site_1=site_1: C_V_hexring(
        [(site_1, tau_1)], x_out, x_in, beta, mu, t
    )
    contrib = integrate_1d_with_kinks(integrand, 10, beta, fixed_kinks)
    site_contribs_n1[site_1] = contrib
    c_1_cdet += contrib
    print(f"    site_1={site_1}: contrib = {contrib:+.10e}")
elapsed_b = time.time() - t_cdet0
print(f"  CDet n=1 (sum over 6 sites): {c_1_cdet:.10f}")
print(f"  ({elapsed_b:.1f}s)")

diff = compare(c_1_cdet, dG_dU_fd)
report(f"B. c_1(CDet) = {c_1_cdet:.10f}, c_1(ED-FD) = {dG_dU_fd:.10f}",
       diff < 1e-4, f"rel = {diff:.2e}")
print()


# =============================================================================
# Check C: n=2 d^2G/dU^2 via 5-point stencil vs CDet integration
# =============================================================================
print("Check C: n=2, d^2G/dU^2|_{U=0} via 5-pt stencil vs CDet")
print("  Reference: 5-pt stencil needs 5 ED eighs at U in {+/-2h, +/-h, 0}")
print()

h = 1e-3
t_ref0 = time.time()

# Pre-eigh
def cache_for_U(U):
    H = build_H_hexring(beta, mu, t, U, ops=ops).toarray()
    return np.linalg.eigh(H)

U_values = [+2*h, +h, 0.0, -h, -2*h]
eigh_caches = {U: cache_for_U(U) for U in U_values}
print(f"  ED eighs (x5) done ({time.time()-t_ref0:.1f}s)")

def g_at(U):
    return G_exact_hexring(i_out, i_in, tau, beta, mu, t, U, ops=ops, eigh_cache=eigh_caches[U])

g_2p = g_at(+2*h)
g_p  = g_at(+h)
g_0  = g_at(0.0)
g_m  = g_at(-h)
g_2m = g_at(-2*h)

d2G_dU2_fd = (-g_2p + 16*g_p - 30*g_0 + 16*g_m - g_2m) / (12 * h**2)
c_2_ed = d2G_dU2_fd / 2.0
print(f"  ED finite-diff c_2 ~= {c_2_ed:.8f}")
print()

t_cdet0 = time.time()
c_2_cdet = 0.0
print("  CDet n=2 via simplex integration (36 site configs)...")
site_contribs_n2 = {}
for site_1 in range(6):
    for site_2 in range(6):
        integrand = (
            lambda tau_1, tau_2, s1=site_1, s2=site_2:
            C_V_hexring([(s1, tau_1), (s2, tau_2)], x_out, x_in, beta, mu, t)
        )
        contrib = integrate_simplex_2d(integrand, 8, beta, fixed_kinks)
        c_2_cdet += contrib
        # Group by unordered multiset
        key = tuple(sorted([site_1, site_2]))
        site_contribs_n2[key] = site_contribs_n2.get(key, 0.0) + contrib

elapsed_c = time.time() - t_cdet0
print(f"  CDet n=2 (sum 36 site configs, simplex): c_2 = {c_2_cdet:.10f}")
print(f"  ({elapsed_c:.1f}s)")

diff = compare(c_2_cdet, c_2_ed)
report(f"C. c_2(CDet) = {c_2_cdet:.10f}, c_2(ED-FD) = {c_2_ed:.10f}",
       diff < 1e-4, f"rel = {diff:.2e}")
print()


# =============================================================================
# Structural analysis at n=2
# =============================================================================
print("Structural decomposition at n=2 by unordered site multiset:")
print(f"{'multiset':>15} | {'sum of contributions':>22}")
print('-' * 45)

groups_n2 = {}
all_pairs = set()
for s1 in range(6):
    for s2 in range(6):
        all_pairs.add(tuple(sorted([s1, s2])))

for key in sorted(all_pairs):
    val = site_contribs_n2.get(key, 0.0)
    flag = ""
    if abs(val) < 1e-12:
        flag = "  ZERO"
    print(f"  {key!s:>13} | {val:+.10e}{flag}")

print()


# =============================================================================
# Summary
# =============================================================================
print("=" * 72)
print(f"  Summary: {N_PASS} PASS, {N_FAIL} FAIL")
print("=" * 72)
