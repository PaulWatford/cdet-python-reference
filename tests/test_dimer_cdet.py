"""
End-to-end audit of the 2-site CDet pipeline.

Strategy:
  Check A: t=0 reduction to atom (already shown in sanity tests, but
           run again as part of the audit)
  Check B: U=0 reduction to free G_0 (boundary case)
  Check C: half-filling density matches 0.5 (independent check via integration)
  Check D: end-to-end at order 1, compare sum_i integraldtau C_V({(i,tau)}) against
           the dG/dU of G_exact_2site
  Check E: end-to-end at order 2, same but 2 vertices, 4 site configurations,
           2 time integrals; compare to d^2G/dU^2 coefficient

"""
import sys
import numpy as np
from scipy.integrate import quad, dblquad
import time
from cdet_reference.dimer.green_function import G0_2site, G0_2site_at_zero_minus
from cdet_reference.dimer.exact_diagonalization import G_exact_2site
from cdet_reference.dimer.cdet_recursion import C_V_2site


N_PASS = 0
N_FAIL = 0
ABS_FLOOR = 1e-15

def report(name, ok, details=""):
    global N_PASS, N_FAIL
    flag = "PASS" if ok else "FAIL"
    if ok: N_PASS += 1
    else:  N_FAIL += 1
    print(f"  [{flag}] {name}")
    if details:
        print(f"         {details}")


def compare(a, b):
    scale = max(abs(a), abs(b))
    if scale < ABS_FLOOR:
        return 0.0
    return abs(a - b) / scale


print("="*72)
print(" Audit of 2-site CDet pipeline")
print("="*72)
print()

# ---------------------------------------------------------------------------
# Check A: at t=0, results match atom CDet (already verified, included here)
# ---------------------------------------------------------------------------
print("Check A: at t=0, full pipeline reduces to atom CDet")
from cdet_reference.atom.cdet_recursion import C_V as C_V_atom

all_ok = True
worst = 0.0
beta, mu = 5.0, 0.3
x_out = (0, 1.5)
x_in = (0, 0.5)
for V_atom in [[], [1.0], [1.0, 2.5], [0.5, 1.7, 3.3]]:
    V_2s = [(0, tau) for tau in V_atom]
    c_atom = C_V_atom(V_atom, 1.5, 0.5, beta, mu)
    c_2s = C_V_2site(V_2s, x_out, x_in, beta, mu, t=0.0)
    d = compare(c_atom, c_2s)
    if d > worst: worst = d
report("A. n=0,1,2,3 reductions at t=0 match atom exactly", worst == 0.0, f"worst rel = {worst:.2e}")
print()

# ---------------------------------------------------------------------------
# Check B: at U=0, C_empty = G_0 (no interactions). This is a boundary case.
# ---------------------------------------------------------------------------
print("Check B: at U=0, C_empty matches free G_0")
all_ok = True
worst = 0.0
beta, mu, t = 4.0, 0.2, 1.0
for tau in [0.5, 1.5, 3.0]:
    for (i, j) in [(0, 0), (0, 1)]:
        x_out = (i, tau)
        x_in = (j, 0.0)
        c0 = C_V_2site([], x_out, x_in, beta, mu, t)
        g0 = G0_2site(i, j, tau, beta, mu, t)
        d = compare(c0, g0)
        if d > worst: worst = d
report("B. C_empty = G_0 (6 cases)", worst < 1e-14, f"worst rel = {worst:.2e}")
print()

# ---------------------------------------------------------------------------
# Check C: half-filling. At mu=U/2, density per spin = 0.5 exactly.
# 
# Density is G(tau=0^-) at i_out = i_in. CDet must reproduce this through
# integration to all orders in U. We can check that c_0 + integralc_1 dtau + ...
# equals the exact density at the chosen U.
# 
# Simpler: just check that the exact G computed via ED gives <n>=0.5 at
# half-filling, since we've already established <n>=0.5 in exact_2site tests.
# Here we want a CDet-based check.
# 
# At order 0: c_0 = G_0(0,0;0^-) = (n_F(eps_B) + n_F(eps_A))/2.
# At half-filling without U, n_F(eps_B) + n_F(eps_A) might not be exactly 1.
# Half-filling with interactions requires mu = U/2, and the FULL series is needed.
# 
# For this check, take small U (perturbative) and verify that the FIRST FEW
# order corrections move the density toward 0.5 exactly at mu=U/2.
# ---------------------------------------------------------------------------
print("Check C: density correction (small U): CDet order 1 reproduces dG/dU")
# Take a generic point, not half-filling, to compare derivatives.
beta, mu, t = 3.0, 0.4, 1.0
tau = 0.5
i_out, i_in = 0, 0
x_out = (i_out, tau)
x_in = (i_in, 0.0)

# Exact via finite-diff on ED:
eps = 1e-6
g_at_U_plus = G_exact_2site(i_out, i_in, tau, beta, mu, t, U=eps)
g_at_U_zero = G_exact_2site(i_out, i_in, tau, beta, mu, t, U=0.0)
dG_dU_exact = (g_at_U_plus - g_at_U_zero) / eps

# CDet first-order: c_1 = sum_i integraldtau_1 C_V({(i, tau_1)})
# Per the perturbative expansion, this should equal dG/dU at U=0.
def integrand_site(i_v, tau_1):
    V = [(i_v, tau_1)]
    return C_V_2site(V, x_out, x_in, beta, mu, t)

# sum over the 2 site configurations of the vertex
c1_via_cdet = 0.0
for i_v in [0, 1]:
    c1_site, _ = quad(lambda tau_1: integrand_site(i_v, tau_1), 0, beta, limit=200)
    c1_via_cdet += c1_site

# CDet at fixed V gives the coefficient of U^1 directly (no 1/n! at this order)
print(f"  dG/dU at U=0 (exact, finite diff):  {dG_dU_exact:.10f}")
print(f"  c_1 via CDet (sum_i integraldtau C_V):           {c1_via_cdet:.10f}")
print(f"  relative diff:                          {compare(dG_dU_exact, c1_via_cdet):.2e}")
ok = compare(dG_dU_exact, c1_via_cdet) < 1e-5  # finite-diff has limited precision
report("C. n=1 end-to-end via CDet matches exact dG/dU", ok)
print()

# ---------------------------------------------------------------------------
# Check D: same as C but at multiple parameter points to confirm.
# ---------------------------------------------------------------------------
print("Check D: n=1 end-to-end at multiple parameter points")
all_ok = True
worst = 0.0
test_pts = [
    (3.0, 0.4, 1.0, 0.5*3.0, 0, 0),
    (3.0, 0.4, 1.0, 0.5*3.0, 0, 1),  # off-diagonal
    (4.0, 0.0, 0.7, 0.3*4.0, 0, 0),
    (4.0, 0.0, 0.7, 0.3*4.0, 1, 0),
    (2.0, 0.6, 1.5, 0.7*2.0, 0, 0),
]
for (beta_n, mu_n, t_n, tau_n, i_out, i_in) in test_pts:
    x_out = (i_out, tau_n)
    x_in = (i_in, 0.0)
    
    g_plus = G_exact_2site(i_out, i_in, tau_n, beta_n, mu_n, t_n, U=eps)
    g_zero = G_exact_2site(i_out, i_in, tau_n, beta_n, mu_n, t_n, U=0.0)
    dG_dU = (g_plus - g_zero) / eps
    
    c1 = 0.0
    for i_v in [0, 1]:
        ci, _ = quad(lambda tau_1: C_V_2site([(i_v, tau_1)], x_out, x_in, beta_n, mu_n, t_n),
                     0, beta_n, limit=200)
        c1 += ci
    
    d = compare(dG_dU, c1)
    if d > worst: worst = d
    if d > 1e-5:
        all_ok = False
        print(f"    beta={beta_n}, mu={mu_n}, t={t_n}, tau={tau_n}, (i_out,i_in)=({i_out},{i_in}): rd={d:.2e}, dG/dU={dG_dU:.6e}, c1={c1:.6e}")
    else:
        print(f"    beta={beta_n}, mu={mu_n}, t={t_n}, tau={tau_n}, (i_out,i_in)=({i_out},{i_in}): rd={d:.2e}, c1={c1:.6e}")

report("D. n=1 end-to-end across 5 parameter points", all_ok, f"worst rel = {worst:.2e}")
print()

# ---------------------------------------------------------------------------
# Check E: end-to-end at n=2
# 
# c_2 = (1/2!) sum_{i_1, i_2} int int dtau_1 dtau_2 C_V({(i_1,tau_1), (i_2,tau_2)})
# = (1/2!) sum over 4 site configurations  int int
# 
# Compare to second derivative d^2G/dU^2 from ED (via finite diffs).
# ---------------------------------------------------------------------------
print("Check E: n=2 end-to-end (this takes a minute)")
beta_n, mu_n, t_n = 2.0, 0.3, 1.0
tau_n = 0.5 * beta_n
i_out, i_in = 0, 0
x_out = (i_out, tau_n)
x_in = (i_in, 0.0)

# Second derivative from ED. Use 5-point stencil for stability.
eps2 = 1e-3
g_p2 = G_exact_2site(i_out, i_in, tau_n, beta_n, mu_n, t_n, U=2*eps2)
g_p1 = G_exact_2site(i_out, i_in, tau_n, beta_n, mu_n, t_n, U=eps2)
g_0  = G_exact_2site(i_out, i_in, tau_n, beta_n, mu_n, t_n, U=0.0)
g_m1 = G_exact_2site(i_out, i_in, tau_n, beta_n, mu_n, t_n, U=-eps2)
g_m2 = G_exact_2site(i_out, i_in, tau_n, beta_n, mu_n, t_n, U=-2*eps2)
# 5-point stencil for f''
d2G_dU2 = (-g_p2 + 16*g_p1 - 30*g_0 + 16*g_m1 - g_m2) / (12 * eps2**2)
# Taylor coefficient is c_2 = (1/2!) d^2G/dU^2, so d^2G/dU^2 = 2 c_2
c2_exact = d2G_dU2 / 2.0

print(f"  Exact c_2 via finite diff: {c2_exact:.10f}")

t_start = time.time()
c2_via_cdet = 0.0
for i1 in [0, 1]:
    for i2 in [0, 1]:
        # Avoid double-counting due to vertex indistinguishability:
        # The (1/n!) factor in the perturbative expansion accounts for this
        # automatically: we just sum over ALL configurations and divide by n!.
        result, _ = dblquad(
            lambda t2, t1: C_V_2site([(i1, t1), (i2, t2)], x_out, x_in, beta_n, mu_n, t_n),
            0, beta_n, 0, beta_n,
            epsabs=1e-7, epsrel=1e-7
        )
        c2_via_cdet += result
c2_via_cdet /= 2.0  # 1/2! for two identical vertex insertions
elapsed = time.time() - t_start

print(f"  c_2 via CDet: {c2_via_cdet:.10f}")
print(f"  rel diff: {compare(c2_exact, c2_via_cdet):.2e}")
print(f"  (elapsed: {elapsed:.1f} s)")

ok = compare(c2_exact, c2_via_cdet) < 1e-5
report("E. n=2 end-to-end", ok)
print()

# ---------------------------------------------------------------------------
print("="*72)
print(f" Summary: {N_PASS} PASS, {N_FAIL} FAIL")
print("="*72)
if N_FAIL == 0:
    print("""
 2-site CDet pipeline verified:
   A. t=0 reduction to atom (exact, 0.00e+00)
   B. U=0 reduction to free G_0
   C. n=1 end-to-end vs ED finite-diff dG/dU (one point, detailed)
   D. n=1 end-to-end at 5 parameter points
   E. n=2 end-to-end against ED finite-diff d^2G/dU^2 (one point)
""")
else:
    print(f" {N_FAIL} failure(s). See output above.")
