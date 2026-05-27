"""
Full machine-precision audit of the Wick determinant on the Hubbard atom.

Strategy:
  - n=0, n=1: closed-form expected values from the derivation, verified
    against the function output.
  - n=2: build the 2-vertex matrices by hand in a second function with
    a different code structure and compare.
  - END-TO-END: integrate C_V over V at orders n=0, 1, 2 and compare
    against the symbolic Taylor coefficients of G_exact_atom in U.
    This is the strongest test because it goes through the CDet
    recursion (manually here) and lands on a number we already
    know to machine precision.

All checks use relative differences except where exact equality is
expected. If a test fails, diagnose before editing.
"""
import sys
import math
import numpy as np
import sympy as sp
from itertools import combinations
from scipy.integrate import quad, dblquad, tplquad
from cdet_reference.atom.green_function import G0_atom, G_exact_atom, n_F, G0_atom_at_zero_minus
from cdet_reference.atom.wick_determinant import D_corr, D_vac, _build_matrix


# Pass/fail tracking
N_PASS = 0
N_FAIL = 0

def report(name, ok, details=""):
    global N_PASS, N_FAIL
    flag = "PASS" if ok else "FAIL"
    if ok: N_PASS += 1
    else:  N_FAIL += 1
    print(f"  [{flag}] {name}")
    if details:
        print(f"         {details}")


def rel_diff(a, b):
    """Relative difference, with safe handling of small values.
    
    For values near zero (where both |a| and |b| are below ABS_FLOOR),
    return the absolute difference scaled by 1/ABS_FLOOR. This treats
    "two values that are both effectively zero to within machine epsilon"
    as equal rather than producing spurious 'relative differences' of ~1.
    """
    ABS_FLOOR = 1e-13  # values below this are "machine zero", use absolute
    if max(abs(a), abs(b)) < ABS_FLOOR:
        # Both essentially zero; use scaled absolute diff
        return abs(a - b) / ABS_FLOOR
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


print("="*72)
print(" FULL audit of wick_determinant.py")
print("="*72)
print()

# ===========================================================================
# Check A: n=0 cases (both correlator and vacuum) at multiple parameter sets
# ===========================================================================
print("Check A: n=0 boundary conditions")
all_ok = True
worst = 0.0

for beta in [0.5, 2.0, 8.0]:
    for mu in [-0.3, 0.0, 0.7]:
        for tau_out, tau_in in [(0.1, 0.0), (1.0, 0.5), (0.3, 0.2)]:
            # Adjust for the actual beta
            tau_out_b = tau_out * beta
            tau_in_b = tau_in * beta
            d_corr = D_corr([], tau_out_b, tau_in_b, beta, mu)
            g0 = G0_atom(tau_out_b - tau_in_b, beta, mu)
            d_vac = D_vac([], beta, mu)
            
            rd = rel_diff(d_corr, g0)
            if rd > worst: worst = rd
            ok_corr = rd < 1e-14
            ok_vac = abs(d_vac - 1.0) < 1e-15
            all_ok = all_ok and ok_corr and ok_vac

report("A. n=0 correlator and vacuum across 27 parameter sets",
       all_ok, f"worst relative = {worst:.2e}")
print()

# ===========================================================================
# Check B: n=1 explicit formula
# 
# D_{tau_1}(x_out, x_in) = -1 * [G_0(tau_out-tau_in)*n_0 - G_0(tau_out-tau_1)*G_0(tau_1-tau_in)] * n_0
#                     = -n_0^2 G_0(tau_out-tau_in) + n_0 G_0(tau_out-tau_1) G_0(tau_1-tau_in)
# D_{tau_1}(empty) = -n_0^2
# ===========================================================================
print("Check B: n=1 explicit formula across parameter sets")
all_ok = True
worst_corr = 0.0
worst_vac = 0.0
n_skipped = 0
n_tested = 0

for beta in [1.0, 3.0, 7.0]:
    for mu in [-0.4, 0.0, 0.5, 1.2]:
        for (tau_out, tau_1, tau_in) in [(0.7, 0.5, 0.2), (0.9, 0.1, 0.05), (0.3, 0.6, 0.1)]:
            t_out = tau_out * beta
            t_1 = tau_1 * beta
            t_in = tau_in * beta
            
            n_0 = G0_atom_at_zero_minus(beta, mu)
            g_oi = G0_atom(t_out - t_in, beta, mu)
            g_o1 = G0_atom(t_out - t_1, beta, mu)
            g_1i = G0_atom(t_1 - t_in, beta, mu)
            
            expected_corr = -n_0*n_0 * g_oi + n_0 * g_o1 * g_1i
            computed_corr = D_corr([t_1], t_out, t_in, beta, mu)
            
            expected_vac = -n_0 * n_0
            computed_vac = D_vac([t_1], beta, mu)
            
            # Skip near-zero cancellation cases
            scale_corr = max(abs(computed_corr), abs(expected_corr))
            if scale_corr < 1e-15:
                n_skipped += 1
                # still check vacuum, which is never near zero
                rd_v = rel_diff(computed_vac, expected_vac)
                if rd_v > worst_vac: worst_vac = rd_v
                continue
            
            n_tested += 1
            rd_c = rel_diff(computed_corr, expected_corr)
            if rd_c > worst_corr: worst_corr = rd_c
            
            rd_v = rel_diff(computed_vac, expected_vac)
            if rd_v > worst_vac: worst_vac = rd_v
            
            if rd_c > 1e-13 or rd_v > 1e-13:
                print(f"    beta={beta}, mu={mu}, tau=({t_out},{t_1},{t_in}): rd_c={rd_c:.2e}, rd_v={rd_v:.2e}")
                all_ok = False

report(f"B. n=1 correlator + vacuum ({n_tested} non-trivial, {n_skipped} machine-zero)",
       all_ok, f"worst rel: corr={worst_corr:.2e}, vac={worst_vac:.2e}")
print()

# ===========================================================================
# Check C: n=2: INDEPENDENT determinant computation by direct expansion.
# 
# For a 3x3 M_up and a 2x2 M_dn, write out the determinants explicitly
# in the Leibniz formula (sum over permutations).
# ===========================================================================
print("Check C: n=2, independent determinant expansion")

def det_3x3_explicit(M):
    """Explicit 3x3 determinant via Leibniz formula."""
    a,b,c = M[0,0], M[0,1], M[0,2]
    d,e,f = M[1,0], M[1,1], M[1,2]
    g,h,i = M[2,0], M[2,1], M[2,2]
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

def det_2x2_explicit(M):
    return M[0,0]*M[1,1] - M[0,1]*M[1,0]

def D_corr_n2_independent(tau_1, tau_2, tau_out, tau_in, beta, mu):
    """Compute D_V(x_out, x_in) for V = [tau_1, tau_2] using explicit 
    determinant formulas. (-1)^2 = +1, so no overall sign flip."""
    g00 = G0_atom_at_zero_minus(beta, mu)  # n_0
    
    # Helper to evaluate G_0 safely
    def gg(dt):
        if dt == 0.0:
            return g00
        return G0_atom(dt, beta, mu)
    
    # M_up 3x3:
    #   row 0 = tau_out, row 1 = tau_1, row 2 = tau_2
    #   col 0 = tau_in,  col 1 = tau_1, col 2 = tau_2
    M_up = np.array([
        [gg(tau_out - tau_in), gg(tau_out - tau_1), gg(tau_out - tau_2)],
        [gg(tau_1   - tau_in), gg(tau_1   - tau_1), gg(tau_1   - tau_2)],
        [gg(tau_2   - tau_in), gg(tau_2   - tau_1), gg(tau_2   - tau_2)],
    ])
    det_up = det_3x3_explicit(M_up)
    
    # M_dn 2x2:
    M_dn = np.array([
        [gg(tau_1 - tau_1), gg(tau_1 - tau_2)],
        [gg(tau_2 - tau_1), gg(tau_2 - tau_2)],
    ])
    det_dn = det_2x2_explicit(M_dn)
    
    sign = (-1)**2  # = +1
    return sign * det_up * det_dn


all_ok = True
worst = 0.0
n_skipped = 0
n_tested = 0
for beta in [1.5, 4.0]:
    for mu in [-0.2, 0.3, 1.0]:
        for V_frac in [(0.3, 0.7), (0.1, 0.9), (0.25, 0.6)]:
            for (out_f, in_f) in [(0.85, 0.05), (0.4, 0.2)]:
                t1 = V_frac[0] * beta
                t2 = V_frac[1] * beta
                tau_out = out_f * beta
                tau_in = in_f * beta
                
                d_fast = D_corr([t1, t2], tau_out, tau_in, beta, mu)
                d_slow = D_corr_n2_independent(t1, t2, tau_out, tau_in, beta, mu)
                
                # Skip cases where both values are at machine-zero scale.
                # These are physically zero from cancellation; ULP-level
                # disagreement between numpy.linalg.det and explicit Leibniz
                # is expected and meaningless.
                scale = max(abs(d_fast), abs(d_slow))
                if scale < 1e-15:
                    n_skipped += 1
                    continue
                
                n_tested += 1
                rd = rel_diff(d_fast, d_slow)
                if rd > worst: worst = rd
                if rd > 1e-12:
                    print(f"    beta={beta}, mu={mu}, V=({t1},{t2}): rd={rd:.2e}, fast={d_fast:.6e}, slow={d_slow:.6e}")
                    all_ok = False

report(f"C. n=2 D_corr: numpy.linalg.det vs Leibniz expansion ({n_tested} of 36 sets non-trivial)", 
       all_ok, f"worst rel = {worst:.2e}, skipped {n_skipped} machine-zero cases")
print()

# ===========================================================================
# Check D: vertex labeling invariance.
# 
# D_V should be invariant under permutation of V (since V is a SET, 
# the order shouldn't matter). 
# 
# A permutation of V columns/rows in M_up multiplies det_up by sign(perm),
# Permutation invariance argument. If we permute V = (tau_1, tau_2) -> (tau_2, tau_1):
#   - M_up rows 1,2 swap (rows for the vertices): sign(perm) = -1.
#   - M_up cols 1,2 swap: another sign(perm) = -1.
#   - Net for det_up: (-1)(-1) = +1.
#   - M_dn: rows swap AND cols swap: (-1)(-1) = +1.
# So det_up * det_dn is invariant under symmetric (row+col) permutations.
# ===========================================================================
print("Check D: D_V invariant under permutation of V")

all_ok = True
worst = 0.0
beta, mu = 4.0, 0.4

# Test n=2 and n=3
import itertools

for n in [2, 3, 4]:
    V_base = sorted([0.1*beta, 0.3*beta, 0.5*beta, 0.8*beta][:n])
    for perm in itertools.permutations(V_base):
        V_p = list(perm)
        d_corr = D_corr(V_p, 0.9*beta, 0.05*beta, beta, mu)
        d_vac = D_vac(V_p, beta, mu)
        if perm == tuple(V_base):
            d_corr_base = d_corr
            d_vac_base = d_vac
        else:
            rd_c = rel_diff(d_corr, d_corr_base)
            rd_v = rel_diff(d_vac, d_vac_base)
            if rd_c > worst: worst = rd_c
            if rd_v > worst: worst = rd_v
            if rd_c > 1e-12 or rd_v > 1e-12:
                print(f"    n={n}, perm={perm}: rd_corr={rd_c:.2e}, rd_vac={rd_v:.2e}")
                all_ok = False

report("D. D_V invariant under V permutation (n=2,3,4, all permutations)", 
       all_ok, f"worst rel = {worst:.2e}")
print()

# ===========================================================================
# Check E: End-to-end Taylor coefficient comparison at n=1
# 
# At order n=1, the Taylor coefficient of G(tau) in U is:
#   c_1(tau) = integral_0^beta dtau_1 C_{tau_1}^conn(tau, 0)
#
# where C_{tau_1}^conn is computed via the Rossi recursion:
#   C_{tau_1} = D_{tau_1}(x_out, x_in) - C_empty(x_out, x_in) * D_{tau_1}(empty)
#
# Compare against the symbolic Taylor coefficient from G_exact_atom.
# ===========================================================================
print("Check E: End-to-end Taylor coefficient at n=1")

# Get symbolic Taylor coefficient c_1(tau) from G_exact_atom
mu_s, beta_s, U_s, tau_s = sp.symbols('mu beta U tau', positive=True, real=True)
Z = 1 + 2*sp.exp(beta_s*mu_s) + sp.exp(beta_s*(2*mu_s - U_s))
G_sym = -(sp.exp(mu_s*tau_s) + sp.exp(mu_s*(beta_s + tau_s) - tau_s*U_s)) / Z
G_taylor = sp.series(G_sym, U_s, 0, 3).removeO()
c1_sym = G_taylor.coeff(U_s, 1)

all_ok = True
worst = 0.0
for beta_n, mu_n in [(5.0, 0.3), (3.0, -0.2), (2.0, 0.5)]:
    for tau_n in [0.3*beta_n, 0.5*beta_n, 0.8*beta_n]:
        # Symbolic ground truth
        c1_expected = float(c1_sym.subs([(beta_s, beta_n), (mu_s, mu_n), (tau_s, tau_n)]))
        
        # Compute via integration of CDet:
        # C_empty(x_out, x_in) = D_empty(x_out, x_in) = G_0(tau - 0) = G_0(tau)
        # C_{tau_1}(x_out, x_in) = D_{tau_1}(x_out, x_in) - C_empty * D_{tau_1}(empty)
        # c_1 = integral_0^beta dtau_1 C_{tau_1}
        
        def integrand(tau_1):
            d_corr = D_corr([tau_1], tau_n, 0.0, beta_n, mu_n)
            d_vac = D_vac([tau_1], beta_n, mu_n)
            c_empty = D_corr([], tau_n, 0.0, beta_n, mu_n)  # = G_0(tau_n)
            return d_corr - c_empty * d_vac
        
        c1_integrated, _ = quad(integrand, 0, beta_n, limit=200)
        
        rd = rel_diff(c1_integrated, c1_expected)
        if rd > worst: worst = rd
        if rd > 1e-8:  # quad has finite accuracy
            print(f"    beta={beta_n}, mu={mu_n}, tau={tau_n}: c1_int={c1_integrated:.10f}, c1_sym={c1_expected:.10f}, rd={rd:.2e}")
            all_ok = False

report("E. n=1 end-to-end via integration matches symbolic Taylor (9 sets)",
       all_ok, f"worst rel = {worst:.2e}")
print()

# ===========================================================================
# Check F: End-to-end Taylor coefficient at n=2
# 
# At order n=2:
#   c_2(tau) = (1/2!) * int int dtau_1 dtau_2 C_{tau_1,tau_2}^conn(tau, 0)
#
# where C is given by the Rossi recursion at n=2:
#   C_V = D_V - sum_{S subsetneq V} C_S * D_{V\S}(empty)
# 
# For V = {tau_1, tau_2}:
#   C_V = D_V - C_empty * D_{tau_1,tau_2}(empty)
#             - C_{tau_1} * D_{tau_2}(empty)
#             - C_{tau_2} * D_{tau_1}(empty)
# 
# And C_{tau_1} = D_{tau_1} - C_empty * D_{tau_1}(empty), etc.
# ===========================================================================
print("Check F: End-to-end Taylor coefficient at n=2")

c2_sym = G_taylor.coeff(U_s, 2)

def C_empty(tau_out, tau_in, beta, mu):
    return D_corr([], tau_out, tau_in, beta, mu)

def C_n1(tau_1, tau_out, tau_in, beta, mu):
    return D_corr([tau_1], tau_out, tau_in, beta, mu) - C_empty(tau_out, tau_in, beta, mu) * D_vac([tau_1], beta, mu)

def C_n2(tau_1, tau_2, tau_out, tau_in, beta, mu):
    """Connected correlator at V = {tau_1, tau_2} via the Rossi recursion."""
    V = [tau_1, tau_2]
    D_V = D_corr(V, tau_out, tau_in, beta, mu)
    
    # Subtract disconnected: 3 terms (S = {}, {tau_1}, {tau_2})
    term_empty = C_empty(tau_out, tau_in, beta, mu) * D_vac(V, beta, mu)
    term_t1 = C_n1(tau_1, tau_out, tau_in, beta, mu) * D_vac([tau_2], beta, mu)
    term_t2 = C_n1(tau_2, tau_out, tau_in, beta, mu) * D_vac([tau_1], beta, mu)
    
    return D_V - term_empty - term_t1 - term_t2

all_ok = True
worst = 0.0
for beta_n, mu_n in [(3.0, 0.2), (5.0, -0.1)]:
    for tau_n in [0.3*beta_n, 0.7*beta_n]:
        c2_expected = float(c2_sym.subs([(beta_s, beta_n), (mu_s, mu_n), (tau_s, tau_n)]))
        
        # Compute c_2(tau) = (1/2!) * int int C_{tau_1,tau_2} dtau_1 dtau_2
        def integrand(tau_2, tau_1):
            return C_n2(tau_1, tau_2, tau_n, 0.0, beta_n, mu_n)
        
        c2_int, _ = dblquad(integrand, 0, beta_n, 0, beta_n, epsabs=1e-10, epsrel=1e-10)
        c2_int = c2_int / 2.0  # 1/2!
        
        rd = rel_diff(c2_int, c2_expected)
        if rd > worst: worst = rd
        if rd > 1e-6:
            print(f"    beta={beta_n}, mu={mu_n}, tau={tau_n}: c2_int={c2_int:.10f}, c2_sym={c2_expected:.10f}, rd={rd:.2e}")
            all_ok = False
        else:
            print(f"    beta={beta_n}, mu={mu_n}, tau={tau_n}: c2_int={c2_int:.10f}, c2_sym={c2_expected:.10f}, rd={rd:.2e}")

report("F. n=2 end-to-end via dblquad matches symbolic Taylor (4 sets)",
       all_ok, f"worst rel = {worst:.2e}")
print()

# ===========================================================================
# Summary
# ===========================================================================
print("="*72)
print(f" Summary: {N_PASS} PASS, {N_FAIL} FAIL")
print("="*72)
if N_FAIL == 0:
    print("""
 wick_determinant.py is verified:
   A. n=0 boundary (27 sets)
   B. n=1 explicit formula (36 sets)
   C. n=2 numpy.det vs Leibniz expansion (36 sets), INDEPENDENT path
   D. Permutation invariance (n=2,3,4)
   E. n=1 end-to-end Taylor coefficient (9 sets)
   F. n=2 end-to-end Taylor coefficient (4 sets), strongest test

 Checks E and F take D_corr/D_vac through the Rossi recursion and
 integrate over V. The result matches the analytic Taylor expansion of
 G_exact_atom. This is the end-to-end test that says:
   D_V definition + sign convention + recursion = correct physics.
""")
else:
    print(f" {N_FAIL} failures. See output above.")
