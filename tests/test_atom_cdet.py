"""
Full audit suite for the CDet recursion on the Hubbard atom.

Strategy:
  Check A: n=0 boundary case across parameter sets
  Check B: n=1 against analytic Hartree formula
  Check C: equivalence with manual recursion (independent code path) for n=1, n=2
  Check D: permutation invariance of V (CDet must be set-invariant)
  Check E: end-to-end at n=1 via quad, match symbolic Taylor U^1
  Check F: end-to-end at n=2 via dblquad, match symbolic Taylor U^2
  Check G: end-to-end at n=3 via tplquad, match symbolic Taylor U^3
           This is the strongest test. It pushes the recursion to a
           depth beyond what the Wick-determinant audit covers.

"""
import sys
import math
import numpy as np
import sympy as sp
from itertools import permutations
from scipy.integrate import quad, dblquad, tplquad
from cdet_reference.atom.green_function import G0_atom, G_exact_atom, G0_atom_at_zero_minus
from cdet_reference.atom.wick_determinant import D_corr, D_vac
from cdet_reference.atom.cdet_recursion import C_V


N_PASS = 0
N_FAIL = 0
ABS_FLOOR = 1e-15  # cancellation-zero threshold

def report(name, ok, details=""):
    global N_PASS, N_FAIL
    flag = "PASS" if ok else "FAIL"
    if ok: N_PASS += 1
    else:  N_FAIL += 1
    print(f"  [{flag}] {name}")
    if details:
        print(f"         {details}")


def compare(a, b):
    """Return relative diff, or 0 if both below ABS_FLOOR (cancellation zero)."""
    scale = max(abs(a), abs(b))
    if scale < ABS_FLOOR:
        return 0.0
    return abs(a - b) / scale


print("="*72)
print(" Full audit of cdet_recursion.py")
print("="*72)
print()

# ---------------------------------------------------------------------------
# Check A: n=0 boundary
# ---------------------------------------------------------------------------
print("Check A: n=0 boundary C_empty = G_0(tau_out - tau_in)")
all_ok = True
worst = 0.0
for beta in [0.5, 2.0, 8.0]:
    for mu in [-0.3, 0.0, 0.7]:
        for (tau_out_f, tau_in_f) in [(0.1, 0.05), (0.7, 0.3), (0.95, 0.05)]:
            tau_out = tau_out_f * beta
            tau_in = tau_in_f * beta
            c = C_V([], tau_out, tau_in, beta, mu)
            g = G0_atom(tau_out - tau_in, beta, mu)
            d = compare(c, g)
            if d > worst: worst = d
            if d > 1e-14:
                print(f"    beta={beta}, mu={mu}, tau=({tau_out},{tau_in}): rd={d:.2e}")
                all_ok = False
report("A. n=0 boundary across 27 parameter sets",
       all_ok, f"worst rel = {worst:.2e}")
print()

# ---------------------------------------------------------------------------
# Check B: n=1 analytic Hartree
# ---------------------------------------------------------------------------
print("Check B: n=1 against analytic Hartree C_tau1 = n_0 * G_0(out-tau_1) * G_0(tau_1-in)")
all_ok = True
worst_norm = 0.0   # relative error normalized to cancellation depth
worst_raw = 0.0    # raw relative error (for reporting)
n_skipped = 0
n_tested = 0
for beta in [1.0, 3.0, 7.0]:
    for mu in [-0.4, 0.0, 0.5, 1.2]:
        for (out_f, t1_f, in_f) in [(0.7, 0.5, 0.2), (0.9, 0.1, 0.05), (0.3, 0.6, 0.1), (0.5, 0.5, 0.5)]:
            tau_out = out_f * beta
            tau_1 = t1_f * beta
            tau_in = in_f * beta
            
            n_0 = G0_atom_at_zero_minus(beta, mu)
            g_o1 = G0_atom(tau_out - tau_1, beta, mu)
            g_1i = G0_atom(tau_1 - tau_in, beta, mu)
            expected = n_0 * g_o1 * g_1i
            
            computed = C_V([tau_1], tau_out, tau_in, beta, mu)
            
            scale = max(abs(expected), abs(computed))
            if scale < ABS_FLOOR:
                n_skipped += 1
                continue
            n_tested += 1
            
            # Compute the cancellation depth from the recursion's internals.
            # The recursion computes C = D - C_empty * D_vac.
            # The expected catastrophic-cancellation factor is:
            #     max(|D|, |C_empty * D_vac|) / |C|
            # Multiply the expected relative tolerance (1e-15, machine epsilon)
            # by this factor to get the achievable tolerance.
            D = D_corr([tau_1], tau_out, tau_in, beta, mu)
            Dv = D_vac([tau_1], beta, mu)
            C_empty = D_corr([], tau_out, tau_in, beta, mu)
            cancellation_scale = max(abs(D), abs(C_empty * Dv))
            cancellation_factor = cancellation_scale / max(abs(computed), 1e-300)
            
            raw_rel = compare(computed, expected)
            normalized_rel = raw_rel / max(cancellation_factor, 1.0)
            
            if raw_rel > worst_raw: worst_raw = raw_rel
            if normalized_rel > worst_norm: worst_norm = normalized_rel
            
            # The realistic precision is machine epsilon (~1e-15) times the
            # cancellation factor. Demand the test pass at ~10x that:
            tolerance = max(1e-14, 1e-15 * cancellation_factor * 10)
            if raw_rel > tolerance:
                print(f"    beta={beta}, mu={mu}, tau=({tau_out},{tau_1},{tau_in}): raw_rel={raw_rel:.2e}, cancellation_factor={cancellation_factor:.2e}, tol={tolerance:.2e}")
                all_ok = False
report(f"B. n=1 analytic Hartree ({n_tested} non-trivial / {n_skipped} cancellation-zero)",
       all_ok, f"worst raw rel = {worst_raw:.2e}, worst normalized = {worst_norm:.2e}")
print()

# ---------------------------------------------------------------------------
# Check C: equivalence with manual recursion (independent code path)
# 
# This compares cdet_recursion.C_V against the hand-inlined recursion
# from the Wick-determinant audit. Same physics, different code path.
# ---------------------------------------------------------------------------
print("Check C: cdet_recursion.C_V vs manual hand-inlined recursion")

def C_manual_n1(tau_1, tau_out, tau_in, beta, mu):
    """Manual recursion at n=1."""
    D = D_corr([tau_1], tau_out, tau_in, beta, mu)
    Dv = D_vac([tau_1], beta, mu)
    C_empty = D_corr([], tau_out, tau_in, beta, mu)
    return D - C_empty * Dv


def C_manual_n2(tau_1, tau_2, tau_out, tau_in, beta, mu):
    """Manual recursion at n=2."""
    V = [tau_1, tau_2]
    D = D_corr(V, tau_out, tau_in, beta, mu)
    C_empty = D_corr([], tau_out, tau_in, beta, mu)
    term_empty = C_empty * D_vac(V, beta, mu)
    term_t1 = C_manual_n1(tau_1, tau_out, tau_in, beta, mu) * D_vac([tau_2], beta, mu)
    term_t2 = C_manual_n1(tau_2, tau_out, tau_in, beta, mu) * D_vac([tau_1], beta, mu)
    return D - term_empty - term_t1 - term_t2


all_ok = True
worst = 0.0
for beta in [2.0, 5.0]:
    for mu in [-0.2, 0.3, 0.8]:
        # n=1 cases
        for tau_1_f in [0.3, 0.7, 0.5]:
            tau_1 = tau_1_f * beta
            tau_out = 0.9 * beta
            tau_in = 0.05 * beta
            c_recur = C_V([tau_1], tau_out, tau_in, beta, mu)
            c_manual = C_manual_n1(tau_1, tau_out, tau_in, beta, mu)
            d = compare(c_recur, c_manual)
            if d > worst: worst = d
            if d > 1e-13:
                print(f"    n=1: beta={beta}, mu={mu}, tau_1={tau_1}: rd={d:.2e}")
                all_ok = False
        
        # n=2 cases
        for (t1_f, t2_f) in [(0.2, 0.7), (0.4, 0.6), (0.1, 0.9)]:
            t1 = t1_f * beta
            t2 = t2_f * beta
            tau_out = 0.85 * beta
            tau_in = 0.05 * beta
            c_recur = C_V([t1, t2], tau_out, tau_in, beta, mu)
            c_manual = C_manual_n2(t1, t2, tau_out, tau_in, beta, mu)
            d = compare(c_recur, c_manual)
            if d > worst: worst = d
            if d > 1e-13:
                print(f"    n=2: beta={beta}, mu={mu}, V=({t1},{t2}): rd={d:.2e}, recur={c_recur:.6e}, manual={c_manual:.6e}")
                all_ok = False

report("C. C_V via recursion code vs manual hand-inlined (independent paths)",
       all_ok, f"worst rel = {worst:.2e}")
print()

# ---------------------------------------------------------------------------
# Check D: permutation invariance of V
# 
# CDet must give the same answer for any ordering of V (V is a set).
# ---------------------------------------------------------------------------
print("Check D: C_V invariant under permutation of V")
all_ok = True
worst = 0.0
beta, mu = 4.0, 0.4
for n in [2, 3, 4]:
    V_base = [0.1*beta, 0.3*beta, 0.5*beta, 0.8*beta][:n]
    tau_out = 0.95 * beta
    tau_in = 0.05 * beta
    c_base = C_V(V_base, tau_out, tau_in, beta, mu)
    for perm in permutations(V_base):
        if perm == tuple(V_base):
            continue
        c_perm = C_V(list(perm), tau_out, tau_in, beta, mu)
        d = compare(c_perm, c_base)
        if d > worst: worst = d
        if d > 1e-13:
            print(f"    n={n}, perm={perm}: rd={d:.2e}")
            all_ok = False
report("D. C_V invariance under V permutation (n=2,3,4)", all_ok, f"worst rel = {worst:.2e}")
print()

# ---------------------------------------------------------------------------
# End-to-end Taylor coefficient comparisons
# ---------------------------------------------------------------------------
# Symbolic expansion of G_exact_atom in U
mu_s, beta_s, U_s, tau_s = sp.symbols('mu beta U tau', positive=True, real=True)
Z = 1 + 2*sp.exp(beta_s*mu_s) + sp.exp(beta_s*(2*mu_s - U_s))
G_sym = -(sp.exp(mu_s*tau_s) + sp.exp(mu_s*(beta_s + tau_s) - tau_s*U_s)) / Z
G_taylor = sp.series(G_sym, U_s, 0, 4).removeO()
c1_sym = G_taylor.coeff(U_s, 1)
c2_sym = G_taylor.coeff(U_s, 2)
c3_sym = G_taylor.coeff(U_s, 3)

# ---------------------------------------------------------------------------
# Check E: end-to-end n=1
# ---------------------------------------------------------------------------
print("Check E: end-to-end n=1 via quad against symbolic c_1(tau)")
all_ok = True
worst = 0.0
for beta_n, mu_n in [(5.0, 0.3), (3.0, -0.2), (2.0, 0.5)]:
    for tau_n in [0.3*beta_n, 0.5*beta_n, 0.8*beta_n]:
        c1_exact = float(c1_sym.subs([(beta_s, beta_n), (mu_s, mu_n), (tau_s, tau_n)]))
        integrand = lambda t1: C_V([t1], tau_n, 0.0, beta_n, mu_n)
        c1_int, _ = quad(integrand, 0, beta_n, limit=200)
        d = compare(c1_int, c1_exact)
        if d > worst: worst = d
        if d > 1e-8:
            print(f"    beta={beta_n}, mu={mu_n}, tau={tau_n}: rd={d:.2e}, exact={c1_exact}, int={c1_int}")
            all_ok = False
report("E. n=1 end-to-end (9 sets)", all_ok, f"worst rel = {worst:.2e}")
print()

# ---------------------------------------------------------------------------
# Check F: end-to-end n=2
# ---------------------------------------------------------------------------
print("Check F: end-to-end n=2 via dblquad against symbolic c_2(tau)")
all_ok = True
worst = 0.0
for beta_n, mu_n in [(3.0, 0.2), (5.0, -0.1)]:
    for tau_n in [0.3*beta_n, 0.7*beta_n]:
        c2_exact = float(c2_sym.subs([(beta_s, beta_n), (mu_s, mu_n), (tau_s, tau_n)]))
        integrand = lambda t2, t1: C_V([t1, t2], tau_n, 0.0, beta_n, mu_n)
        c2_int, _ = dblquad(integrand, 0, beta_n, 0, beta_n, epsabs=1e-10, epsrel=1e-10)
        c2_int = c2_int / 2.0   # 1/2! for two identical vertices
        d = compare(c2_int, c2_exact)
        if d > worst: worst = d
        print(f"    beta={beta_n}, mu={mu_n}, tau={tau_n}: c2_int={c2_int:.10f}, c2_sym={c2_exact:.10f}, rd={d:.2e}")
        if d > 1e-6:
            all_ok = False
report("F. n=2 end-to-end (4 sets)", all_ok, f"worst rel = {worst:.2e}")
print()

# ---------------------------------------------------------------------------
# Check G: end-to-end n=3 (THE deepest test)
# ---------------------------------------------------------------------------
print("Check G: end-to-end n=3 via tplquad against symbolic c_3(tau)")
print("  (this is the deepest verification; takes ~minute per point)")
all_ok = True
worst = 0.0

# Use modest parameters for tractability: fewer sets, lower precision tolerance
for beta_n, mu_n in [(2.0, 0.1)]:
    for tau_n in [0.5*beta_n]:
        c3_exact = float(c3_sym.subs([(beta_s, beta_n), (mu_s, mu_n), (tau_s, tau_n)]))
        
        # Triple integral of C_V over [0, beta]^3
        # tplquad signature: f(z, y, x), x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
        integrand = lambda t3, t2, t1: C_V([t1, t2, t3], tau_n, 0.0, beta_n, mu_n)
        
        import time
        t_start = time.time()
        c3_int, abserr = tplquad(integrand, 0, beta_n, 0, beta_n, 0, beta_n,
                                  epsabs=1e-8, epsrel=1e-8)
        c3_int = c3_int / 6.0   # 1/3! for three identical vertices
        elapsed = time.time() - t_start
        
        d = compare(c3_int, c3_exact)
        if d > worst: worst = d
        print(f"    beta={beta_n}, mu={mu_n}, tau={tau_n}: c3_int={c3_int:.10f}, c3_sym={c3_exact:.10f}, rd={d:.2e}")
        print(f"      tplquad elapsed: {elapsed:.1f} s, reported abs err: {abserr:.2e}")
        if d > 1e-5:  # Looser tolerance, tplquad accuracy is limited
            all_ok = False
report("G. n=3 end-to-end (1 set)", all_ok, f"worst rel = {worst:.2e}")
print()

# ---------------------------------------------------------------------------
print("="*72)
print(f" Summary: {N_PASS} PASS, {N_FAIL} FAIL")
print("="*72)
if N_FAIL == 0:
    print("""
 cdet_recursion.py verified:
   A. n=0 boundary case (27 sets)
   B. n=1 analytic Hartree (12+ sets)
   C. Recursion code vs manual hand-inlined recursion (independent paths)
   D. Permutation invariance under V (n=2,3,4)
   E. n=1 end-to-end Taylor coefficient (9 sets)
   F. n=2 end-to-end Taylor coefficient (4 sets)
   G. n=3 end-to-end Taylor coefficient (1 set, the deepest test)
 
 The CDet recursion is producing perturbative Taylor coefficients of
 G_exact_atom(tau; U) correctly through order 3 in U.
""")
