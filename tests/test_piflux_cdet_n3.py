"""
End-to-end audit of the 4-site pi-flux CDet pipeline at order n=3.

Reference: hybrid-precision ED (float64 eigendecomposition + mpmath matrix
arithmetic for the spectral sum) at dps=30, providing ~12-digit precision
for Taylor coefficients via 7-point central stencil.

CDet: simplex integration summing over 4^3 = 64 site configurations.

For permutation-symmetric integrand (C_V is invariant under joint
permutation of (site_i, tau_i) pairs), the simplex 0 <= tau_1 <= tau_2 <= tau_3 <= beta
replaces the hypercube divided by 3!. With 64 site configs each integrated
over the simplex, we get the n=3 contribution.

Total expected runtime: ~60s reference + ~2-3 minutes CDet.

"""
import sys
import time
import numpy as np
import mpmath

from cdet_reference.piflux_square.cdet_recursion import C_V_4site_piflux
from cdet_reference.piflux_square.exact_diagonalization import build_4site_operators, build_H_piflux
from cdet_reference.quadrature.quadrature_nested import integrate_simplex_3d


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
# Hybrid mpmath ED reference
# ============================================================================
def G_exact_hybrid(i, j, tau, beta, mu, t, U, dps=30):
    """Hybrid float64-eigendecomp + mpmath-spectral-sum G_exact.
    
    Used for high-precision Taylor coefficient extraction. Bottleneck is
    the N^2 double sum (256^2 = 65536 terms) in mpmath arithmetic.
    """
    mpmath.mp.dps = dps
    ops = build_4site_operators()
    H = build_H_piflux(beta, mu, t, float(U), ops=ops)
    eigvals_np, eigvecs_np = np.linalg.eigh(H)
    
    c_i = ops[f'c_up_{i}']
    cdag_j = ops[f'cdag_up_{j}']
    c_eig_np = eigvecs_np.T @ c_i @ eigvecs_np
    cdag_eig_np = eigvecs_np.T @ cdag_j @ eigvecs_np
    
    e_min = float(eigvals_np.min())
    
    Z = mpmath.mpf(0)
    for em in eigvals_np:
        Z += mpmath.exp(-beta * (em - e_min))
    
    num = mpmath.mpf(0)
    N = len(eigvals_np)
    for m in range(N):
        em = float(eigvals_np[m])
        wm = mpmath.exp(-(beta - tau) * (em - e_min))
        for n in range(N):
            cmn = float(c_eig_np[m, n])
            if cmn == 0:
                continue
            en = float(eigvals_np[n])
            wn = mpmath.exp(-tau * (en - e_min))
            cdnm = float(cdag_eig_np[n, m])
            num += wm * cmn * wn * cdnm
    
    return -num / Z


def taylor_coeff_3(func, h='1e-3', dps=30):
    """c_3 = f'''(0) / 6 via 7-point central stencil at given h."""
    mpmath.mp.dps = dps
    h_mp = mpmath.mpf(h)
    
    g3h  = func(+3*h_mp)
    g2h  = func(+2*h_mp)
    gh   = func(+h_mp)
    gmh  = func(-h_mp)
    gm2h = func(-2*h_mp)
    gm3h = func(-3*h_mp)
    
    fppp = (-g3h + 8*g2h - 13*gh + 13*gmh - 8*gm2h + gm3h) / (8 * h_mp**3)
    return fppp / 6  # c_3 = f'''(0) / 3!


# ============================================================================
# Setup
# ============================================================================
beta = 2.0
mu = 0.3
t = 1.0
tau = 1.0
i_out, i_in = 0, 0
x_out = (i_out, tau)
x_in = (i_in, 0.0)
fixed_kinks = [tau]

print("=" * 72)
print("  4-site pi-flux Hubbard CDet audit at n=3")
print("=" * 72)
print()
print(f"Parameters: beta={beta}, mu={mu}, t={t}, tau={tau}, externals at site {i_out}")
print()

# ============================================================================
# Step 1: c_3 reference via hybrid mpmath
# ============================================================================
print("Step 1: Computing c_3 reference via hybrid mpmath (dps=30)...")
print("  This is 6 ED evaluations x ~2s each = ~12s")
print()

t0_ref = time.time()
def G_func(U):
    return G_exact_hybrid(i_out, i_in, tau, beta, mu, t, U=U, dps=30)

c_3_ref = taylor_coeff_3(G_func, h='1e-3', dps=30)
ref_time = time.time() - t0_ref
print(f"  c_3 reference = {mpmath.nstr(c_3_ref, 18)}")
print(f"  ({ref_time:.1f}s total)")
print()

c_3_ref_float = float(c_3_ref)

# Sanity: at h=1e-2, h=1e-3, h=1e-4 should agree to ~10 digits
print("  Stencil precision check at three h values:")
for h in ['1e-2', '1e-3', '1e-4']:
    c3_h = taylor_coeff_3(G_func, h=h, dps=30)
    print(f"    h = {h}: c_3 = {mpmath.nstr(c3_h, 14)}")
print()


# ============================================================================
# Step 2: CDet n=3 via simplex integration (64 site configs)
# ============================================================================
print("Step 2: CDet n=3 via simplex integration (64 site configs x grid=8)...")
t0_cdet = time.time()

n_grid = 8
c_3_cdet = 0.0

site_contribs = {}  # collect by unordered site multiset

for s1 in range(4):
    for s2 in range(4):
        for s3 in range(4):
            sub_t0 = time.time()
            f_cfg = (
                lambda t1, t2, t3, s1=s1, s2=s2, s3=s3:
                C_V_4site_piflux([(s1, t1), (s2, t2), (s3, t3)],
                                 x_out, x_in, beta, mu, t)
            )
            v = integrate_simplex_3d(f_cfg, n_grid, beta, fixed_kinks)
            c_3_cdet += v
            
            key = tuple(sorted([s1, s2, s3]))
            site_contribs[key] = site_contribs.get(key, 0.0) + v
            
            sub_elapsed = time.time() - sub_t0
            # Only print non-tiny contributions
            if abs(v) > 1e-12:
                print(f"  ({s1},{s2},{s3}): {v:+.6e} ({sub_elapsed:.2f}s)")
elapsed = time.time() - t0_cdet
print()
print(f"  CDet c_3 (sum 64 configs, simplex): {c_3_cdet:.12f}")
print(f"  Reference (mpmath dps=30):           {c_3_ref_float:.12f}")
print(f"  ({elapsed/60:.1f} min total)")
print()

diff = compare(c_3_cdet, c_3_ref_float)
report(f"A. c_3(CDet) = {c_3_cdet:.10f}, c_3(ref) = {c_3_ref_float:.10f}",
       diff < 1e-7, f"rel = {diff:.2e}")
print()


# ============================================================================
# Step 3: structural breakdown by unordered site multiset
# ============================================================================
print("Step 3: Structural breakdown by unordered site multiset")
print()
print(f"{'site multiset':>15} | {'total contribution':>22}")
print('-' * 45)

total_check = 0.0
for key in sorted(site_contribs.keys()):
    val = site_contribs[key]
    print(f"  {key!s:>13} | {val:+.10e}")
    total_check += val
print()
print(f"  total = {total_check:.10f}")
print()

# Look for pattern: which configs are exact-zero?
print("Configurations giving machine-zero contribution:")
for key, val in sorted(site_contribs.items()):
    if abs(val) < 1e-10:
        print(f"  {key}: {val:+.2e}")
print()


# ============================================================================
# Summary
# ============================================================================
print("=" * 72)
print(f"  Summary: {N_PASS} PASS, {N_FAIL} FAIL")
print("=" * 72)
