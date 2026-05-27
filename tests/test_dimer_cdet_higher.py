"""
End-to-end audit of the 2-site CDet pipeline at order n=3.

Uses:
  - mpmath at dps=50 for exact Taylor coefficient reference values
  - Kink-aware Gauss-Legendre quadrature for fast convergent CDet integration

The CDet integrand has kinks (derivative discontinuities) at
vertex-coincidence times (when a vertex time equals an external time, or
another vertex's time). G_0(tau) has different functional form for tau>0
versus tau<0, so the determinants in the Wick formula develop kinks.

Plain scipy.tplquad cannot converge on these in reasonable time. Plain
fixed-grid Gauss-Legendre doesn't converge either (Gauss-Legendre's
exponential convergence requires smooth integrands).

The fix: nested integration where inner integrals dynamically include
the outer variable's value as a kink point, then plain Gauss-Legendre
on each smooth piece.

Computational cost at n=3: 8 site configs x ~10s/config = ~80s total.

"""
import sys
import time
import mpmath
from cdet_reference.dimer.cdet_recursion import C_V_2site
from cdet_reference.dimer.exact_taylor import G_exact_2site_mp, taylor_coeff
from cdet_reference.quadrature.quadrature_nested import integrate_1d_with_kinks, integrate_nested_2d, integrate_nested_3d


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
print(" Higher-order audit of 2-site CDet pipeline")
print("="*72)
print()

# Parameters
beta = 2.0
mu = 0.3
t = 1.0
tau = 1.0
i_out, i_in = 0, 0
x_out = (i_out, tau)
x_in = (i_in, 0.0)
fixed_kinks = [tau]  # tau_in = 0 is on boundary, only tau_out is interior kink

print(f"Parameters: beta={beta}, mu={mu}, t={t}, tau={tau}, (i_out, i_in)=({i_out}, {i_in})")
print()

# ---------------------------------------------------------------------------
# Step 1: Reference Taylor coefficients via mpmath (dps=50)
# ---------------------------------------------------------------------------
print("Step 1: Compute reference Taylor coefficients via mpmath (dps=50)")
def G_func(U):
    return G_exact_2site_mp(i_out, i_in, tau, beta, mu, t, U=U, dps=50)

t0 = time.time()
c_ref = [
    taylor_coeff(G_func, 0, dps=50),
    taylor_coeff(G_func, 1, h='1e-6', dps=50),
    taylor_coeff(G_func, 2, h='1e-6', dps=50),
    taylor_coeff(G_func, 3, h='1e-6', dps=50),
]
ref_time = time.time() - t0
for n, cn in enumerate(c_ref):
    print(f"  c_{n} = {mpmath.nstr(cn, 18)}")
print(f"  (mpmath reference: {ref_time:.1f}s)")
print()

c_ref_float = [float(cn) for cn in c_ref]

# ---------------------------------------------------------------------------
# Check A: CDet n=0
# ---------------------------------------------------------------------------
print("Check A: CDet n=0 matches c_0 reference")
c0_cdet = C_V_2site([], x_out, x_in, beta, mu, t)
d = compare(c0_cdet, c_ref_float[0])
report(f"A. c_0: cdet={c0_cdet:.16f}, ref={c_ref_float[0]:.16f}",
       d < 1e-14, f"rel = {d:.2e}")
print()

# ---------------------------------------------------------------------------
# Check B: CDet n=1
# ---------------------------------------------------------------------------
print("Check B: CDet n=1 via kink-aware 1D integration")
t0 = time.time()
c1_cdet = 0.0
for i_v in [0, 1]:
    val = integrate_1d_with_kinks(
        lambda t1: C_V_2site([(i_v, t1)], x_out, x_in, beta, mu, t),
        16, beta, fixed_kinks
    )
    c1_cdet += val
elapsed_n1 = time.time() - t0
d = compare(c1_cdet, c_ref_float[1])
report(f"B. c_1: cdet={c1_cdet:.12f}, ref={c_ref_float[1]:.12f}",
       d < 1e-10, f"rel = {d:.2e} ({elapsed_n1:.1f}s)")
print()

# ---------------------------------------------------------------------------
# Check C: CDet n=2
# ---------------------------------------------------------------------------
print("Check C: CDet n=2 via kink-aware 2D integration")
t0 = time.time()
c2_cdet = 0.0
for i1 in [0, 1]:
    for i2 in [0, 1]:
        val = integrate_nested_2d(
            lambda t1, t2: C_V_2site([(i1, t1), (i2, t2)], x_out, x_in, beta, mu, t),
            12, beta, fixed_kinks
        )
        c2_cdet += val
c2_cdet /= 2.0  # 1/2!
elapsed_n2 = time.time() - t0
d = compare(c2_cdet, c_ref_float[2])
report(f"C. c_2: cdet={c2_cdet:.12f}, ref={c_ref_float[2]:.12f}",
       d < 1e-9, f"rel = {d:.2e} ({elapsed_n2:.1f}s)")
print()

# ---------------------------------------------------------------------------
# Check D: CDet n=3: THE NEW HIGH-ORDER TEST
# ---------------------------------------------------------------------------
print("Check D: CDet n=3 via kink-aware 3D integration (8 site configs)")
t0 = time.time()
c3_cdet = 0.0
n_grid = 10  # found converged at this level on test
config_contribs = {}
for i1 in [0, 1]:
    for i2 in [0, 1]:
        for i3 in [0, 1]:
            sub_t0 = time.time()
            val = integrate_nested_3d(
                lambda t1, t2, t3: C_V_2site(
                    [(i1, t1), (i2, t2), (i3, t3)], x_out, x_in, beta, mu, t
                ),
                n_grid, beta, fixed_kinks
            )
            sub_elapsed = time.time() - sub_t0
            c3_cdet += val
            config_contribs[(i1, i2, i3)] = val
            print(f"    (i1,i2,i3)=({i1},{i2},{i3}): contribution = {val:.6e} ({sub_elapsed:.1f}s)")
c3_cdet /= 6.0  # 1/3!
elapsed_n3 = time.time() - t0
d = compare(c3_cdet, c_ref_float[3])
report(f"D. c_3: cdet={c3_cdet:.10f}, ref={c_ref_float[3]:.10f}",
       d < 1e-6, f"rel = {d:.2e} (total {elapsed_n3/60:.1f}m)")
print()

# Convergence check
print("Check E: Convergence on one n=3 config across grid sizes")
f_one = lambda t1, t2, t3: C_V_2site([(0, t1), (0, t2), (0, t3)], x_out, x_in, beta, mu, t)
prev = None
all_stable = True
for n_test in [8, 10, 12]:
    t0 = time.time()
    v = integrate_nested_3d(f_one, n_test, beta, fixed_kinks)
    if prev is not None:
        d_grid = abs(v - prev) / max(abs(v), abs(prev), 1e-300)
        print(f"  n_grid={n_test}: integral = {v:.14f} (rel chg vs prev: {d_grid:.2e}, {time.time()-t0:.1f}s)")
        if d_grid > 1e-8:
            all_stable = False
    else:
        print(f"  n_grid={n_test}: integral = {v:.14f} ({time.time()-t0:.1f}s)")
    prev = v
report("E. n=3 integration stable across grid sizes (n=8 -> n=12)",
       all_stable, "stable to 1e-8 or better")
print()

# ---------------------------------------------------------------------------
print("="*72)
print(f" Summary: {N_PASS} PASS, {N_FAIL} FAIL")
print("="*72)
if N_FAIL == 0:
    print("""
 Higher-order 2-site CDet verification:
   A. n=0: machine precision against mpmath reference
   B. n=1: kink-aware integration, matched to ~10 digits
   C. n=2: kink-aware integration, matched to ~9 digits
   D. n=3: kink-aware 3D integration, 8 site configs, matched to ~6 digits
   E. Convergence: stable across grid sizes

 The 2-site CDet pipeline produces correct perturbative Taylor coefficients
 of the ED Green's function through 3 loops. Convergence required kink-aware
 integration: the CDet integrand has derivative discontinuities at
 vertex-coincidence times.
""")
else:
    print(f" {N_FAIL} failure(s).")
