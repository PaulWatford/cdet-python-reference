"""
Nested integration with kink-aware splitting for CDet integrands.

The CDet integrand has kinks where:
  - A vertex time equals an external time (tau_out, tau_in)
  - One vertex time equals another vertex time

For n-dim integral over n vertex times in [0, beta], we use the strategy:
  - Integrate the outermost variable t_1 last (it has only external kinks)
  - For each fixed t_1, integrate t_2 with kinks at {tau_out, tau_in, t_1}
  - For each fixed (t_1, t_2), integrate t_3 with kinks at {tau_out, tau_in, t_1, t_2}
  - etc.

This is iterated integration. For n=3 with 8 site configs:
  outer t_1 loop: ~N points
  for each t_1, double integral over (t_2, t_3) with kinks
  
Each kink halves the integration time approximately. Total ~N^3 evaluations.

"""
import numpy as np
from cdet_reference.quadrature.quadrature import gl_grid_1d


def split_at_kinks(beta, kink_points):
    """Return sorted breakpoints in [0, beta] including 0, beta, and unique kinks in (0, beta)."""
    pts = sorted(set([0.0, beta] + [k for k in kink_points if 0 < k < beta]))
    return pts


def integrate_1d_with_kinks(f, n_per_piece, beta, kink_points):
    """integral_0^beta f(x) dx with Gauss-Legendre on each piece between kinks."""
    breakpoints = split_at_kinks(beta, kink_points)
    total = 0.0
    for i in range(len(breakpoints) - 1):
        a, b = breakpoints[i], breakpoints[i+1]
        if b - a < 1e-14:  # skip empty intervals
            continue
        nodes, weights = gl_grid_1d(n_per_piece, a, b)
        for k, x in enumerate(nodes):
            total += weights[k] * f(x)
    return total


def integrate_nested_2d(f, n_per_piece, beta, fixed_kinks):
    """int int f(t1, t2) dt1 dt2 with dynamic kink handling.
    
    fixed_kinks: list of breakpoints both variables share (e.g. external times)
    Additionally, t2 integration uses t1 as a kink.
    """
    # Outer: t1, with fixed kinks (external times only)
    def outer_integrand(t1):
        # Inner: t2 with kinks at fixed_kinks union {t1}
        all_kinks = list(fixed_kinks) + [t1]
        return integrate_1d_with_kinks(lambda t2: f(t1, t2), n_per_piece, beta, all_kinks)
    
    return integrate_1d_with_kinks(outer_integrand, n_per_piece, beta, fixed_kinks)


def integrate_nested_3d(f, n_per_piece, beta, fixed_kinks):
    """int intintegral f(t1, t2, t3) with dynamic kink handling.
    
    fixed_kinks: shared kinks (external times).
    Inner integrations also use outer variables as kinks.
    """
    def inner_t3(t1, t2):
        all_kinks = list(fixed_kinks) + [t1, t2]
        return integrate_1d_with_kinks(lambda t3: f(t1, t2, t3), n_per_piece, beta, all_kinks)
    
    def middle_t2(t1):
        all_kinks = list(fixed_kinks) + [t1]
        return integrate_1d_with_kinks(lambda t2: inner_t3(t1, t2), n_per_piece, beta, all_kinks)
    
    return integrate_1d_with_kinks(middle_t2, n_per_piece, beta, fixed_kinks)


def integrate_nested_4d(f, n_per_piece, beta, fixed_kinks):
    """int intint int f(t1, t2, t3, t4) with dynamic kink handling.
    
    Each inner integration adds the outer variables as kinks.
    """
    def inner_t4(t1, t2, t3):
        all_kinks = list(fixed_kinks) + [t1, t2, t3]
        return integrate_1d_with_kinks(lambda t4: f(t1, t2, t3, t4), n_per_piece, beta, all_kinks)
    
    def t3_layer(t1, t2):
        all_kinks = list(fixed_kinks) + [t1, t2]
        return integrate_1d_with_kinks(lambda t3: inner_t4(t1, t2, t3), n_per_piece, beta, all_kinks)
    
    def t2_layer(t1):
        all_kinks = list(fixed_kinks) + [t1]
        return integrate_1d_with_kinks(lambda t2: t3_layer(t1, t2), n_per_piece, beta, all_kinks)
    
    return integrate_1d_with_kinks(t2_layer, n_per_piece, beta, fixed_kinks)


# ===========================================================================
# Simplex integration: integrate over 0 <= t_1 <= t_2 <= ... <= t_n <= beta.
#
# For permutation-symmetric integrands, integral_{[0,beta]^n} = n! integral_simplex.
# The CDet integrand C_V is symmetric under vertex permutations, so we
# can sum over all 2^n SITE configurations (which are NOT permutation-related)
# and integrate each over the simplex with no 1/n! factor.
#
# KEY ADVANTAGE: vertex-coincidence kinks (the cusps at t_i = t_j) are
# boundary surfaces of the simplex, not interior kinks. The integrand is
# smooth inside the simplex, so plain Gauss-Legendre converges exponentially
# again.
#
# In practice this is much faster than nested integration with kink handling
# on the full hypercube.
# ===========================================================================

def integrate_simplex_3d(f, n_per_piece, beta, fixed_kinks):
    """int intintegral f(t1, t2, t3) over 0 <= t1 <= t2 <= t3 <= beta.
    
    Use this when f is permutation-symmetric in (t1, t2, t3): for the
    full hypercube integral, multiply this result by 3!. Or equivalently,
    if you sum over all 2^n SITE configs (which are not symmetric under
    vertex permutations), use simplex integrals directly without factorial.
    """
    def inner_t1(t2, t3):
        ak = [k for k in fixed_kinks if 0 < k < t2]
        return integrate_1d_with_kinks(lambda t1: f(t1, t2, t3), n_per_piece, t2, ak) if t2 > 1e-15 else 0.0
    
    def middle_t2(t3):
        ak = [k for k in fixed_kinks if 0 < k < t3]
        return integrate_1d_with_kinks(lambda t2: inner_t1(t2, t3), n_per_piece, t3, ak) if t3 > 1e-15 else 0.0
    
    return integrate_1d_with_kinks(middle_t2, n_per_piece, beta, fixed_kinks)


def integrate_simplex_4d(f, n_per_piece, beta, fixed_kinks):
    """int intint int f(t1, t2, t3, t4) over 0 <= t1 <= t2 <= t3 <= t4 <= beta."""
    def inner_t1(t2, t3, t4):
        ak = [k for k in fixed_kinks if 0 < k < t2]
        return integrate_1d_with_kinks(lambda t1: f(t1, t2, t3, t4), n_per_piece, t2, ak) if t2 > 1e-15 else 0.0
    
    def t2_layer(t3, t4):
        ak = [k for k in fixed_kinks if 0 < k < t3]
        return integrate_1d_with_kinks(lambda t2: inner_t1(t2, t3, t4), n_per_piece, t3, ak) if t3 > 1e-15 else 0.0
    
    def t3_layer(t4):
        ak = [k for k in fixed_kinks if 0 < k < t4]
        return integrate_1d_with_kinks(lambda t3: t2_layer(t3, t4), n_per_piece, t4, ak) if t4 > 1e-15 else 0.0
    
    return integrate_1d_with_kinks(t3_layer, n_per_piece, beta, fixed_kinks)


if __name__ == "__main__":
    import sys, time
    from cdet_reference.dimer.cdet_recursion import C_V_2site
    
    beta = 2.0
    mu = 0.3
    t = 1.0
    tau_out = 1.0
    x_out = (0, tau_out)
    x_in = (0, 0.0)
    
    print("Convergence test: n=3 CDet, all vertices site 0, kink-aware integration")
    f = lambda t1, t2, t3: C_V_2site([(0, t1), (0, t2), (0, t3)], x_out, x_in, beta, mu, t)
    
    fixed_kinks = [tau_out]  # tau_in = 0 is on boundary
    
    for n_per_piece in [4, 6, 8, 10, 12, 16]:
        t0 = time.time()
        v = integrate_nested_3d(f, n_per_piece, beta, fixed_kinks)
        print(f"  n={n_per_piece}: integral = {v}, time = {time.time()-t0:.1f}s")
