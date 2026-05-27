"""
Rossi CDet recursion for the 4-site pi-flux square Hubbard model.

Structurally identical to cdet_recursion_2site.py; the recursion is
site-agnostic (operates on bitmask subsets of V), so only the underlying
D_corr / D_vac calls change.

  C_V(x_out, x_in) = D_V(x_out, x_in) - sum_{S subsetneq V} C_S(x_out, x_in) * D_{V\\S}(empty)

V is a list of (site, tau) tuples with site in {0, 1, 2, 3}, tau in (0, beta).
x_out, x_in are (site, tau) tuples for the external operators.

"""
import sys
from cdet_reference.piflux_square.wick_determinant import D_corr_4site_piflux, D_vac_4site_piflux


def popcount(x):
    return bin(x).count('1')


def C_V_4site_piflux(V, x_out, x_in, beta, mu, t):
    """Connected correlator at vertex set V for the 4-site pi-flux Hubbard model.
    
    Args:
      V: list of (site, tau) tuples; site in {0,1,2,3}, tau in (0, beta); can be empty
      x_out: (site, tau), external c_up operator
      x_in: (site, tau), external c_up+ operator
      beta, mu, t: model parameters
    
    Returns:
      C_V(x_out, x_in) via the Rossi recursion.
    """
    V = list(V)
    n = len(V)
    
    if n == 0:
        return D_corr_4site_piflux([], x_out, x_in, beta, mu, t)
    
    # Precompute D values for every subset
    D_corr_table = {}
    D_vac_table = {}
    for mask in range(1 << n):
        S = [V[i] for i in range(n) if mask & (1 << i)]
        D_corr_table[mask] = D_corr_4site_piflux(S, x_out, x_in, beta, mu, t)
        D_vac_table[mask] = D_vac_4site_piflux(S, beta, mu, t)
    
    # Build C_S bottom-up by popcount
    C_table = {}
    for k in range(n + 1):
        for mask in range(1 << n):
            if popcount(mask) != k:
                continue
            
            value = D_corr_table[mask]
            
            # Iterate proper subsets via the standard submask trick
            sm = (mask - 1) & mask
            while True:
                if sm != mask:
                    complement = mask ^ sm
                    value -= C_table[sm] * D_vac_table[complement]
                if sm == 0:
                    break
                sm = (sm - 1) & mask
            
            C_table[mask] = value
    
    full_mask = (1 << n) - 1
    return C_table[full_mask]


# ============================================================================
# Sanity checks
# ============================================================================
if __name__ == "__main__":
    from cdet_reference.atom.cdet_recursion import C_V
    from cdet_reference.dimer.cdet_recursion import C_V_2site
    
    print("=" * 70)
    print("  cdet_recursion_4site_piflux.py, sanity checks")
    print("=" * 70)
    print()
    
    beta = 3.0
    mu = 0.3
    
    # Test 1: at t = 0, all vertices on site 0 -> match atom CDet
    print("Test 1: t=0 + all vertices on site 0 -> match atom CDet")
    x_out_4 = (0, 1.5)
    x_in_4 = (0, 0.5)
    
    for V_atom in [[], [1.0], [1.0, 2.0], [0.7, 1.5, 2.3]]:
        V_4 = [(0, tau) for tau in V_atom]
        c_atom = C_V(V_atom, 1.5, 0.5, beta, mu)
        c_4 = C_V_4site_piflux(V_4, x_out_4, x_in_4, beta, mu, t=0.0)
        diff = abs(c_atom - c_4)
        print(f"  n={len(V_atom)}: C_atom = {c_atom:+.12f}, C_4 = {c_4:+.12f}, diff = {diff:.2e}")
    print()
    
    # Test 2: at t = 0, vertex on isolated site -> C_V = 0 (disconnected)
    print("Test 2: t=0, vertex on site disconnected from externals -> C_V = 0")
    
    # Externals on site 0; vertex on site 2 (always disconnected at t=0)
    V_iso = [(2, 1.0)]
    c_iso = C_V_4site_piflux(V_iso, x_out_4, x_in_4, beta, mu, t=0.0)
    print(f"  V=[(2, 1.0)]: C = {c_iso:.2e}  (should be 0)")
    
    # With multiple vertices, mixing site 0 and site 2 at t=0: still disconnected
    V_mix = [(0, 1.0), (2, 2.0)]
    c_mix = C_V_4site_piflux(V_mix, x_out_4, x_in_4, beta, mu, t=0.0)
    print(f"  V=[(0,1.0), (2,2.0)] at t=0: C = {c_mix:.2e}")
    print(f"  (Site 2 vertex disconnected -> C_V = 0)")
    print()
    
    # Test 3: at t > 0, all vertices on site 0: sanity that result differs from atom
    print("Test 3: At t = 1 with all vertices on site 0, hopping changes C_V")
    V_4 = [(0, 1.0)]
    c_4_t0 = C_V_4site_piflux(V_4, x_out_4, x_in_4, beta, mu, t=0.0)
    c_4_t1 = C_V_4site_piflux(V_4, x_out_4, x_in_4, beta, mu, t=1.0)
    print(f"  V=[(0, 1.0)]:")
    print(f"    t=0: C = {c_4_t0:.10f}")
    print(f"    t=1: C = {c_4_t1:.10f}")
    print()
    
    # Test 4: pi-flux opposite-corner external -> C_empty = 0
    print("Test 4: opposite-corner external operators -> C_empty = 0 (pi-flux structural)")
    x_out_opp = (2, 1.5)
    x_in_opp = (0, 0.5)
    c_opp_n0 = C_V_4site_piflux([], x_out_opp, x_in_opp, beta, mu, t=1.0)
    print(f"  C_empty(x_out=(2,1.5), x_in=(0,0.5)) at t=1: {c_opp_n0:.2e}")
    
    # At higher orders the opposite-corner external could in principle pick
    # up nonzero contributions from bridging vertices. The gauge-symmetry
    # argument (c_2 -> -c_2) says it stays zero at all orders; we verify at
    # n=1 and n=2 below.
    V_a = [(0, 1.0)]
    V_b = [(1, 1.0)]
    V_c = [(0, 0.8), (1, 1.6)]
    
    c1a = C_V_4site_piflux(V_a, x_out_opp, x_in_opp, beta, mu, t=1.0)
    c1b = C_V_4site_piflux(V_b, x_out_opp, x_in_opp, beta, mu, t=1.0)
    c2c = C_V_4site_piflux(V_c, x_out_opp, x_in_opp, beta, mu, t=1.0)
    print(f"  C_(0,1.0)(2<-0) at t=1: {c1a:.2e}")
    print(f"  C_(1,1.0)(2<-0) at t=1: {c1b:.2e}")
    print(f"  C_[(0,0.8),(1,1.6)](2<-0) at t=1: {c2c:.2e}")
    print("  (All should be 0 by gauge symmetry c_2 -> -c_2)")
    print()
    
    # Test 5: Permutation invariance
    print("Test 5: Permutation invariance of V")
    V_a = [(0, 0.7), (1, 1.5), (3, 2.3)]
    V_b = [(1, 1.5), (0, 0.7), (3, 2.3)]
    V_c = [(3, 2.3), (1, 1.5), (0, 0.7)]
    
    c_a = C_V_4site_piflux(V_a, x_out_4, x_in_4, beta, mu, t=1.0)
    c_b = C_V_4site_piflux(V_b, x_out_4, x_in_4, beta, mu, t=1.0)
    c_c = C_V_4site_piflux(V_c, x_out_4, x_in_4, beta, mu, t=1.0)
    print(f"  V = {V_a}: C = {c_a:.12f}")
    print(f"  V = {V_b}: C = {c_b:.12f}")
    print(f"  V = {V_c}: C = {c_c:.12f}")
    max_diff = max(abs(c_a - c_b), abs(c_b - c_c), abs(c_a - c_c))
    print(f"  Max diff: {max_diff:.2e}")
