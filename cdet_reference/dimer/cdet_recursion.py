"""
Rossi CDet recursion for the 2-site Hubbard model.

Structurally identical to cdet_recursion.py for the atom; the only
difference is that the underlying D_corr / D_vac calls go through the
2-site versions that handle (site, time) vertices.

  C_V(x_out, x_in) = D_V(x_out, x_in) - sum_{S subsetneq V} C_S(x_out, x_in) * D_{V\\S}(empty)

V is now a list of (site, tau) tuples.
x_out, x_in are (site, tau) tuples.

"""
import sys
from cdet_reference.dimer.wick_determinant import D_corr_2site, D_vac_2site


def popcount(x):
    return bin(x).count('1')


def C_V_2site(V, x_out, x_in, beta, mu, t):
    """Connected correlator at vertex set V for the 2-site Hubbard model.
    
    Args:
      V:        list of (site, tau) tuples; can be empty
      x_out:    (site, tau) tuple, external c_up operator
      x_in:     (site, tau) tuple, external c_up+ operator
      beta, mu, t: Hubbard parameters
    
    Returns:
      C_V(x_out, x_in) via the Rossi recursion.
    """
    V = list(V)
    n = len(V)
    
    if n == 0:
        # C_empty = D_empty = G_0(x_out, x_in)
        return D_corr_2site([], x_out, x_in, beta, mu, t)
    
    # Precompute D values for every subset (bitmask of length n)
    D_corr_table = {}
    D_vac_table = {}
    for mask in range(1 << n):
        S = [V[i] for i in range(n) if mask & (1 << i)]
        D_corr_table[mask] = D_corr_2site(S, x_out, x_in, beta, mu, t)
        D_vac_table[mask] = D_vac_2site(S, beta, mu, t)
    
    # Build C_S bottom-up by popcount
    C_table = {}
    for k in range(n + 1):
        for mask in range(1 << n):
            if popcount(mask) != k:
                continue
            
            value = D_corr_table[mask]
            
            # Iterate proper subsets of mask via standard submask trick
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


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from cdet_reference.dimer.green_function import G0_2site, G0_2site_at_zero_minus
    from cdet_reference.atom.cdet_recursion import C_V
    
    print("=== cdet_recursion_2site.py sanity checks ===\n")
    
    beta = 5.0
    mu = 0.3
    
    # Test 1: at t=0, all vertices on site 0 -> must match atom CDet
    print("Test 1: At t=0, all vertices on site 0 -> match atom CDet")
    
    x_out = (0, 1.5)
    x_in = (0, 0.5)
    
    for V_atom in [[], [1.0], [1.0, 2.5], [0.8, 2.0, 3.5]]:
        V_2s = [(0, tau) for tau in V_atom]
        c_atom = C_V(V_atom, 1.5, 0.5, beta, mu)
        c_2s = C_V_2site(V_2s, x_out, x_in, beta, mu, t=0.0)
        diff = abs(c_atom - c_2s)
        print(f"  n={len(V_atom)}: C_atom = {c_atom:.12f}, C_2s = {c_2s:.12f}, diff = {diff:.2e}")
    print()
    
    # Test 2: at t=0, vertex on opposite site -> factorizes
    print("Test 2: At t=0, V on opposite site -> C_V = 0 (no connected diagrams cross sites)")
    
    V_oppsite = [(1, 1.0)]
    c_oppsite = C_V_2site(V_oppsite, x_out, x_in, beta, mu, t=0.0)
    print(f"  V=[(1, 1.0)], externals on site 0: C = {c_oppsite:.2e}")
    print("  (Should be ~0 since the vertex at site 1 cannot connect to externals at site 0)")
    print()
    
    # Test 3: at t>0, things differ from atom (sanity that t matters)
    print("Test 3: At t=1, results differ from atom")
    V_2s = [(0, 1.0), (1, 2.5)]
    c_t1 = C_V_2site(V_2s, x_out, x_in, beta, mu, t=1.0)
    print(f"  V=[(0, 1.0), (1, 2.5)], t=1: C = {c_t1:.10f}")
    
    # Compare to a simple case to ensure non-trivial computation
    V_2s_b = [(0, 1.0), (0, 2.5)]
    c_t1_b = C_V_2site(V_2s_b, x_out, x_in, beta, mu, t=1.0)
    print(f"  V=[(0, 1.0), (0, 2.5)], t=1: C = {c_t1_b:.10f}")
    print(f"  Different by: {abs(c_t1 - c_t1_b):.2e}  (yes, vertex sites matter)")
