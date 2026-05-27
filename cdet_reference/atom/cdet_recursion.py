"""
The Rossi connected-determinant recursion.

Implements:
    C_V(x_out, x_in) = D_V(x_out, x_in) - sum_{S subsetneq V} C_S(x_out, x_in) * D_{V\\S}(empty)

This sums all CONNECTED Feynman diagrams at fixed vertex set V by
recursively subtracting disconnected contributions from D_V.

For the Hubbard atom, vertices are just imaginary times.

Algorithm:
  1. Enumerate all 2^|V| subsets of V (indexed by bitmask).
  2. Precompute D_S(x_out, x_in) and D_S(empty) for every subset S.
  3. Iterate subsets by ascending |S| (popcount). For each S, compute
     C_S using the recursion.
  4. Return C_V for the full set.

Complexity:
  - Storage: O(2^n)
  - Naive submask loop: O(4^n)  [used below for simplicity]
  - Optimal submask iteration: O(3^n)

For n <= ~12 the naive version is fine. The submask-iteration optimization
is straightforward to add later if needed.

See cdet_derivation.md for the full derivation.

"""
import sys
from cdet_reference.atom.wick_determinant import D_corr, D_vac


def popcount(x):
    """Count set bits in x. (For Python 3.10+ this is bin(x).count('1') or x.bit_count().)"""
    return bin(x).count('1')


def C_V(V, tau_out, tau_in, beta, mu):
    """Connected correlator C_V(x_out, x_in) via the CDet recursion.
    
    Args:
      V:        list of internal vertex times (length n; can be empty)
      tau_out:  external time of the c_up annihilation
      tau_in:   external time of the c_up+ creation
      beta, mu: temperature inverse and chemical potential
    
    Returns:
      C_V, the connected-diagrams sum at vertex set V.
    
    For n=0: returns G_0(tau_out - tau_in).
    For n>0: applies the Rossi recursion using D_corr and D_vac from
    wick_determinant.py.
    """
    V = list(V)
    n = len(V)
    
    if n == 0:
        # C_empty = D_empty = G_0(tau_out - tau_in)
        return D_corr([], tau_out, tau_in, beta, mu)
    
    # Precompute D values for every subset of V.
    # Subset indexed by bitmask: bit i set <-> V[i] is in the subset.
    D_corr_table = {}
    D_vac_table = {}
    
    for mask in range(1 << n):
        S = [V[i] for i in range(n) if mask & (1 << i)]
        D_corr_table[mask] = D_corr(S, tau_out, tau_in, beta, mu)
        D_vac_table[mask] = D_vac(S, beta, mu)
    
    # Compute C_S for every subset, bottom-up by popcount.
    C_table = {}
    
    for k in range(n + 1):  # process all subsets of size k
        for mask in range(1 << n):
            if popcount(mask) != k:
                continue
            
            # C_S = D_S(x_out, x_in) - sum_{S' subsetneq S} C_{S'} * D_{S\S'}(empty)
            value = D_corr_table[mask]
            
            # Iterate proper subsets S' of S (i.e., submasks of mask, excluding mask itself)
            submask = mask
            # Subtract the C_{S'} * D_{S\S'}(empty) terms for all S' subsetneq S
            # Iterate via the standard submask trick:
            #   submask = (submask - 1) & mask, visit all submasks in decreasing order
            #   plus include submask = 0 (empty set) at the end
            sm = (mask - 1) & mask
            while True:
                if sm != mask:  # exclude S' = S itself (we want proper subsets)
                    complement = mask ^ sm  # mask \ sm  ==  set difference as bitmask
                    value -= C_table[sm] * D_vac_table[complement]
                if sm == 0:
                    break
                sm = (sm - 1) & mask
            
            C_table[mask] = value
    
    # Return C for the full set V (mask with all bits set)
    full_mask = (1 << n) - 1
    return C_table[full_mask]


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from cdet_reference.atom.green_function import G0_atom, G0_atom_at_zero_minus
    
    print("=== cdet_recursion.py basic sanity checks ===\n")
    
    beta = 5.0
    mu = 0.3
    n_0 = G0_atom_at_zero_minus(beta, mu)
    
    # n=0
    tau_out, tau_in = 1.5, 0.5
    c0 = C_V([], tau_out, tau_in, beta, mu)
    g0 = G0_atom(tau_out - tau_in, beta, mu)
    print(f"n=0: C_empty = {c0:.10f}, G_0 = {g0:.10f}, diff = {abs(c0 - g0):.2e}")
    
    # n=1
    tau_1 = 1.0
    c1 = C_V([tau_1], tau_out, tau_in, beta, mu)
    g_o1 = G0_atom(tau_out - tau_1, beta, mu)
    g_1i = G0_atom(tau_1 - tau_in, beta, mu)
    expected_c1 = n_0 * g_o1 * g_1i
    print(f"n=1: C_tau1 = {c1:.10f}")
    print(f"     expected (n_0*G_0*G_0) = {expected_c1:.10f}")
    print(f"     diff = {abs(c1 - expected_c1):.2e}")
    
    # n=2: quick sanity (will be deeply audited separately)
    tau_2 = 2.5
    c2 = C_V([tau_1, tau_2], tau_out, tau_in, beta, mu)
    print(f"\nn=2: C_{{tau_1,tau_2}} = {c2:.10f}")
    print(f"     (audit will check this against Taylor coefficient via integration)")
    
    # n=3 timing check
    import time
    t0 = time.time()
    c3 = C_V([0.5, 1.5, 3.0], tau_out, tau_in, beta, mu)
    t1 = time.time()
    print(f"\nn=3: C = {c3:.10f}, computed in {(t1-t0)*1000:.2f} ms")
    
    # n=8 timing check
    V8 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    t0 = time.time()
    c8 = C_V(V8, tau_out, tau_in, beta, mu)
    t1 = time.time()
    print(f"n=8: C = {c8:.10e}, computed in {(t1-t0)*1000:.2f} ms")
