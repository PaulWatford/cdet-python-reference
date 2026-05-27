"""
Rossi CDet recursion for the 6-site hexagonal ring Hubbard model.

The recursion is site-agnostic (operates on bitmask subsets of V), so
the implementation is shared with the smaller-lattice cases.
"""
import sys
from cdet_reference.hexring.wick_determinant import D_corr_hexring, D_vac_hexring


def popcount(x):
    return bin(x).count('1')


def C_V_hexring(V, x_out, x_in, beta, mu, t):
    """Connected correlator at V for the 6-site hexagonal ring.
    
    V: list of (site, tau) with site in {0..5}, tau in (0, beta)
    x_out, x_in: (site, tau) tuples
    """
    V = list(V)
    n = len(V)
    
    if n == 0:
        return D_corr_hexring([], x_out, x_in, beta, mu, t)
    
    D_corr_table = {}
    D_vac_table = {}
    for mask in range(1 << n):
        S = [V[i] for i in range(n) if mask & (1 << i)]
        D_corr_table[mask] = D_corr_hexring(S, x_out, x_in, beta, mu, t)
        D_vac_table[mask] = D_vac_hexring(S, beta, mu, t)
    
    C_table = {}
    for k in range(n + 1):
        for mask in range(1 << n):
            if popcount(mask) != k:
                continue
            value = D_corr_table[mask]
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


# Sanity checks
if __name__ == "__main__":
    from cdet_reference.atom.cdet_recursion import C_V
    
    print("=" * 72)
    print("  CDet recursion on 6-site hexring, sanity checks")
    print("=" * 72)
    print()
    
    beta, mu = 3.0, 0.3
    
    # Test 1: t=0 + all vertices at site 0 -> matches atom CDet
    print("Test 1: t=0 + all vertices on site 0 -> matches atom CDet")
    x_out_h = (0, 1.5)
    x_in_h = (0, 0.5)
    
    for V_atom in [[], [1.0], [1.0, 2.0], [0.7, 1.5, 2.3]]:
        V_h = [(0, tau) for tau in V_atom]
        c_a = C_V(V_atom, 1.5, 0.5, beta, mu)
        c_h = C_V_hexring(V_h, x_out_h, x_in_h, beta, mu, t=0.0)
        diff = abs(c_a - c_h)
        print(f"  n={len(V_atom)}: C_atom={c_a:+.12f}, C_hex={c_h:+.12f}, diff={diff:.2e}")
    print()
    
    # Test 2: t=0, vertex on isolated site -> C_V = 0
    print("Test 2: t=0, vertex on isolated site -> C_V = 0")
    V_iso = [(3, 1.0)]
    c_iso = C_V_hexring(V_iso, x_out_h, x_in_h, beta, mu, t=0.0)
    print(f"  V=[(3, 1.0)] at t=0: C = {c_iso:.2e}")
    print()
    
    # Test 3: At t=1, finite hopping
    print("Test 3: At t=1, hopping changes C_V")
    V = [(0, 1.0)]
    c_t0 = C_V_hexring(V, x_out_h, x_in_h, beta, mu, t=0.0)
    c_t1 = C_V_hexring(V, x_out_h, x_in_h, beta, mu, t=1.0)
    print(f"  t=0: C={c_t0:+.10f}")
    print(f"  t=1: C={c_t1:+.10f}")
    print()
    
    # Test 4: Permutation invariance
    print("Test 4: C_V invariant under vertex ordering")
    V_a = [(0, 0.7), (2, 1.5), (4, 2.3)]
    V_b = [(2, 1.5), (0, 0.7), (4, 2.3)]
    V_c = [(4, 2.3), (0, 0.7), (2, 1.5)]
    c_a = C_V_hexring(V_a, x_out_h, x_in_h, beta, mu, t=1.0)
    c_b = C_V_hexring(V_b, x_out_h, x_in_h, beta, mu, t=1.0)
    c_c = C_V_hexring(V_c, x_out_h, x_in_h, beta, mu, t=1.0)
    print(f"  V_a: C = {c_a:+.12f}")
    print(f"  V_b: C = {c_b:+.12f}")
    print(f"  V_c: C = {c_c:+.12f}")
    print(f"  Max diff: {max(abs(c_a-c_b), abs(c_b-c_c), abs(c_a-c_c)):.2e}")
    print()
    
    # Test 5: Z_6 rotational symmetry
    print("Test 5: Z_6 rotational symmetry of CDet output")
    print("  C_V(externals=(0,...), V_at_site_k) should equal")
    print("  C_V(externals=(j,...), V_at_site_(k+j) mod 6)")
    print()
    
    # Pick a single-vertex test
    t_v = 1.0
    x_in_at_0 = (0, 0.5)
    x_out_at_0 = (0, 1.5)
    V_at_1 = [(1, 1.0)]
    c_ref = C_V_hexring(V_at_1, x_out_at_0, x_in_at_0, beta, mu, t_v)
    print(f"  Reference: externals (0,0.5)->(0,1.5), V=[(1, 1.0)]: C = {c_ref:+.10f}")
    
    for j in range(1, 6):
        # Translate everything by j sites: (i, tau) -> (i+j mod 6, tau)
        x_out_rot = ((0 + j) % 6, 1.5)
        x_in_rot = ((0 + j) % 6, 0.5)
        V_rot = [((1 + j) % 6, 1.0)]
        c_rot = C_V_hexring(V_rot, x_out_rot, x_in_rot, beta, mu, t_v)
        diff = abs(c_rot - c_ref)
        print(f"  Rotation by {j}: externals ({j},0.5)->({j},1.5), V=[({(1+j)%6}, 1.0)]: C = {c_rot:+.10f} (diff {diff:.2e})")
