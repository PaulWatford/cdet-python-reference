# CDet recursion: derivation

Implementation in `cdet_recursion.py`. See `wick_derivation.md` for the
underlying $D_V$ definition and the sign convention $(-1)^{|V|}$.

---


## The formula (Moutenet Eq. 2 / Rossi 2017)

  C_V(x_out, x_in) = D_V(x_out, x_in) - sum_{S subsetneq V} C_S(x_out, x_in) * D_{V\S}(empty)

where:
  - V is a set of internal interaction vertex positions (in our case,
    just imaginary times for the Hubbard atom)
  - C_V is the sum of CONNECTED diagrams at vertex set V
  - D_V is the sum of ALL diagrams (connected + disconnected) at V
  - D_V(empty) is the vacuum diagram sum (no external operators) at V
  - The sum S subsetneq V is over PROPER subsets (excludes S = V)

The recursion expresses each C_V in terms of:
  - Its own D_V (just computed via wick_determinant.py)
  - Earlier-computed C_S for S strictly smaller than V
  - D values on V\S which are also known

## Why this works algorithmically

The recursion has a natural ordering: compute C_S for S = {} first
(C_empty = D_empty), then |S| = 1, then |S| = 2, ..., then |S| = |V|. At each
step the right-hand side uses only already-computed quantities.

Computational complexity (Moutenet section II):
  - For a single V of size n: enumerate 2^n subsets of V, for each one
    a sum of 2^|S| terms, gives 3^n total operations after summing
    binomial coefficients.
  - For PRECOMPUTING D_S over all subsets: 2^n * O(n^3) (one determinant
    per subset).
  - Net: O(3^n + 2^n * n^3), dominated by 3^n at large n.

This is the famous "n! -> 2^n then 3^n" reduction.

## Data structure

Subsets of V indexed by integers 0..2^n-1 using a bitmask:
  - Bit i set <-> tau_i is in the subset
  - bitmask = 0 <-> S = {}
  - bitmask = 2^n - 1 <-> S = V
  - popcount(bitmask) = |S|

For each subset S, store:
  - D_S(x_out, x_in)   (correlator D)
  - D_S(empty)              (vacuum D)
  - C_S(x_out, x_in)   (connected, computed by the recursion)

In Python: a dict from bitmask -> tuple (D_corr, D_vac, C).

## Recursion order

Process bitmasks by popcount, ascending:
  - popcount 0: S = empty. C_empty = D_empty (no proper subsets to subtract).
  - popcount 1: 1-vertex sets. C = D - C_empty * D_{V\S}(empty).
  - popcount 2, 3, ...: each handled when all smaller-popcount C's exist.

## How to iterate proper subsets of V efficiently

Given V represented by bitmask v, iterate S subsetneq v:
  - In Python: iterate s from 0 to v, keep those with (s & v) == s and s != v.
  - More efficient using "submask iteration":
    s = (s - 1) & v   starting from s = (v - 1) & v
    visits all proper non-empty subsets in decreasing order
    plus s = 0 separately
  - For our purposes the explicit "loop and check" is plenty fast.

## Sanity check at n=1

V = {tau_1}. Bitmasks 0, 1.
  S = 0 (empty): C_empty = D_empty(x_out, x_in) = G_0(tau_out - tau_in).
  S = 1 (V): C_V = D_V - sum_{S' subsetneq V} C_{S'} * D_{V\S'}(empty)
                = D_V - C_empty * D_{V}(empty)
                = D_{tau_1}(x_out, x_in) - G_0 * D_{tau_1}(empty)
                = -n_0^2 G_0 + n_0 G_0(out-1) G_0(1-in) - G_0 * (-n_0^2)
                = +n_0 G_0(out-1) G_0(1-in)

Matches the verified result.

## Sanity check at n=2

V = {tau_1, tau_2}. Bitmasks 0, 1, 2, 3.
  S = 0 (empty): C_empty = D_empty = G_0(out-in)
  S = 1 ({tau_1}): C_{tau_1} = D_{tau_1} - C_empty * D_{tau_1}(empty), same as n=1 above
  S = 2 ({tau_2}): C_{tau_2} = D_{tau_2} - C_empty * D_{tau_2}(empty), same form
  S = 3 (V): C_V = D_V - sum_{S' subsetneq V} C_{S'} * D_{V\S'}(empty)
              = D_V - C_empty * D_V(empty)
                    - C_{tau_1} * D_{tau_2}(empty)
                    - C_{tau_2} * D_{tau_1}(empty)

Three terms in the sum (proper subsets: {}, {tau_1}, {tau_2}, but NOT V).

The Wick-determinant audit (`tests/test_atom_wick_determinant.py`) does this
manually and the result matches the symbolic Taylor coefficients to 5e-15,
confirming the recursion.

## What the code should look like

```python
def C_V(V, tau_out, tau_in, beta, mu):
    """Compute C_V(x_out, x_in), connected diagrams sum at vertex set V."""
    n = len(V)
    # Map: bitmask -> C value (we only need C_S(x_out, x_in))
    # Need D_S(x_out, x_in) and D_S(empty) for all S (precompute)
    
    D_corr_table = {}
    D_vac_table = {}
    C_table = {}
    
    # Precompute D values for all subsets
    for mask in range(2**n):
        S = [V[i] for i in range(n) if mask & (1 << i)]
        D_corr_table[mask] = D_corr(S, tau_out, tau_in, beta, mu)
        D_vac_table[mask] = D_vac(S, beta, mu)
    
    # Build C bottom-up by popcount
    for k in range(n + 1):
        for mask in range(2**n):
            if bin(mask).count('1') != k:
                continue
            # C_S = D_S - sum over proper subsets S' of S
            C = D_corr_table[mask]
            for submask in range(mask):
                if (submask & mask) == submask:  # submask subseteq mask
                    complement = mask ^ submask  # mask \ submask
                    C -= C_table[submask] * D_vac_table[complement]
            C_table[mask] = C
    
    return C_table[(1 << n) - 1]  # full set V
```

Two issues to think about:

1. Iteration efficiency. The "for submask in range(mask)" approach is
   O(4^n) overall (n levels x 2^n masks each x 2^n submasks). We want
   3^n. Fix: use the submask iteration idiom that only visits actual
   submasks: s = (s-1) & mask. But for n <= ~15 the inefficient version
   is fine; n=10 takes 10^6 ops, n=15 takes 10^9. We'll iterate later
   if needed.

2. The recursion stores tables that are O(2^n) entries. Same scaling.

We iterate naively. The submask trick is an option if higher orders
(beyond $n = 10$-$12$) become a target.

## Independent verification path

1. Boundary case n=0: C_empty = D_empty = G_0(tau_out - tau_in). Verify.
2. n=1: C_{tau_1} = n_0 * G_0(out-1) * G_0(1-in). Verify.
3. n=2: Manual recursion result matches the Wick-determinant audit
   (`tests/test_atom_wick_determinant.py`).
4. End-to-end at multiple n: integrate C_V over V at orders n=1, 2, 3
   (where computationally feasible) and compare against symbolic Taylor
   coefficients of G_exact_atom.

The end-to-end test at n=3 will be slow (triple integral via scipy.tplquad)
but doable for one or two parameter sets. This pushes the verification
deeper than the wick_determinant audit went.

5. Reuse: rerun the audit_wick_determinant Check F using cdet_recursion
   instead of the manual inline recursion. Same numerical result confirms
   the recursion code is equivalent to the hand-inlined version.
