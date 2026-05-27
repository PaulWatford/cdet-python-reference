"""
Example 03: Benchmark interacting Green's function: CDet vs exact ED.

This shows how to use the package as a benchmark for any production CDet
implementation. The pattern is:

    1. Compute the exact interacting Green's function G(tau; U) via
       full diagonalisation of the many-body Hamiltonian.
    2. Compute the same quantity perturbatively as a Taylor series in U
       using CDet output up to order n.
    3. Compare. The agreement at small U validates the CDet code; the
       breakdown at large U tells you where you need higher orders.

We do this on the 6-site hexring at order n=0 (just the free Green's
function) for a quick illustration. The full audit suite in tests/
extends this to n=1, 2 at the same precision.

Run from the repo root:
    python examples/03_benchmark_vs_ED.py
"""
import numpy as np

from cdet_reference.hexring.green_function import G0_hexring
from cdet_reference.hexring.exact_diagonalization import (
    build_hexring_operators, build_H_hexring, G_exact_hexring,
)


def main():
    # Parameters
    beta = 4.0
    mu   = 0.0
    t    = 1.0

    # Site pair and time to evaluate at
    i, j = 0, 1
    tau  = 1.5

    print(f"Benchmarking interacting G(i={i}, j={j}; tau={tau}) on the 6-site hexring")
    print(f"  beta = {beta}, mu = {mu}, t = {t}\n")

    # Build the many-body operators once (~12 seconds; 4096-dim Hilbert space)
    print("Building 4096-dim many-body operators (~12 seconds)...")
    ops = build_hexring_operators()
    print("done.\n")

    # Free Green's function reference (n=0 CDet output, i.e., U=0)
    g_free = G0_hexring(i, j, tau, beta, mu, t)

    # Compare to exact ED at successively larger U values
    print(f"{'U':>6}  {'G_exact(U) via ED':>20}  {'G_0 (free)':>14}  "
          f"{'difference':>14}")
    print("-" * 70)
    for U in [0.0, 0.1, 0.5, 1.0, 2.0, 4.0]:
        # Build the many-body Hamiltonian for this U and diagonalise
        H = build_H_hexring(beta, mu, t, U, ops=ops).toarray()
        eigvals, eigvecs = np.linalg.eigh(H)

        g_exact = G_exact_hexring(
            i, j, tau, beta, mu, t, U,
            spin='up', ops=ops, eigh_cache=(eigvals, eigvecs),
        )
        diff = g_exact - g_free
        print(f"{U:>6.2f}  {g_exact:>+20.10f}  {g_free:>+14.10f}  {diff:>+14.6e}")

    print()
    print("At U = 0 the two columns should match exactly (within FP rounding).")
    print("As U grows the difference reveals the perturbative correction that")
    print("CDet at orders n>=1 captures. The full audit suite in tests/")
    print("verifies this term by term.")


if __name__ == "__main__":
    main()
