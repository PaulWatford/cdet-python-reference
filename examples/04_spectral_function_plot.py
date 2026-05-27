"""
Example 04: Compute and plot the single-particle spectral function
A(k, omega) on the 6-site hexring at finite Hubbard U.

The spectral function tells you how electrons propagate in the system:

    A(k, omega) = -(1/pi) Im G_R(k, omega + i*eta)

where G_R is the retarded Green's function. Peaks in A(k, omega) at a
given momentum k mark the energies at which the system has well-defined
quasiparticle states for that momentum.

We construct A(k, omega) from the Lehmann representation, using all 4096
many-body eigenstates of the interacting Hamiltonian. This is exact on
the small cluster (no analytic continuation needed) and makes a useful
benchmark for any DiagMC code that produces spectral functions.

Output: PNG file 'A_kw_hexring.png' in the current directory.

Run from the repo root:
    python examples/04_spectral_function_plot.py
"""
import numpy as np
import matplotlib.pyplot as plt

from cdet_reference.hexring.exact_diagonalization import (
    build_hexring_operators, build_H_hexring,
)
from cdet_reference.spectral_function.spectral_function import spectral_function_exact


def main():
    # Parameters
    beta = 8.0      # low T so peaks are sharp
    mu   = 0.0      # half-filling
    t    = 1.0
    U    = 1.0      # moderate interaction
    eta  = 0.05     # Lorentzian broadening of the delta peaks

    print(f"Computing A(k, omega) on the 6-site hexring")
    print(f"  beta = {beta}, mu = {mu}, t = {t}, U = {U}, eta = {eta}\n")

    # Build the many-body Hamiltonian and diagonalise (~15 seconds)
    print("Building H (~15 seconds) and diagonalising 4096-dim Hilbert space "
          "(~10 seconds)...")
    ops = build_hexring_operators()
    H = build_H_hexring(beta, mu, t, U, ops=ops).toarray()
    eigvals, eigvecs = np.linalg.eigh(H)
    print(f"  ground-state energy E_0 = {eigvals[0]:.6f}\n")

    # Sample omega range
    omega = np.linspace(-4, 4, 401)

    # Compute A(k, omega) for the 4 distinct band momenta
    # k = 0 (Gamma), k = 1 (one Dirac valley), k = 2, k = 3 (M point)
    print("Computing A(k, omega) for the 4 distinct band momenta...")
    A_by_k = {}
    for k in [0, 1, 2, 3]:
        A_by_k[k] = spectral_function_exact(
            k, omega, beta, mu, t, U,
            spin='up', eta=eta,
            eigh_cache=(eigvals, eigvecs), ops=ops,
        )
        # Quick sum-rule check
        integral = np.sum(A_by_k[k]) * (omega[1] - omega[0])
        print(f"  k={k}:  integral A(k, omega) d_omega = {integral:.4f}  "
              f"(sum rule = 1.0, slight loss to broadening tails)")
    print()

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    band_labels = {0: "Gamma  (k=0)", 1: "K   (k=1)", 2: "K'  (k=2)", 3: "M   (k=3)"}
    for ax, k in zip(axes.flat, [0, 1, 2, 3]):
        ax.plot(omega, A_by_k[k], linewidth=1.5)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_title(f"A(k, omega) at {band_labels[k]}")
        ax.set_xlabel("omega / t")
        ax.set_ylabel("A(k, omega)")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Single-particle spectral function on the 6-site hexring\n"
                 f"beta={beta}, mu={mu}, t={t}, U={U}",
                 fontsize=11)
    fig.tight_layout()

    out_path = "A_kw_hexring.png"
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    print()
    print("Physics to look for in the plot:")
    print("  - At U=0, A(k, omega) would be a single sharp peak at the band energy.")
    print("  - At U>0, the peak broadens and shifts, and additional peaks (Hubbard")
    print("    satellites) appear at higher and lower frequencies.")
    print("  - The K and K' valleys (k=1 and k=2) should be qualitatively similar")
    print("    by particle-hole symmetry at mu = 0.")


if __name__ == "__main__":
    main()
