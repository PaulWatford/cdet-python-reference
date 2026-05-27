"""
Example 01: Compute the free Green's function G_0(i, j; tau) on each lattice.

This is the most basic use of the package. It shows how to call the free
Green's function for the four lattice systems and check a couple of
physical consistency relations.

Run from the repo root:
    python examples/01_compute_G0.py
"""
from cdet_reference.atom.green_function import G0_atom
from cdet_reference.dimer.green_function import G0_2site
from cdet_reference.piflux_square.green_function import G0_4site_piflux
from cdet_reference.hexring.green_function import G0_hexring


def main():
    # Common parameters
    beta = 4.0      # inverse temperature
    mu   = 0.0      # chemical potential
    t    = 1.0      # hopping amplitude
    tau  = 1.5      # imaginary time

    print(f"Free Green's function G_0(i, j; tau={tau}) at beta={beta}, mu={mu}, t={t}\n")

    # Atom: only one site, no spatial indices
    g_atom = G0_atom(tau, beta, mu)
    print(f"  atom:                      G_0(tau) = {g_atom:+.10f}")

    # Dimer: 2 sites, with spatial structure
    g_dimer_diag = G0_2site(0, 0, tau, beta, mu, t)
    g_dimer_off  = G0_2site(0, 1, tau, beta, mu, t)
    print(f"  dimer:    diagonal G_0(0,0; tau) = {g_dimer_diag:+.10f}")
    print(f"            off-diag G_0(0,1; tau) = {g_dimer_off:+.10f}")

    # 4-site pi-flux square
    g_pf_diag    = G0_4site_piflux(0, 0, tau, beta, mu, t)
    g_pf_nn      = G0_4site_piflux(0, 1, tau, beta, mu, t)
    g_pf_diag_pl = G0_4site_piflux(0, 2, tau, beta, mu, t)
    print(f"  pi-flux:  diagonal G_0(0,0; tau) = {g_pf_diag:+.10f}")
    print(f"            nearest  G_0(0,1; tau) = {g_pf_nn:+.10f}")
    print(f"            opposite G_0(0,2; tau) = {g_pf_diag_pl:+.10e}"
          f"  (zero by gauge structure)")

    # 6-site hexring
    g_hex_diag   = G0_hexring(0, 0, tau, beta, mu, t)
    g_hex_d1     = G0_hexring(0, 1, tau, beta, mu, t)
    g_hex_d2     = G0_hexring(0, 2, tau, beta, mu, t)
    g_hex_d3     = G0_hexring(0, 3, tau, beta, mu, t)
    print(f"  hexring:  diagonal G_0(0,0; tau) = {g_hex_diag:+.10f}")
    print(f"            d=1     G_0(0,1; tau) = {g_hex_d1:+.10f}")
    print(f"            d=2     G_0(0,2; tau) = {g_hex_d2:+.10f}")
    print(f"            d=3     G_0(0,3; tau) = {g_hex_d3:+.10f}")
    print()

    # Consistency check 1: at half-filling (mu=0), particle-hole symmetry
    # means G_0(0,0; beta/2) integrates against itself.
    print("Consistency check 1: anti-periodicity G_0(tau - beta) = -G_0(tau)")
    g_pos = G0_hexring(0, 1, tau, beta, mu, t)
    g_neg = G0_hexring(0, 1, tau - beta, beta, mu, t)
    print(f"  G(tau)      = {g_pos:+.10f}")
    print(f"  -G(tau-beta) = {-g_neg:+.10f}")
    print(f"  diff        = {abs(g_pos + g_neg):.2e}")
    print()

    # Consistency check 2: t -> 0 limit reduces to atom Green's function
    print("Consistency check 2: t -> 0 reduces to atom G_0")
    g_hex_at_t0 = G0_hexring(0, 0, tau, beta, mu=0.3, t=0.0)
    g_atom_ref  = G0_atom(tau, beta, mu=0.3)
    print(f"  hexring at t=0, mu=0.3:  G_0(0,0; tau) = {g_hex_at_t0:+.10f}")
    print(f"  atom            mu=0.3:  G_0(tau)      = {g_atom_ref:+.10f}")
    print(f"  diff = {abs(g_hex_at_t0 - g_atom_ref):.2e}")


if __name__ == "__main__":
    main()
