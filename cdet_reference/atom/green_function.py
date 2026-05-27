"""
Hubbard atom Green's functions.

The Hubbard atom is a single site with two spin species:

    H = U * n_up * n_down - mu * (n_up + n_down)

We need three things:
  1. G0_atom(tau, beta, mu): free Green's function (U=0)
  2. n_F(xi, beta): Fermi function (for G0(0-) = <n>)
  3. G_exact_atom(tau, beta, mu, U): exact interacting Green's function

Time convention: tau in (-beta, beta), antiperiodic G(tau + beta) = -G(tau).
We define G(tau) = -<T_tau c(tau) c+(0)>.

"""
import numpy as np
import math


def n_F(xi, beta):
    """Fermi function n_F(xi) = 1 / (exp(beta * xi) + 1).
    
    Stable for any sign of xi.
    """
    # Avoid overflow for large positive beta*xi: write n_F(xi) = exp(-bx) / (1 + exp(-bx)) when bx>0
    bx = beta * xi
    if bx > 0:
        e = math.exp(-bx)
        return e / (1.0 + e)
    else:
        return 1.0 / (1.0 + math.exp(bx))


def G0_atom(tau, beta, mu):
    """Non-interacting Hubbard atom Green's function.
    
    H_0 = -mu * n  =>  single-particle energy xi = -mu.
    
        G_0(tau) = -<T_tau c(tau) c+(0)>_0
    
    For 0 < tau < beta:   G_0(tau) = -(1 - n_F(xi)) * exp(-xi * tau)
    For -beta < tau < 0:  G_0(tau) = +n_F(xi) * exp(-xi * tau)
    
    Antiperiodic: G_0(tau + beta) = -G_0(tau).
    """
    xi = -mu
    
    # Fold tau into (-beta, beta] using antiperiodicity
    # First reduce mod 2*beta then handle sign
    while tau >  beta: tau -= 2*beta
    while tau <= -beta: tau += 2*beta
    
    if tau > 0:
        return -(1.0 - n_F(xi, beta)) * math.exp(-xi * tau)
    elif tau < 0:
        return +n_F(xi, beta) * math.exp(-xi * tau)
    else:
        # tau = 0 exactly: by convention, use tau = 0^- (i.e., +n_F)
        # This is the value that appears in the Hartree term G_0(0^-) = <n>
        return n_F(xi, beta)


def G0_atom_at_zero_minus(beta, mu):
    """G_0(tau = 0^-) = n_F(-mu) = <n>_0, the non-interacting density per spin."""
    return n_F(-mu, beta)


def G_exact_atom(tau, beta, mu, U):
    """Exact interacting Green's function of the Hubbard atom in imaginary time.
    
    The Hilbert space is 4-dimensional: |0>, |up>, |dn>, |up,dn>.
    Energies: E_0 = 0, E_up = E_dn = -mu, E_2 = U - 2*mu.
    Partition function: Z = 1 + 2*exp(beta*mu) + exp(beta*(2*mu - U)).
    
    For spin up:  G_up(tau) = -<T_tau c_up(tau) c_up+(0)>
    
    Lehmann sum, for 0 < tau < beta:
        G_up(tau) = -(1/Z) sum_{m,n} |<m|c_up|n>|^2 exp(-(beta-tau)*E_m - tau*E_n)
    
    Nonzero matrix elements:
        c_up |up> = |0>       -> (m=0, n=up)
        c_up |up,dn> = -|dn>  -> (m=dn, n=2)
    
    Term 1 (m=0, n=up):    E_m=0, E_n=-mu
        exp(-(beta-tau)*0 - tau*(-mu)) = exp(mu*tau)
    
    Term 2 (m=dn, n=2):    E_m=-mu, E_n=U-2*mu
        exp(-(beta-tau)*(-mu) - tau*(U-2*mu))
        = exp(mu*(beta-tau) - tau*(U-2*mu))
        = exp(mu*beta - mu*tau - tau*U + 2*mu*tau)
        = exp(mu*beta + mu*tau - tau*U)
        = exp(mu*(beta+tau) - tau*U)
    
    So:  G_up(tau) = -(1/Z) * [exp(mu*tau) + exp(mu*(beta+tau) - tau*U)]
    
    At U=0:  G_up = -(1/(1+exp(beta*mu))^2) * exp(mu*tau) * (1 + exp(beta*mu))
                  = -exp(mu*tau) / (1 + exp(beta*mu))
    which equals G_0(tau) = -(1 - n_F(-mu)) * exp(mu*tau).  (checks out)
    """
    # Fold tau into [0, beta) for the formula; handle other ranges by antiperiodicity.
    sign = 1.0
    while tau >= beta:
        tau -= beta
        sign *= -1.0
    while tau < 0:
        tau += beta
        sign *= -1.0
    # Now 0 <= tau < beta.
    
    # Partition function
    Z = 1.0 + 2.0 * math.exp(beta * mu) + math.exp(beta * (2.0*mu - U))
    
    # Two Lehmann terms
    term1 = math.exp(mu * tau)
    term2 = math.exp(mu * (beta + tau) - tau * U)
    
    return sign * (-(term1 + term2) / Z)


def density_exact(beta, mu, U):
    """Exact density per spin <n_sigma> for the Hubbard atom.
    
    <n_up> = (1/Z) * [exp(beta*mu) + exp(beta*(2*mu - U))]
    """
    Z = 1.0 + 2.0 * math.exp(beta * mu) + math.exp(beta * (2.0*mu - U))
    return (math.exp(beta * mu) + math.exp(beta * (2.0*mu - U))) / Z


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Hubbard atom Green's function sanity checks ===\n")
    
    beta = 5.0
    mu = 0.3
    U = 0.0  # at U=0, exact should equal free
    
    print(f"Parameters: beta = {beta}, mu = {mu}, U = {U}")
    print()
    
    # Test 1: G_exact at U=0 should equal G_0
    print("Test 1: G_exact at U=0 vs G_0_atom")
    print(f"{'tau':>6} {'G_0':>14} {'G_exact (U=0)':>16} {'diff':>10}")
    for tau in [0.1, 0.5, 1.0, 2.5, 4.5]:
        g0 = G0_atom(tau, beta, mu)
        gx = G_exact_atom(tau, beta, mu, U=0.0)
        print(f"{tau:>6.2f} {g0:>14.10f} {gx:>16.10f} {abs(g0-gx):>10.2e}")
    print()
    
    # Test 2: antiperiodicity G(tau - beta) = -G(tau) for 0 < tau < beta
    print("Test 2: Antiperiodicity G(tau - beta) = -G(tau)")
    print(f"{'tau':>6} {'G(tau)':>14} {'-G(tau-beta)':>14} {'diff':>10}")
    for tau in [0.5, 2.0, 3.7]:
        g_pos = G0_atom(tau, beta, mu)
        g_neg = G0_atom(tau - beta, beta, mu)
        print(f"{tau:>6.2f} {g_pos:>14.10f} {-g_neg:>14.10f} {abs(g_pos + g_neg):>10.2e}")
    print()
    
    # Test 3: G_0(0^-) = <n>_0
    print("Test 3: G_0(0^-) = <n>_0 = n_F(-mu)")
    val = G0_atom_at_zero_minus(beta, mu)
    expected = n_F(-mu, beta)
    print(f"  G_0(0^-) = {val:.10f}")
    print(f"  n_F(-mu) = {expected:.10f}")
    print(f"  diff     = {abs(val - expected):.2e}")
    print()
    
    # Test 4: density expansion. <n_sigma>_exact at small U should match <n>_0 + O(U)
    print("Test 4: Exact density vs free density at small U")
    n0 = density_exact(beta, mu, U=0.0)
    n_free = n_F(-mu, beta)
    print(f"  <n>_exact (U=0) = {n0:.10f}")
    print(f"  <n>_0 (free)    = {n_free:.10f}")
    print(f"  diff            = {abs(n0 - n_free):.2e}")
    
    print()
    print("Now with U > 0:")
    for U_val in [0.5, 1.0, 2.0]:
        n_int = density_exact(beta, mu, U=U_val)
        print(f"  U = {U_val:.2f}: <n>_exact = {n_int:.10f}")
