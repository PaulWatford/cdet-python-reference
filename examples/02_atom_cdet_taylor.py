"""
Example 02: Compute a CDet Taylor coefficient and compare to the exact answer.

The CDet algorithm computes the n-th Taylor coefficient of the
interacting Green's function with respect to the on-site interaction U:

    G(tau; U) = G_0(tau) + U * c_1(tau) + U^2 * c_2(tau) + ...

This example computes c_1(tau) on the Hubbard atom in two independent ways:

  (a) by integrating the CDet output: c_1(tau) = int over V of C_V
  (b) by analytic Taylor expansion of the exact (symbolic) Green's function

These two paths use completely different code and should agree to machine
precision. This is the most basic correctness check for the CDet implementation.

Run from the repo root:
    python examples/02_atom_cdet_taylor.py
"""
from scipy.integrate import quad

from cdet_reference.atom.green_function import G0_atom, G_exact_atom
from cdet_reference.atom.cdet_recursion import C_V


def cdet_c1(tau, beta, mu):
    """First Taylor coefficient c_1(tau) via CDet integration over the single
    vertex time tau_1 in [0, beta].
    """
    def integrand(tau_1):
        # V is a list of vertex times. For the atom there is no site index.
        V = [tau_1]
        return C_V(V, tau_out=tau, tau_in=0.0, beta=beta, mu=mu)

    val, _ = quad(integrand, 0.0, beta)
    return val


def exact_c1_via_finite_diff(tau, beta, mu, dU=1e-5):
    """Reference: c_1 from finite-difference of the exact (symbolic) Green's
    function at U = 0.

        c_1(tau) = d G(tau; U) / dU  at U = 0
    """
    g_plus  = G_exact_atom(tau, beta, mu,  dU)
    g_minus = G_exact_atom(tau, beta, mu, -dU)
    return float((g_plus - g_minus) / (2 * dU))


def main():
    beta = 4.0
    mu   = 0.2

    print(f"CDet c_1(tau) on the atom, beta={beta}, mu={mu}\n")
    print(f"{'tau':>6}  {'c_1 via CDet':>16}  {'c_1 via finite-diff':>20}  {'rel diff':>10}")
    print("-" * 60)

    for tau in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        c1_cdet = cdet_c1(tau, beta, mu)
        c1_ref  = exact_c1_via_finite_diff(tau, beta, mu)
        scale   = max(abs(c1_cdet), abs(c1_ref), 1e-15)
        rel     = abs(c1_cdet - c1_ref) / scale
        print(f"{tau:>6.2f}  {c1_cdet:>+16.10f}  {c1_ref:>+20.10f}  {rel:>10.2e}")

    print()
    print("Both columns should agree to machine precision (around 1e-8 to 1e-10")
    print("limited by the finite-difference step dU = 1e-5).")


if __name__ == "__main__":
    main()
