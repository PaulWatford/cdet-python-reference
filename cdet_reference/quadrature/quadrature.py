"""
Gauss-Legendre fixed-grid quadrature for CDet integrals.

For smooth integrands (which these are, products of exponentials), 
Gauss-Legendre converges exponentially in the number of grid points.
A 16-point grid on each axis gives near-machine precision on smooth
functions, with deterministic cost.

For n=3, total integrand evaluations = 16^3 = 4096 per site config.
For 8 site configs at n=3: 32,768 evaluations total. Each takes maybe
1 ms -> total ~33 seconds for n=3, manageable.

"""
import numpy as np


def gl_grid_1d(n_points, a, b):
    """Return Gauss-Legendre nodes and weights on [a, b]."""
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    # Map from [-1, 1] to [a, b]
    nodes_mapped = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    weights_mapped = 0.5 * (b - a) * weights
    return nodes_mapped, weights_mapped


def integrate_1d(f, n_points, a, b):
    """integral_a^b f(x) dx via n-point Gauss-Legendre."""
    nodes, weights = gl_grid_1d(n_points, a, b)
    total = 0.0
    for i, x in enumerate(nodes):
        total += weights[i] * f(x)
    return total


def integrate_2d(f, n_points, a, b):
    """int int_{[a,b]^2} f(x, y) dx dy via Gauss-Legendre."""
    nodes, weights = gl_grid_1d(n_points, a, b)
    total = 0.0
    for i, xi in enumerate(nodes):
        for j, xj in enumerate(nodes):
            total += weights[i] * weights[j] * f(xi, xj)
    return total


def integrate_3d(f, n_points, a, b):
    """int intintegral_{[a,b]^3} f(x, y, z) dx dy dz via Gauss-Legendre."""
    nodes, weights = gl_grid_1d(n_points, a, b)
    total = 0.0
    for i, xi in enumerate(nodes):
        for j, xj in enumerate(nodes):
            for k, xk in enumerate(nodes):
                total += weights[i] * weights[j] * weights[k] * f(xi, xj, xk)
    return total


def integrate_4d(f, n_points, a, b):
    """int intint int_{[a,b]^4} f via Gauss-Legendre."""
    nodes, weights = gl_grid_1d(n_points, a, b)
    total = 0.0
    for i, xi in enumerate(nodes):
        for j, xj in enumerate(nodes):
            for k, xk in enumerate(nodes):
                for l, xl in enumerate(nodes):
                    total += (weights[i] * weights[j] * weights[k] * weights[l] 
                              * f(xi, xj, xk, xl))
    return total


if __name__ == "__main__":
    # Test: integrate x^2 from 0 to 1, exact = 1/3
    f = lambda x: x**2
    for n in [4, 8, 16]:
        v = integrate_1d(f, n, 0, 1)
        print(f"  1D, n={n}: integral = {v}, error = {abs(v - 1/3):.2e}")
    
    # Test: int int x*y dx dy from 0 to 1, exact = 1/4
    print()
    g = lambda x, y: x * y
    for n in [4, 8, 16]:
        v = integrate_2d(g, n, 0, 1)
        print(f"  2D, n={n}: integral = {v}, error = {abs(v - 0.25):.2e}")
    
    # Test: convergence on a function similar to what we'll see
    import math
    h = lambda x, y, z: math.exp(-x - y - z) * math.sin(x*y*z)
    print()
    for n in [4, 8, 16, 24]:
        v = integrate_3d(h, n, 0, 1)
        print(f"  3D oscillatory, n={n}: integral = {v}")
