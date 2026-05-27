"""
Piecewise integration of CDet integrands.

The CDet integrand has discontinuities at vertex times that coincide
with external times (because G_0 changes form when tau crosses 0). For
fixed externals at (tau_in, tau_out), each time integral over [0, beta]
splits into pieces at the external times.

For tau_in = 0, the only interior discontinuity is at tau_out.
For tau_in > 0, both tau_in and tau_out are interior discontinuities.

We split [0, beta] at all interior discontinuities and use Gauss-Legendre
on each smooth piece.

"""
import numpy as np
from cdet_reference.quadrature.quadrature import gl_grid_1d


def split_interval(beta, discontinuities):
    """Return ordered list of break points in [0, beta], including 0 and beta.
    
    discontinuities: list of values that should be break points (may include
    0 or beta, these don't add to the breakpoint count).
    """
    points = sorted(set([0.0, beta] + [d for d in discontinuities if 0 < d < beta]))
    return points


def integrate_piecewise_1d(f, n_points_per_piece, beta, discontinuities):
    """integral_0^beta f(x) dx using Gauss-Legendre on each piece between breakpoints."""
    points = split_interval(beta, discontinuities)
    total = 0.0
    for i in range(len(points) - 1):
        a, b = points[i], points[i+1]
        nodes, weights = gl_grid_1d(n_points_per_piece, a, b)
        for j, x in enumerate(nodes):
            total += weights[j] * f(x)
    return total


def integrate_piecewise_2d(f, n_per_piece, beta, discontinuities_1, discontinuities_2):
    """int int f(x1, x2) dx1 dx2 using piecewise GL on each variable."""
    pts1 = split_interval(beta, discontinuities_1)
    pts2 = split_interval(beta, discontinuities_2)
    total = 0.0
    for i in range(len(pts1) - 1):
        a1, b1 = pts1[i], pts1[i+1]
        nodes1, weights1 = gl_grid_1d(n_per_piece, a1, b1)
        for j in range(len(pts2) - 1):
            a2, b2 = pts2[j], pts2[j+1]
            nodes2, weights2 = gl_grid_1d(n_per_piece, a2, b2)
            for ii, x1 in enumerate(nodes1):
                for jj, x2 in enumerate(nodes2):
                    total += weights1[ii] * weights2[jj] * f(x1, x2)
    return total


def integrate_piecewise_3d(f, n_per_piece, beta, d1, d2, d3):
    """int intintegral f(x1, x2, x3) using piecewise GL on each variable."""
    pts1 = split_interval(beta, d1)
    pts2 = split_interval(beta, d2)
    pts3 = split_interval(beta, d3)
    total = 0.0
    for i in range(len(pts1) - 1):
        a1, b1 = pts1[i], pts1[i+1]
        nodes1, weights1 = gl_grid_1d(n_per_piece, a1, b1)
        for j in range(len(pts2) - 1):
            a2, b2 = pts2[j], pts2[j+1]
            nodes2, weights2 = gl_grid_1d(n_per_piece, a2, b2)
            for k in range(len(pts3) - 1):
                a3, b3 = pts3[k], pts3[k+1]
                nodes3, weights3 = gl_grid_1d(n_per_piece, a3, b3)
                for ii, x1 in enumerate(nodes1):
                    for jj, x2 in enumerate(nodes2):
                        for kk, x3 in enumerate(nodes3):
                            total += (weights1[ii] * weights2[jj] * weights3[kk]
                                      * f(x1, x2, x3))
    return total


if __name__ == "__main__":
    import sys, time
    from cdet_reference.dimer.cdet_recursion import C_V_2site
    
    # Re-test the 3D integration on CDet with discontinuity handling
    beta = 2.0
    mu = 0.3
    t = 1.0
    tau_out = 1.0
    x_out = (0, tau_out)
    x_in = (0, 0.0)
    discontinuities = [tau_out]  # tau_in = 0 is on the boundary
    
    print("Testing piecewise integration on n=3 CDet (1 site config: all on site 0)")
    f = lambda t1, t2, t3: C_V_2site([(0, t1), (0, t2), (0, t3)], x_out, x_in, beta, mu, t)
    
    for n_per_piece in [6, 8, 12, 16]:
        t0 = time.time()
        v = integrate_piecewise_3d(f, n_per_piece, beta, discontinuities, discontinuities, discontinuities)
        print(f"  n_per_piece={n_per_piece}: integral = {v}, time = {time.time()-t0:.1f}s")
