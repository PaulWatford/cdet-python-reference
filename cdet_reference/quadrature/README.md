# Gauss-Legendre integration helpers

General-purpose numerical integration used by all CDet audits.
Independent of the Hubbard physics.

The CDet recursion produces integrands with kinks (sub-derivative
discontinuities) at the external times $\tau_{\rm in}, \tau_{\rm out}$.
Naive Gauss-Legendre across the full interval $[0, \beta]$ loses
precision rapidly near these kinks. The `_piecewise` and `_nested`
variants split the integration domain at the kink locations so each
sub-interval is smooth.

## Files

- `quadrature.py`: base Gauss-Legendre quadrature in 1D-4D.
- `quadrature_piecewise.py`: kink-aware 1D piecewise GL.
- `quadrature_nested.py`: kink-aware nested ND integration including
  simplex 3D/4D (for CDet's $\tau_1 \leq \tau_2 \leq \cdots \leq \tau_n$
  ordered-times integration domain).

## API surface

The functions imported elsewhere in the package:

- `gl_grid_1d(n, a, b)`: Gauss-Legendre points and weights on $[a, b]$.
- `integrate_1d_with_kinks(f, n_per_piece, beta, kinks)`: 1D with
  domain split at the listed kink locations.
- `integrate_nested_2d`: `integrate_nested_3d`, nested kink-aware
  integration in higher dimensions.
- `integrate_simplex_3d`: `integrate_simplex_4d`, integration over
  the ordered-times simplex.
