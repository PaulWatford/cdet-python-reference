"""
cdet_reference, verified small-cluster reference implementation of
the Rossi (2017) connected-determinant algorithm for the fermionic
Hubbard model.

Subpackages:
    atom           , single-site (4-state) reference
    dimer          , 2-site dimer (16-state) reference
    piflux_square  , 4-site pi-flux square (256-state) reference
    hexring        , 6-site honeycomb hexagonal ring (4096-state) reference
    quadrature     , Gauss-Legendre quadrature with kink/discontinuity handling
    spectral_function, A(k, omega) computation on the hexring via Lehmann

See README.md for the build/verification ladder and intended usage.
"""

__version__ = "0.1.0"
