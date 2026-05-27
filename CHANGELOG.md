# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]: 2026-05-26

### Added
- Initial release of the verified small-cluster CDet reference implementation.
- Four lattice subpackages:
  - `cdet_reference.atom`: single-site Hubbard atom (4-state)
  - `cdet_reference.dimer`: 2-site dimer (16-state)
  - `cdet_reference.piflux_square`: 4-site pi-flux square (256-state)
  - `cdet_reference.hexring`: 6-site honeycomb hexagonal ring (4096-state)
- `cdet_reference.quadrature`: Gauss-Legendre quadrature with kink-aware variants.
- `cdet_reference.spectral_function`: exact $A(k, \omega)$ on the hexring via
  the Lehmann representation.
- Verification ladder in `tests/`: 10 audit scripts covering the free Green's
  function, Wick determinant, CDet recursion, and end-to-end Taylor coefficient
  match against ED finite-differences and mpmath references.
- Four runnable usage examples in `examples/`.
- Tutorial Jupyter notebook in `notebooks/`.

### Headline precisions
- Atom n=4 against symbolic Taylor: machine precision
- 2-site dimer n=4 against mpmath dps=50 + simplex: $5.4 \times 10^{-12}$
- 4-site pi-flux n=3 against Richardson-extrapolated reference: $6.5 \times 10^{-7}$
- 6-site hexring n=2 against ED finite-difference: $1.2 \times 10^{-7}$
