# cdet-python-reference

Verified small-cluster reference implementation of the **Rossi (2017)
connected-determinant (CDet) algorithm** for fermionic perturbation
theory on the Hubbard model.

The package provides free Green's functions built from numerical
diagonalisation, exact-diagonalisation many-body references, Wick
determinant constructions, the CDet recursion, and a verification
ladder of audits for four progressively richer lattices, from the single-site Hubbard atom to a 6-site
honeycomb plaquette. Numerical agreement against the exact references
is verified to machine precision through perturbative orders 2 to 4.

Intended uses:

1. **Golden reference** for production CDet implementations (e.g. in
   C++ DiagMC codes). Any production code can be benchmarked against
   the audits in this repo at small cluster sizes.

2. **Rapid prototyping** of CDet variants, alternative diagram
   structures, and lattice geometries before committing to a
   production port.

3. **Pedagogical resource**, each lattice ships with a step-by-step
   derivation explaining how the free Green's function is constructed
   and the structure of the Wick determinant on that geometry.

## Headline precisions

| System                       | Hilbert dim | Order | Reference                | Agreement   |
|------------------------------|------------:|------:|--------------------------|-------------|
| Hubbard atom                 |           4 |   n=4 | symbolic Taylor          | machine     |
| 2-site dimer                 |          16 |   n=4 | mpmath dps=50 + simplex  | 5.4x10^-^1^2   |
| 4-site pi-flux square         |         256 |   n=2 | ED finite-difference     | 4.8x10^-^8    |
| 4-site pi-flux square         |         256 |   n=3 | Richardson-extrapolated  | 6.5x10^-^7    |
| 6-site honeycomb hexring     |        4096 |   n=2 | ED finite-difference     | 1.2x10^-^7    |

All audits are reproducible via `python -m pytest tests/`.

## Install

```bash
git clone https://github.com/PaulWatford/cdet-python-reference.git
cd cdet-python-reference
pip install -e .
```

This registers `cdet_reference` as an importable Python package on your
system. Dependencies are pulled automatically.

## How to use it

There are three ways into the package, in order of effort:

### 1. Import the functions in your own Python code

This is the main use case, a benchmark for any production CDet
implementation, or a building block for a new calculation.

```python
from cdet_reference.hexring.green_function import G0_hexring
from cdet_reference.atom.cdet_recursion import C_V

# Free Green's function on the 6-site hexring
g = G0_hexring(i=0, j=1, tau=1.5, beta=4.0, mu=0.0, t=1.0)
print(f"G_0(0, 1; tau=1.5) = {g}")
# G_0(0, 1; tau=1.5) = 0.0302571631...
```

### 2. Run the built-in examples

Four standalone scripts in `examples/` that exercise the main use
patterns end to end:

```bash
python examples/01_compute_G0.py              # free Green's function on each lattice
python examples/02_atom_cdet_taylor.py        # a first Taylor coefficient via CDet
python examples/03_benchmark_vs_ED.py         # CDet vs exact diagonalisation
python examples/04_spectral_function_plot.py  # produces A(k, omega) plot
```

Each prints to the terminal and can be used as a copy-paste template
for your own scripts.

### 3. Explore interactively in a Jupyter notebook

```bash
jupyter notebook notebooks/tutorial.ipynb
```

The tutorial walks through the full verification ladder visually with
plots, the friendliest entry point for first-time exploration.

### Run the audit suite

To confirm the package works correctly on your machine:

```bash
python tests/test_atom_cdet.py       # fastest, ~1 minute
python -m pytest tests/              # full ladder, ~15 minutes
```

The audit suite is also the precise specification of what numerical
agreement to expect, see the Headline precisions table above.

## Package layout

```
cdet_reference/
|---- quadrature/           Gauss-Legendre quadrature with kink-aware variants
|---- atom/                 single-site Hubbard atom (4-state)
|---- dimer/                2-site Hubbard dimer (16-state)
|---- piflux_square/        4-site pi-flux square (256-state)
|---- hexring/              6-site honeycomb hexring (4096-state)
+---- spectral_function/    A(k, omega) computation on the hexring (Lehmann)

examples/                 runnable usage demos (4 scripts)
notebooks/                Jupyter tutorial
tests/                    audit scripts, verification ladder per system
```

Each lattice subpackage follows the same file pattern:

| File                          | Role                                                 |
|-------------------------------|------------------------------------------------------|
| `derivation.md`               | Hamiltonian, spectral-sum construction, vertex setup |
| `green_function.py`           | Free Green's function $G_0(i, j; \tau)$              |
| `exact_diagonalization.py`    | Exact $G(i, j; \tau; U)$ via ED (multi-site only)    |
| `wick_determinant.py`         | Wick determinant $D_V$ with $(-1)^{|V|}$ sign        |
| `cdet_recursion.py`           | Rossi CDet recursion $C_V$                           |

The atom is the exception: it also has an exact symbolic $G$ via sympy
(in `green_function.py`) rather than a separate ED file, since the
Hilbert space is only 4-state.

## Build and verification ladder

For any new lattice extension, follow the same five-step ladder:

1. **Derivation**, write out $H$, build the free single-particle
   matrix, set out the spectral-sum construction of $G_0$.
2. **Free Green's function**, implement $G_0$ via the spectral sum
   over the numerically computed eigenstates.
   Audit: matches matrix exponential at machine precision; satisfies
   anti-periodicity in $\tau$; reduces correctly in limits.
3. **Exact diagonalisation**, build $H$ for arbitrary $U$, eigendecompose,
   compute $G(i, j; \tau; U)$ via Lehmann. Audit: matches free $G_0$
   at $U = 0$; satisfies sum rules.
4. **Wick determinant**, construct $D_V$ matrix with externals.
   Audit: factorises correctly at $t = 0$; satisfies vertex permutation
   invariance.
5. **CDet recursion**, implement Rossi's connected-determinant
   recursion. Audit: end-to-end CDet matches $\partial^n G / \partial U^n$
   at $U = 0$ from ED.

Each audit script in `tests/` exercises one step of one system.

## Dependencies

- Python >= 3.10
- numpy, scipy, sympy
- mpmath (for arbitrary-precision references in dimer and pi-flux)
- matplotlib (for spectral function plots)

Install via `pip install -e .` after cloning.

## Citation

If you use this code in published work, please cite both this repository
and the original Rossi paper that the CDet algorithm is from:

> R. Rossi, *Determinant Diagrammatic Monte Carlo Algorithm in the Thermodynamic
> Limit*, Physical Review Letters **119**, 045701 (2017). arXiv:1612.05184.

The verification methodology here follows the precision-floor philosophy
used in modern DiagMC validation (see Moutenet et al. 2018, 2019 for related
work on continuum 2D Hubbard).

## Acknowledgements

The connected-determinant algorithm implemented here is from R. Rossi,
*Phys. Rev. Lett.* **119**, 045701 (2017). This package was designed with
eventual interoperability with the [DiagHam](https://www.nick-ux.org/diagham/)
condensed-matter library in mind, but the implementation is independent
and not derived from DiagHam source code.

## License

MIT, see `LICENSE` for the full text.
