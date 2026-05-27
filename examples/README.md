# Runnable usage examples

These scripts show how to use `cdet_reference` from your own Python
code. Each is self-contained, install the package (see top-level
`README.md`) and run any of them directly.

| File | What it does | Runtime |
|---|---|---|
| `01_compute_G0.py`            | Call the free Green's function on each lattice; sanity-check anti-periodicity and the $t \to 0$ limit | < 1 s |
| `02_atom_cdet_taylor.py`      | Compute the first Taylor coefficient $c_1(\tau; \mu)$ via CDet and compare to finite-difference of the exact symbolic answer | ~ 1 s |
| `03_benchmark_vs_ED.py`       | Diagonalise the 4096-state hexring Hamiltonian for several values of $U$; compare to the free $G_0$ to see the perturbative regime emerge | ~ 30 s |
| `04_spectral_function_plot.py`| Compute $A(k, \omega)$ on the hexring via the Lehmann representation; save a 4-panel PNG plot | ~ 30 s |

## Run any one of them

From the repo root:

```bash
python examples/01_compute_G0.py
python examples/02_atom_cdet_taylor.py
python examples/03_benchmark_vs_ED.py
python examples/04_spectral_function_plot.py
```

## Use as templates

Each example follows a copy-paste-friendly pattern: import the
functions you need, set physical parameters, call the function. Take
any of them as a starting point for your own benchmark, parameter
sweep, or plotting code.

## Recommended reading order

1. `01_compute_G0.py`, confirms the package is installed and you can
   call its functions.
2. `02_atom_cdet_taylor.py`, the simplest possible end-to-end CDet
   check; the conceptual workflow at minimum scale.
3. `03_benchmark_vs_ED.py`, the typical benchmark use case: compare a
   production code's output against the exact diagonalisation reference.
4. `04_spectral_function_plot.py`, visual output you can show in a
   talk or paper.
