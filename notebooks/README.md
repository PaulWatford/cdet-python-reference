# Jupyter tutorials

| File | What it covers |
|---|---|
| `tutorial.ipynb` | End-to-end walk-through: free Green's function on each lattice -> plot it as a function of imaginary time -> run a CDet Taylor coefficient -> exact diagonalisation comparison -> spectral function $A(k, \omega)$ |

## Running

Install the package first:

```bash
pip install -e .
```

Then launch Jupyter from the repo root:

```bash
jupyter notebook notebooks/tutorial.ipynb
# or
jupyter lab
```

Each notebook can be run top to bottom. Run-time on a 2024-era laptop:
about 60 seconds total (most of which is the ED step in cells 4 and 5).
