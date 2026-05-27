# Single-particle spectral function A(k, omega) on the hexring

Exact computation of the single-particle spectral function

$$A(k, \omega) = -\frac{1}{\pi} \operatorname{Im} G_R(k, \omega + i\eta)$$

on the 6-site honeycomb hexring via the Lehmann representation.

At this cluster size all 4096 eigenvalues and matrix elements of
$c_{k\sigma}$ are known exactly from ED, so the spectral function is
constructed as a closed-form sum of Lorentzians without any analytic
continuation. This makes it a clean reference for benchmarking
DiagMC-derived spectral functions on graphene-class systems.

## Files

- `spectral_function.py`: `spectral_function_exact(k, omega_range, ...)`
  and `density_of_states`. Mahan convention: $A(k, \omega)$ peaks at
  $\omega = \varepsilon_k$ for the non-interacting case.

## Sample outputs

`A_ED_hexring_panel.png` and `A_ED_hexring_DOS.png` show the
non-interacting spectral function at the four band momenta (Gamma, K, K',
M) and the total density of states.

## Band assignment on the hexring

| Momentum index $k$ | $\varepsilon_k = -2t \cos(2\pi k / 6)$ | Band       |
|--------------------|----------------------------------------|------------|
| 0                  | $-2t$                                  | Gamma (lowest) |
| 1, 5               | $-t$                                   | K, K' (Dirac lower) |
| 2, 4               | $+t$                                   | (Dirac upper) |
| 3                  | $+2t$                                  | M (highest) |

## Dependencies

Uses `cdet_reference.hexring.exact_diagonalization` for the ED solver.
