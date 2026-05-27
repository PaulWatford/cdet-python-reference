# Single-site Hubbard atom

The single-orbital Hubbard model. 4-state Hilbert space: $\{|0\rangle, |\uparrow\rangle, |\downarrow\rangle, |\uparrow\downarrow\rangle\}$.

This is the minimal nontrivial fermionic system. CDet here can be
verified against a fully symbolic Taylor expansion in $U$ (no ED
needed), so the audits run at machine precision through $n=4$.

## Files

- `lehmann_derivation.md`: derivation of the free Green's function
  $G_0(\tau)$ via Lehmann decomposition.
- `wick_derivation.md`: derivation of the Wick determinant $D_V$
  with the $(-1)^{|V|}$ sign convention.
- `cdet_derivation.md`: derivation of the Rossi connected-determinant
  recursion $C_V$.
- `green_function.py`: `G0_atom`, `G_exact_atom` (sympy), `n_F`.
- `wick_determinant.py`: `D_corr`, `D_vac`.
- `cdet_recursion.py`: `C_V`.

## Tests

Tests for this subsystem live in the top-level `tests/` directory:
`test_atom_green_function.py`, `test_atom_wick_determinant.py`,
`test_atom_cdet.py`.
