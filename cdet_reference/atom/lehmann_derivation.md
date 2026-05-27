# Hubbard atom Green's function: Lehmann derivation

The single-site Hubbard model with on-site interaction $U n_\uparrow n_\downarrow$
and chemical potential $\mu$ has a 4-state Hilbert space:

| state $|n\rangle$ | $(n_\uparrow, n_\downarrow)$ | energy $E_n$ |
|---|---|---|
| $|0\rangle$ | $(0, 0)$ | $0$ |
| $|\uparrow\rangle$ | $(1, 0)$ | $-\mu$ |
| $|\downarrow\rangle$ | $(0, 1)$ | $-\mu$ |
| $|\uparrow\downarrow\rangle$ | $(1, 1)$ | $U - 2\mu$ |

The partition function is

$$Z = \sum_n e^{-\beta E_n} = 1 + 2 e^{\beta \mu} + e^{-\beta(U - 2\mu)}.$$

## Matrix elements of $c_\uparrow$

Fix the Fock-state convention $|\uparrow\downarrow\rangle = c^\dagger_\downarrow c^\dagger_\uparrow |0\rangle$
(spin-down operator outermost). Anti-commutation gives:

- $c_\uparrow |0\rangle = 0$
- $c_\uparrow |\uparrow\rangle = |0\rangle$, matrix element $\langle 0 | c_\uparrow | \uparrow \rangle = 1$
- $c_\uparrow |\downarrow\rangle = 0$
- $c_\uparrow |\uparrow\downarrow\rangle = -|\downarrow\rangle$, matrix element $\langle \downarrow | c_\uparrow | \uparrow\downarrow \rangle = -1$

The sign in the last line comes from anti-commuting $c_\uparrow$ past $c^\dagger_\downarrow$.

## Lehmann representation

The imaginary-time Green's function is

$$G_\uparrow(\tau) = -\langle \mathcal{T}_\tau\, c_\uparrow(\tau)\, c^\dagger_\uparrow(0) \rangle.$$

For $0 < \tau < \beta$, inserting a complete set of energy eigenstates:

$$G_\uparrow(\tau) = -\frac{1}{Z} \sum_{m, n} |\langle m | c_\uparrow | n \rangle|^2 \,
                    e^{-(\beta - \tau) E_m - \tau E_n}.$$

Only two terms have non-zero matrix elements:

| $n$ | $m$ | $E_n$ | $E_m$ | $|\langle m | c_\uparrow | n \rangle|^2$ |
|---|---|---|---|---|
| $|\uparrow\rangle$ | $|0\rangle$ | $-\mu$ | $0$ | $1$ |
| $|\uparrow\downarrow\rangle$ | $|\downarrow\rangle$ | $U - 2\mu$ | $-\mu$ | $1$ |

Substituting:

$$G_\uparrow(\tau) = -\frac{1}{Z}\left[\, e^{\mu \tau} + e^{\mu(\beta + \tau) - U \tau}\, \right]
                  = -\frac{e^{\mu \tau}}{Z}\left[\, 1 + e^{\mu \beta - U \tau}\, \right].$$

## Reduction at $U = 0$

At $U = 0$, the partition function factorises as $Z_0 = (1 + e^{\beta \mu})^2$ and

$$G_\uparrow(\tau)\bigg|_{U=0} = -\frac{e^{\mu \tau}(1 + e^{\beta \mu})}{(1 + e^{\beta \mu})^2}
                              = -\frac{e^{\mu \tau}}{1 + e^{\beta \mu}}.$$

This matches the standard free atom Green's function $G_0(\tau) = -(1 - n_F(-\mu))\, e^{\mu \tau}$
with $n_F(\xi) = 1/(e^{\beta \xi} + 1)$, providing a cross-check on the
sign and exponent conventions.

## Anti-periodic extension

The expression above gives $G_\uparrow(\tau)$ for $\tau \in (0, \beta)$.
For $\tau \in (-\beta, 0)$, anti-periodicity in $\tau$ gives

$$G_\uparrow(\tau) = -G_\uparrow(\tau + \beta).$$

This is implemented in `green_function.py` as `G_exact_atom(tau, beta, mu, U)`.
