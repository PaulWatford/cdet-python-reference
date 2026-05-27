# Wick determinant D_V for the Hubbard atom: derivation

## Final result (the formula to implement)

For the Hubbard atom with on-site interaction U n_up n_dn and external
operators c_up(x_out), c_up+(x_in):

$$\boxed{D_V(x_\text{out}, x_\text{in}) = (-1)^{|V|} \cdot \det(M_\uparrow) \cdot \det(M_\downarrow)}$$

$$\boxed{D_V(\emptyset) = (-1)^{|V|} \cdot \det(A)^2}$$

where:

- $M_\uparrow$ is $(|V|+1) \times (|V|+1)$:
  - row 0 <- $x_\text{out}$, rows 1..n <- V[k]
  - col 0 <- $x_\text{in}$, cols 1..n <- V[k]
  - $M_\uparrow[a, b] = G_0(\tau^{\text{row}}_a - \tau^{\text{col}}_b)$
  - diagonal $a = b > 0$: $G_0(0^-) = n_0$ (the free density)

- $M_\downarrow$ is $|V| \times |V|$:
  - rows and cols both indexed by V (any consistent order)
  - $M_\downarrow[i, j] = G_0(\tau_i - \tau_j)$
  - diagonal: $G_0(0^-) = n_0$

- $A$ is the same $|V| \times |V|$ matrix as $M_\downarrow$ (for the
  Hubbard atom both spins are equivalent, so the up and down vacuum
  determinants are equal).

The $(-1)^{|V|}$ comes from the perturbative expansion of $e^{-U \int n_\uparrow n_\downarrow}$ at order $|V|$. Each interaction vertex contributes a $(-U)$, so collecting $U^{|V|}$ leaves $(-1)^{|V|}$ as a sign.

---

## Setup

The Hubbard atom has a single site, so a "vertex" is just an imaginary
time $\tau \in [0, \beta]$. A perturbation-order-$n$ contribution has
$n$ internal interaction vertices $V = \{\tau_1, \tau_2, \ldots, \tau_n\}$,
each carrying a factor of $U$.

The interaction is $U n_\uparrow n_\downarrow$. Each interaction vertex
pairs one $\uparrow$ line with one $\downarrow$ line at the same time.
So a vertex $\tau_i$ can be thought of as two "half-vertices": one on
the up Wick contraction graph, one on the dn graph.

## Free Green's function and its convention

$$G_0(\tau, \tau') = -\langle T_\tau c(\tau) c^\dagger(\tau')\rangle_0$$

For the atom, this depends only on $\tau - \tau'$. The function `G0_atom`
in `green_function.py` implements this with antiperiodic boundary
conditions.

At equal time $\tau = 0^-$:

$$G_0(0^-) = \langle c^\dagger c \rangle_0 = n_F(-\mu) = n_0$$

(the free density at chemical potential $\mu$).

## D_V(x_out, x_in): all diagrams (connected + disconnected)

By Wick's theorem applied to a fermionic theory with local interaction
$U n_\uparrow n_\downarrow$, the sum of ALL Feynman diagrams at fixed $V$
factors into two independent determinants (one per spin), each over the
relevant contractions.

For the **two-point function** with external vertices $x_\text{in} = (\tau_\text{in}, \uparrow)$ and $x_\text{out} = (\tau_\text{out}, \uparrow)$:

- The up contractions involve the two external operators plus the $n$
  pairs from the vertices. The matrix is $(n+1) \times (n+1)$.
- The dn contractions involve only the $n$ vertex pairs (no externals
  for dn). The matrix is $n \times n$.

## Origin of the $(-1)^{|V|}$ sign factor

The perturbative expansion of the interacting Green's function is

$$G(\tau) = -\langle T_\tau c(\tau) c^\dagger(0) \exp\!\left[-U \int_0^\beta d\tau'\, n_\uparrow(\tau') n_\downarrow(\tau')\right] \rangle_0.$$

Expanding the exponential at order $|V|$ gives a factor $(-U)^{|V|} / |V|!$. The $|V|!$ is absorbed by summing over labelled vertex configurations, leaving the $(-1)^{|V|}$ as the sign attached to $D_V$ once $U^{|V|}$ is factored out.

### Consistency at $n = 1$

As a sanity check, the first-order Taylor coefficient of $G$ is

$$c_1(\tau) = n_0 \int_0^\beta d\tau_1\, G_0(\tau - \tau_1)\, G_0(\tau_1),$$

with a positive sign. Using the $(-1)^{|V|}$ convention,

$$D_{\tau_1}(x_\text{out}, x_\text{in}) = -\big[\,G_0(\tau_\text{out} - \tau_\text{in})\, n_0 - G_0(\tau_\text{out} - \tau_1)\, G_0(\tau_1 - \tau_\text{in})\,\big]\, n_0,$$
$$D_{\tau_1}(\emptyset) = -n_0^2.$$

The CDet recursion then gives

$$C_{\tau_1} = D_{\tau_1}(x_\text{out}, x_\text{in}) - C_\emptyset\, D_{\tau_1}(\emptyset) = +n_0 \, G_0(\tau_\text{out} - \tau_1)\, G_0(\tau_1 - \tau_\text{in}),$$

recovering the positive sign of $c_1$ (after the $\tau_1$ integration). Numerical check at $(\beta=5, \mu=0.3, \tau=1.5)$: $c_1$ via the Hartree integral matches the sympy Taylor expansion to $2.2 \times 10^{-16}$.

---

## Verification at n = 0

$V = \emptyset$, so $M_\uparrow$ is $1 \times 1 = [G_0(\tau_\text{out} - \tau_\text{in})]$, $M_\downarrow$ is $0 \times 0$ with $\det = 1$.

$$D_\emptyset(x_\text{out}, x_\text{in}) = (-1)^0 \cdot G_0(\tau_\text{out} - \tau_\text{in}) \cdot 1 = G_0(\tau_\text{out} - \tau_\text{in})$$
$$D_\emptyset(\emptyset) = (-1)^0 \cdot 1 \cdot 1 = 1$$

## Verification at n = 1

$V = \{\tau_1\}$.

$$M_\uparrow = \begin{pmatrix} G_0(\tau_\text{out} - \tau_\text{in}) & G_0(\tau_\text{out} - \tau_1) \\ G_0(\tau_1 - \tau_\text{in}) & G_0(0^-) \end{pmatrix}$$

$$\det M_\uparrow = G_0(\tau_\text{out} - \tau_\text{in}) \cdot n_0 - G_0(\tau_\text{out} - \tau_1) \cdot G_0(\tau_1 - \tau_\text{in})$$

$$M_\downarrow = [G_0(0^-)] = [n_0]; \quad \det M_\downarrow = n_0$$

$$D_{\tau_1}(x_\text{out}, x_\text{in}) = -[n_0^2 G_0 - n_0 G_0 G_0] = -n_0^2 G_0 + n_0 G_0 G_0$$

$$D_{\tau_1}(\emptyset) = -n_0^2$$

CDet recursion:
$$C_{\tau_1} = D_{\tau_1}(x_\text{out}, x_\text{in}) - C_\emptyset \cdot D_{\tau_1}(\emptyset)$$
$$= (-n_0^2 G_0 + n_0 G_0 G_0) - G_0 \cdot (-n_0^2)$$
$$= n_0 G_0(\tau_\text{out} - \tau_1) G_0(\tau_1 - \tau_\text{in}) \checkmark$$

---

## Extension to 2-site, 4-site

The formula generalises directly: replace $G_0(\tau)$ with $G_0(i, j; \tau)$ where $i, j$ are site labels. Vertices become $(i, \tau)$ pairs.

See `../dimer/derivation.md` and `../piflux_square/derivation.md` for the lattice extensions.

---

## End-to-end test status

For each system, the verification ladder is:

1. **n = 0**: $D_\emptyset(x_\text{out}, x_\text{in}) = G_0$, direct check.
2. **n = 1**: integrate $C_{\tau_1}$ over $\tau_1$; compare to $c_1$ from
   Taylor expansion of $G_\text{exact}$.
3. **n >= 2**: simplex integration over $V$; compare to mpmath finite-diff
   reference at high dps.

All systems (atom, 2-site, 4-site pi-flux) have passed this ladder through
at least $n = 3$ at machine precision floor (or up to reference precision
floor where the reference is finite-diff-limited).

## Implementation note

The sign factor $(-1)^{|V|}$ is easy to miss from the formal Wick
algebra alone. The implementation in `wick_determinant.py` is anchored
against the symbolic atom Green's function (verified independently in
`green_function.py`) at $n = 0$ and $n = 1$, which fixes the sign
unambiguously. The recursion in `cdet_recursion.py` then uses this
verified $D_V$ definition.
