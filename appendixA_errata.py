# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Validation code for: "J-Invariant Subgroups of SU(2^n): Replacing Lie-Algebra Dimension"
# (Cooperman & Petrunina V.V., 29 October 2026).

"""Appendix A -- reproduces each error of the earlier draft (numerical checks behind the errata table) of
'The Q-Structured Simulability Dichotomy: Antiunitary Parity, Real Forms and Continuum Limits',
the predecessor of 'J-Invariant Subgroups of SU(2^n): Replacing Lie-Algebra Dimension'.
Each block tests one draft claim; PASS means the errata verdict is confirmed.
Section labels below use the numbering of the earlier draft."""
import itertools as it
import numpy as np
from common import (check, note, section, summary, kron, SY, SX, SZ, I2, op, pauli_string, all_pauli_labels,
                    weight, J_apply_conj, J_squared, majoranas_z_onsite, majoranas_x_onsite, find_monomial,
                    mask, anticommute, label_of)

rng = np.random.default_rng(1)


def expm_herm(H, t):
    w, V = np.linalg.eigh(H)
    return V @ np.diag(np.exp(-1j * t * w)) @ V.conj().T


section("Thm 1.2  J_n^2 = (-1)^n I  -- correct")
check("J_n^2 = (-1)^n I, n = 1..7", all(np.allclose(J_squared(n), (-1) ** n * np.eye(2 ** n)) for n in range(1, 8)))
check("'involution' wording: J_n^2 = -I for odd n, so J_n is not an involution there", np.allclose(J_squared(3), -np.eye(8)))

section("Lemma 1.4  odd-weight criterion -- correct only 'for all theta'")
ok = True
for n in (2, 3):
    for lab in all_pauli_labels(n):
        if lab == "I" * n: continue
        P = pauli_string(lab)
        for th in rng.uniform(0.1, 3.0, 3):
            U = expm_herm(P, th / 2)
            ok &= np.allclose(J_apply_conj(n, U), U) == (weight(lab) % 2 == 1)
check("generic theta: exp(-i theta P/2) is J-fixed iff w(P) odd (all Pauli strings, n=2,3)", ok)
U = expm_herm(pauli_string("ZZ"), np.pi)          # theta = 2 pi, even weight
check("but at theta = 2 pi an even-weight rotation (= -I) is J-fixed: the lemma needs 'for all theta'",
      np.allclose(J_apply_conj(2, U), U))

section("Lemma 2.1  dim k_n = (4^n - (-2)^n)/2 -- correct")
check("odd-weight count = formula = dim sp/so, n = 1..10",
      all(sum(len(list(it.combinations(range(n), w))) * 3 ** w for w in range(1, n + 1, 2))
          == (4 ** n - (-2) ** n) // 2 == ((2 ** (n - 1)) * (2 ** n + 1) if n % 2 else 2 ** (n - 1) * (2 ** n - 1))
          for n in range(1, 11)))

section("Lemma 2.2  degree 6 -- result correct, proof mixes two Jordan-Wigner conventions")
n = 4; g = majoranas_z_onsite(n)
check("Z-on-site JW: Z1Z2Z3 = i g1...g6", np.allclose(op(n, {0: "Z", 1: "Z", 2: "Z"}), 1j * g[1] @ g[2] @ g[3] @ g[4] @ g[5] @ g[6]))
check("Z-on-site JW: Z_j = -i g_(2j-1) g_(2j)", np.allclose(op(n, {1: "Z"}), -1j * g[3] @ g[4]))
f = find_monomial(op(n, {1: "X"}), g)
check("Z-on-site JW: X_2 is NOT i g3 g4 (proof's first line); it is a degree-3 monomial", f is not None and len(f[0]) == 3, str(f))
gx = majoranas_x_onsite(5); f = find_monomial(op(5, {1: "Z", 2: "Z", 3: "Z"}), gx)
check("X-on-site JW (where X_j is quadratic): Z2Z3Z4 has degree 5, not 6", f is not None and len(f[0]) == 5, str(f))

section("Thm 3.2  'sufficient but not necessary' -- unproved; counterexample invalid")
m = 2  # exponential block on m qubits: odd-weight generators X, Z, ZZZ... here use X1, Z1, X2, Z2, Z1Z2-free set on 2 qubits
gens = [pauli_string(l) for l in ("XI", "ZI", "IX", "IZ", "XY")]
moved = []
for _ in range(20):
    U = np.eye(4, dtype=complex)
    for G in gens:
        U = expm_herm(G, rng.uniform(0, 2 * np.pi)) @ U
    moved.append(abs(U[0, 0]) ** 2)
check("a non-abelian block does not leave |0...0> in place: P(0...0) < 1 for random angles, "
      "so Pr(x) = ... * delta_(x_exp,0) is false", max(moved) < 0.999, f"max P(00) = {max(moved):.3f}")
note("displayed relation uses the divisibility sign '∤' and the direction 'poly dim ⇏ intractable'; "
      "non-necessity is 'C_sample ∈ Poly ⇏ dim ≤ Poly'")

section("Lemma 3.3  'adjoint action preserves degree 6' -- false")
gz = majoranas_z_onsite(6)
A = gz[1] @ gz[2] @ gz[3] @ gz[4] @ gz[5] @ gz[6]
B = gz[6] @ gz[7] @ gz[8] @ gz[9] @ gz[10] @ gz[11]
C = A @ B - B @ A
f = find_monomial(C / 2, gz) or find_monomial(-C / 2, gz) or find_monomial(1j * C / 2, gz) or find_monomial(-1j * C / 2, gz)
check("[deg-6, deg-6] sharing one Majorana is a NONZERO monomial of degree 10", np.abs(C).max() > 0 and f is not None and len(f[0]) == 10,
      str(f[0]) if f else "")
note("dim of the state space is 2^n (as in the statement), but the proof writes O(n^(2n)); the operator space is 4^n")

section("Thm 3.5  disconnected commutation graph does not give a tensor factorization")
XX, ZZ = op(2, {0: "X", 1: "X"}), op(2, {0: "Z", 1: "Z"})
check("XX and ZZ commute (two components, no edge) yet act on the same qubits", np.allclose(XX @ ZZ, ZZ @ XX))
U = expm_herm(XX, 0.7) @ expm_herm(ZZ, 1.1)
psi = U[:, 0]; P = np.abs(psi) ** 2
p1 = np.array([P[0] + P[1], P[2] + P[3]]); p2 = np.array([P[0] + P[2], P[1] + P[3]])
check("output distribution does not factor over qubits (P != P1 x P2)", not np.allclose(P, np.kron(p1, p2)), str(np.round(P, 3)))
note("with disjoint supports of size <= c log n the DLA is <= m * 4^(c log n) = poly: dim g cannot -> infinity "
      "superpolynomially, so the 'independent of dim g' clause is vacuous")

section("Thm 4.3  beta-invariance of delta* = k -- false as stated (same as earlier paper)")
for N in (10, 100, 1000):
    d1, d2 = N * (N + 1) // 2, N * N
    print(f"    N={N}: real dims of GOE vs GUE microstate spaces: {d1} vs {d2}")
note("GOE and GUE microstate spaces have different real dimension, so their log-volumes differ at order N^2 log N, not O(1/N)")

section("Thm 5.2  reflection positivity obstruction -- contradicts its own hypothesis")
D, c = 12, 5
Mx = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); theta = Mx @ Mx.conj().T   # positive: RP holds
a = rng.normal(size=D) + 1j * rng.normal(size=D)
Pa = a.copy(); Pa[c:] = 0; tail = a - Pa
full = np.vdot(a, theta @ a).real; trunc = np.vdot(Pa, theta @ Pa).real; R = np.vdot(tail, theta @ tail).real
check("truncated vector is a vector of the full space, so <theta P A, P A> >= 0 whenever RP holds", trunc >= 0)
check("the identity <theta PA,PA> = <theta A,A> - R_tail omits cross terms (fails for generic theta)",
      not np.isclose(trunc, full - R), f"{trunc:.2f} vs {full - R:.2f}")

section("Lemma 5.3  parity of n and the ground state")
H = -ZZ
check("Z1Z2 commutes with J_2 (even weight)", np.allclose(J_apply_conj(2, H), H))
w = np.linalg.eigvalsh(H)
check("J_n^2 = +I does NOT force a unique ground state: -Z1Z2 has a 2-fold degenerate ground level",
      np.isclose(w[0], w[1]), str(np.round(w, 3)))
H3 = sum(rng.normal() * pauli_string(l) for l in all_pauli_labels(3) if weight(l) % 2 == 0 and l != "III")
w3 = np.linalg.eigvalsh(H3)
check("J_n^2 = -I (n=3): a random J-symmetric H has all levels in Kramers pairs (correct part of the lemma)",
      np.allclose(w3[0::2], w3[1::2]))

section("Thm 5.4  no isometry can intertwine J_n and J_(n+1)")


def intertwiner_dim(n1, n2):
    """real dimension of {iota : C^(2^n1) -> C^(2^n2) linear, iota J_n1 = J_n2 iota}."""
    S1, S2 = kron(*[SY] * n1), kron(*[SY] * n2)
    d1, d2 = 2 ** n1, 2 ** n2
    # iota J1 x = J2 iota x  <=>  iota S1 conj(x) = S2 conj(iota) conj(x)  <=>  iota S1 = S2 conj(iota)
    rows = []
    for k in range(2 * d1 * d2):
        E = np.zeros(d1 * d2, complex)
        E[k % (d1 * d2)] = 1 if k < d1 * d2 else 1j
        X = E.reshape(d2, d1)
        Rm = X @ S1 - S2 @ X.conj()
        rows.append(np.concatenate([Rm.real.ravel(), Rm.imag.ravel()]))
    Mtx = np.array(rows).T
    return Mtx.shape[1] - np.linalg.matrix_rank(Mtx)


for n1 in (1, 2, 3):
    d_next, d_skip = intertwiner_dim(n1, n1 + 1), intertwiner_dim(n1, n1 + 2)
    check(f"n={n1}: maps intertwining J_{n1} and J_{n1 + 1} = 0 only; J_{n1} and J_{n1 + 2}: {d_skip}-dim space",
          d_next == 0 and d_skip > 0)

raise SystemExit(summary("Assessment checks"))
