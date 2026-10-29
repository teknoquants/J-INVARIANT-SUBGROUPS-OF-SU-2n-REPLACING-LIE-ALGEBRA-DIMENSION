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

"""Section 1 -- The canonical antiunitary framework.

Theorem 1.2, Definition 1.3, Lemma 1.4, Corollary 1.5, Corollary 1.6 (with Exercise 10).
"""
import itertools as it
import numpy as np
from common import (check, note, section, summary, kron, SY, SX, SZ, I2, pauli_string, all_pauli_labels,
                    weight, J_apply_conj, J_squared, rot, expm_herm, real_rank)

rng = np.random.default_rng(101)

# ------------------------------------------------------------------ Theorem 1.2
section("Theorem 1.2  J_n antiunitary, J_n^2 = (-1)^n I")
for n in range(1, 8):
    check(f"n={n}: J_n^2 = {(-1) ** n:+d} I", np.allclose(J_squared(n), (-1) ** n * np.eye(2 ** n)))
for n in (1, 2, 3):
    S = kron(*[SY] * n)
    x = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    y = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    Jx, Jy = S @ x.conj(), S @ y.conj()
    a, b = rng.normal() + 1j * rng.normal(), rng.normal() + 1j * rng.normal()
    anti = np.allclose(S @ (a * x + b * y).conj(), a.conjugate() * Jx + b.conjugate() * Jy)
    unit = np.isclose(np.vdot(Jx, Jy), np.vdot(y, x))
    check(f"n={n}: J_n is antilinear and <J x, J y> = <y, x> (antiunitary)", anti and unit)
check("S_n K S_n K = S_n conj(S_n) = (-1)^n S_n^2 (proof step), n = 1..5",
      all(np.allclose(kron(*[SY] * n) @ kron(*[SY] * n).conj(), (-1) ** n * np.eye(2 ** n)) for n in range(1, 6)))
check("J_n is an involution only for even n", np.allclose(J_squared(2), np.eye(4)) and not np.allclose(J_squared(3), np.eye(8)))

# ------------------------------------------------------------------ Definition 1.3
section("Definition 1.3  K_n is a group; k_n is its Lie algebra")
n = 3
odd = [l for l in all_pauli_labels(n) if weight(l) % 2 == 1]


def random_K(n):
    U = np.eye(2 ** n, dtype=complex)
    for _ in range(12):
        U = rot(pauli_string(odd[rng.integers(len(odd))]), rng.uniform(0, 6)) @ U
    return U


U, V = random_K(n), random_K(n)
fixed = lambda M: np.allclose(J_apply_conj(n, M), M)
check("products and inverses of J-fixed unitaries are J-fixed (closure of K_3)",
      fixed(U) and fixed(V) and fixed(U @ V) and fixed(U.conj().T))
A = sum(rng.normal() * 1j * pauli_string(l) for l in odd)
check("exp of an element of k_3 lies in K_3", fixed(expm_herm(1j * A, 0.7)))

# ------------------------------------------------------------------ Lemma 1.4
section("Lemma 1.4  odd-weight criterion (for all theta)")
for n in (1, 2, 3):
    ok = True
    for lab in all_pauli_labels(n):
        if lab == "I" * n:
            continue
        P = pauli_string(lab)
        ok &= np.allclose(J_apply_conj(n, P), (-1) ** weight(lab) * P)
    check(f"n={n}: J P J^-1 = (-1)^w(P) P for all {4 ** n - 1} Pauli strings", ok)
for n in (2, 3):
    ok = True
    for lab in all_pauli_labels(n):
        if lab == "I" * n:
            continue
        P = pauli_string(lab)
        all_theta = all(np.allclose(J_apply_conj(n, rot(P, th)), rot(P, th))
                        for th in rng.uniform(0.05, 6.2, 6))
        ok &= all_theta == (weight(lab) % 2 == 1)
    check(f"n={n}: rotation is J-fixed for all sampled theta  <=>  w(P) odd", ok)
P = pauli_string("ZZ")
check("qualifier needed: at theta = 2 pi an even-weight rotation equals -I and is J-fixed",
      np.allclose(rot(P, 2 * np.pi), -np.eye(4)) and np.allclose(J_apply_conj(2, rot(P, 2 * np.pi)), rot(P, 2 * np.pi)))
check("derivative at theta = 0: J(-iP/2)J^-1 = -(-1)^w (-iP/2), so equality needs w odd",
      np.allclose(J_apply_conj(1, -0.5j * pauli_string("X")), -0.5j * pauli_string("X")))
check("one-qubit identity sigma_y conj(sigma_a) sigma_y = -sigma_a, a = x, y, z; and = +I for I",
      all(np.allclose(SY @ M.conj() @ SY, -M) for M in (SX, SY, SZ)) and np.allclose(SY @ I2 @ SY, I2))

# ------------------------------------------------------------------ Corollary 1.5
section("Corollary 1.5  J-symmetric Hamiltonians are even weight; k_n = span{iP : w odd}")
for n in (2, 3):
    ok = True
    for lab in all_pauli_labels(n):
        ok &= np.allclose(J_apply_conj(n, pauli_string(lab)), pauli_string(lab)) == (weight(lab) % 2 == 0)
    check(f"n={n}: Pauli P commutes with J_n  <=>  w(P) even", ok)
    labs_even = [l for l in all_pauli_labels(n) if weight(l) % 2 == 0]
    labs_odd = [l for l in all_pauli_labels(n) if weight(l) % 2 == 1]
    H = sum(rng.normal() * pauli_string(l) for l in labs_even)
    H2 = H + 0.3 * pauli_string(labs_odd[0])
    check(f"n={n}: random even-weight H commutes with J; adding one odd term breaks it",
          np.allclose(J_apply_conj(n, H), H) and not np.allclose(J_apply_conj(n, H2), H2))
    basis = [1j * pauli_string(l) for l in all_pauli_labels(n) if l != "I" * n]
    kernel = len(basis) - real_rank([J_apply_conj(n, B) - B for B in basis])
    check(f"n={n}: dim of J-fixed part of su(2^n) = number of odd-weight strings = {len(labs_odd)}",
          kernel == len(labs_odd))

# ------------------------------------------------------------------ Corollary 1.6
section("Corollary 1.6  Kramers degeneracy (odd n); nothing forced for even n")
for n in (1, 3, 5):
    labs = [l for l in all_pauli_labels(n) if weight(l) % 2 == 0 and l != "I" * n]
    ok = True
    for _ in range(5):
        H = sum(rng.normal() * pauli_string(l) for l in labs) if labs else np.zeros((2, 2))
        w, V = np.linalg.eigh(H)
        ok &= np.allclose(w[0::2], w[1::2])
        v = V[:, 0]; Jv = kron(*[SY] * n) @ v.conj()
        ok &= abs(np.vdot(v, Jv)) < 1e-10 and np.allclose(H @ Jv, w[0] * Jv)
    check(f"n={n}: every level of a random J-symmetric H is paired; J v is an eigenvector orthogonal to v", ok)
labs = [l for l in all_pauli_labels(4) if weight(l) % 2 == 0 and l != "IIII"]
w = np.linalg.eigvalsh(sum(rng.normal() * pauli_string(l) for l in labs))
check("n=4: generic J-symmetric H has a simple spectrum (no forced degeneracy)", np.min(np.diff(w)) > 1e-6)
w2 = np.linalg.eigvalsh(-pauli_string("ZZ"))
check("n=2: -Z1Z2 is J-symmetric with a doubly degenerate ground level (no forced uniqueness)",
      np.allclose(J_apply_conj(2, -pauli_string("ZZ")), -pauli_string("ZZ")) and np.isclose(w2[0], w2[1]))
labs3 = [l for l in all_pauli_labels(3) if weight(l) % 2 == 0 and l != "III"]
w3 = np.linalg.eigvalsh(sum(rng.normal() * pauli_string(l) for l in labs3))
check("Exercise 10, second part: for odd n, E_1 - E_0 = 0, so the gap is E_next - E_0 > 0",
      np.isclose(w3[1] - w3[0], 0) and w3[2] - w3[0] > 1e-6)

raise SystemExit(summary("Section 1"))
