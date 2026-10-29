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

"""Section 5 -- Truncation, symmetry and continuum limits (Problem 1).

Definition 5.1, Proposition 5.2, Question 5.3 (open), Proposition 5.4, Conjecture 5.5 (open),
Proposition 5.6 and the following Remark, Conjecture 5.7 (open).
"""
import itertools as it
import numpy as np
from common import (check, note, section, summary, kron, SY, pauli_string, all_pauli_labels, weight, J_apply_conj)

rng = np.random.default_rng(505)

# ------------------------------------------------------------------ Definition 5.1
section("Definition 5.1  truncation projection and encoding isometry")
D, c = 16, 6
P = np.diag([1.0] * c + [0.0] * (D - c))
check("P_jmax is an orthogonal projection (P^2 = P = P^dag)", np.allclose(P @ P, P) and np.allclose(P, P.T))
n = 3
V = np.linalg.qr(rng.normal(size=(2 ** n, c)) + 1j * rng.normal(size=(2 ** n, c)))[0]
check(f"an isometry V: P H -> (C^2)^(x){n} exists when dim P H = {c} <= 2^n = {2 ** n}", np.allclose(V.conj().T @ V, np.eye(c)))

# ------------------------------------------------------------------ Proposition 5.2
section("Proposition 5.2  truncation preserves positivity")
for trial in range(5):
    M = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); theta = M @ M.conj().T
    a = rng.normal(size=D) + 1j * rng.normal(size=D)
    Pa, Qa = P @ a, a - P @ a
    full = np.vdot(a, theta @ a).real
    rhs = np.vdot(Pa, theta @ Pa).real + 2 * np.vdot(Pa, theta @ Qa).real + np.vdot(Qa, theta @ Qa).real
    check(f"trial {trial}: <tA,A> = <tPA,PA> + 2Re<tPA,QA> + <tQA,QA>, and <tPA,PA> >= 0",
          np.isclose(full, rhs) and np.vdot(Pa, theta @ Pa).real >= -1e-12)

section("Question 5.3")
note("Question 5.3 (Q-structure and reflection positivity) is open; no computation applies")

# ------------------------------------------------------------------ Proposition 5.4
section("Proposition 5.4  constraints on Q-symmetric encodings")
for n in (2, 3, 4):
    labs = all_pauli_labels(n)
    H = sum(rng.normal() * pauli_string(l) for l in labs)
    Hsym = (H + J_apply_conj(n, H)) / 2
    coeffs = {l: np.trace(pauli_string(l) @ Hsym).real / 2 ** n for l in labs}
    check(f"n={n}: (1) symmetrising a random H under J kills exactly the odd-weight terms",
          all(abs(c) < 1e-10 for l, c in coeffs.items() if weight(l) % 2 == 1))
for n in (3, 5):
    labs = [l for l in all_pauli_labels(n) if weight(l) % 2 == 0 and l != "I" * n]
    w = np.linalg.eigvalsh(sum(rng.normal() * pauli_string(l) for l in labs))
    check(f"n={n}: (2) ground level degenerate; gap measured to the next distinct level is > 0",
          np.isclose(w[0], w[1]) and w[2] - w[0] > 1e-6)
labs = [l for l in all_pauli_labels(4) if weight(l) % 2 == 0 and l != "IIII"]
w = np.linalg.eigvalsh(sum(rng.normal() * pauli_string(l) for l in labs))
w2 = np.linalg.eigvalsh(-pauli_string("ZZ"))
check("(3) n even: unique ground level possible (random H, n=4) and degenerate possible (-Z1Z2, n=2)",
      w[1] - w[0] > 1e-6 and np.isclose(w2[0], w2[1]))

section("Conjecture 5.5")
note("Conjecture 5.5 (parity independence of the continuum gap) is open; no computation applies")

# ------------------------------------------------------------------ Proposition 5.6
section("Proposition 5.6  J-compatible isometric embeddings exist iff n' - n is even")


def intertwiner_dim(n1, n2):
    S1, S2 = kron(*[SY] * n1), kron(*[SY] * n2)
    d1, d2 = 2 ** n1, 2 ** n2
    cols = []
    for k in range(2 * d1 * d2):
        E = np.zeros(d1 * d2, complex); E[k % (d1 * d2)] = 1 if k < d1 * d2 else 1j
        X = E.reshape(d2, d1)
        R = X @ S1 - S2 @ X.conj()
        cols.append(np.concatenate([R.real.ravel(), R.imag.ravel()]))
    M = np.array(cols).T
    return M.shape[1] - np.linalg.matrix_rank(M)


for n1 in (1, 2, 3):
    check(f"n={n1}: real-linear maps with iota J_{n1} = J_{n1 + 1} iota: only 0", intertwiner_dim(n1, n1 + 1) == 0)
    check(f"n={n1}: maps intertwining J_{n1} and J_{n1 + 2}: nonzero space", intertwiner_dim(n1, n1 + 2) > 0)
phi = np.array([0, 1, 1, 0], complex) / np.sqrt(2)
check("phi = (|01>+|10>)/sqrt2 satisfies J_2 phi = phi", np.allclose(kron(SY, SY) @ phi.conj(), phi))
for n, m in ((1, 1), (2, 1), (1, 2), (3, 2)):
    d = 2 ** n
    pm = phi
    for _ in range(m - 1):
        pm = np.kron(pm, phi)
    iota = np.kron(np.eye(d), pm.reshape(-1, 1))
    ok = np.allclose(iota.conj().T @ iota, np.eye(d))
    for _ in range(10):
        x = rng.normal(size=d) + 1j * rng.normal(size=d)
        ok &= np.allclose(iota @ (kron(*[SY] * n) @ x.conj()), kron(*[SY] * (n + 2 * m)) @ (iota @ x).conj())
    check(f"n={n} -> n'={n + 2 * m}: iota(x) = x (x) phi^(x){m} is an isometry with iota J_n = J_n' iota", ok)

# Remark after Proposition 5.6: compatible forms are constant along the chain
n = 2; d = 2 ** n
iota = np.kron(np.eye(d), phi.reshape(-1, 1))
M = rng.normal(size=(4 * d, 4 * d)) + 1j * rng.normal(size=(4 * d, 4 * d)); theta2 = M + M.conj().T
theta1 = iota.conj().T @ theta2 @ iota
x = rng.normal(size=d) + 1j * rng.normal(size=d)
check("Remark: if iota* theta_(n+2) iota = theta_n then <theta x, x> is the same at both stages",
      np.isclose(np.vdot(x, theta1 @ x), np.vdot(iota @ x, theta2 @ (iota @ x))))

section("Conjecture 5.7")
note("Conjecture 5.7 (inductive Q-regularisation of Yang-Mills) is open; no computation applies")

raise SystemExit(summary("Section 5"))
