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

"""Section 4 -- Real and quaternionic free group factors (Problem 2).

Definition 4.1, Theorem 4.2 (consistency), Proposition 4.3 (Exercise 6), Remark 4.4, Conjecture 4.5
(dimension count and why the draft theorem fails), Question 4.6 (open).
Free group factors are infinite-dimensional; the checks are exact arithmetic for the parameter formulas
and finite-dimensional models for the algebraic identities used in the proofs.
"""
from fractions import Fraction as Fr
import itertools as it
import math
import random
import numpy as np
from common import (check, note, section, summary, qmul, qconj, QBASIS, phi, real_rank, complex_rank, I2, kron)

rng = np.random.default_rng(404)
random.seed(404)
amp = lambda r, T: 1 + (r - 1) / T        # parameter of L(F_r)_t with T = t^2

# ------------------------------------------------------------------ Theorem 4.2
section("Theorem 4.2  algebraic consistency of r -> 1 + (r-1)/t^2 (cited: Dykema, Radulescu)")
ok_act = ok_inv = ok_rng = True
for _ in range(2000):
    r = Fr(random.randint(101, 5000), 100)
    S, T = Fr(random.randint(1, 900), random.randint(1, 30)), Fr(random.randint(1, 900), random.randint(1, 30))
    ok_act &= amp(amp(r, S), T) == amp(r, S * T)
    ok_inv &= amp(amp(r, T), 1 / T) == r
    ok_rng &= amp(r, T) > 1
check("(L(F_r)_s)_t = L(F_r)_(st) (group action), 2000 exact rational cases", ok_act)
check("f_(1/t) inverts f_t", ok_inv)
check("parameter stays > 1", ok_rng)
check("t = 1/k reproduces Voiculescu's L(F_n)_(1/k) = L(F_(1+k^2(n-1)))",
      all(amp(Fr(n), Fr(1, k * k)) == 1 + k * k * (n - 1) for n in range(2, 30) for k in range(1, 30)))
note("Theorem 4.2 itself is a cited theorem about infinite-dimensional algebras and is not computable")

# ------------------------------------------------------------------ Proposition 4.3
section("Proposition 4.3  M_2(L(F_k)) = L(F_((k+3)/4)); same complexification (Exercise 6)")
check("1 + (k-1)/2^2 = (k+3)/4 exactly, k = 2..1000", all(amp(Fr(k), 4) == Fr(k + 3, 4) for k in range(2, 1001)))
check("same identity for 1000 random rational k > 1",
      all(amp(k, 4) == (k + 3) / 4 for k in (Fr(random.randint(101, 10 ** 5), 100) for _ in range(1000))))
check("k = 5 gives L(F_2)", Fr(5 + 3, 4) == 2)
imgs = [phi(b) for b in QBASIS]
check("H (x)_R C = M_2(C): phi is multiplicative and its C-span is all of M_2(C)",
      all(np.allclose(phi(qmul(a, b)), phi(a) @ phi(b)) for a in QBASIS for b in QBASIS) and complex_rank(imgs) == 4)
perms = list(it.permutations(range(3))); idx = {p: i for i, p in enumerate(perms)}
comp = lambda p, q: tuple(p[q[i]] for i in range(3))


def lam(p):
    M = np.zeros((6, 6))
    for q in perms:
        M[idx[comp(p, q)], idx[q]] = 1
    return M


regA = [lam(p) for p in perms]
AH = [np.kron(phi(q), L) for L in regA for q in QBASIS]
M2A = [np.kron(E, L) for L in regA for E in (np.diag([1, 0]), np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]]), np.diag([0, 1]))]
check("finite model A = R[S_3]: (A (x)_R H) (x)_R C has the same C-span as M_2(A (x)_R C)",
      real_rank(AH) == 24 and complex_rank(AH) == complex_rank(M2A) == complex_rank(AH + [m.astype(complex) for m in M2A]) == 24)

# ------------------------------------------------------------------ Remark 4.4
section("Remark 4.4  complexification does not reflect isomorphism")
check("every nonzero quaternion is invertible: det phi(q) = |q|^2 > 0 (H is a division algebra)",
      all(np.isclose(np.linalg.det(phi(tuple(q))).real, np.dot(q, q)) for q in rng.normal(size=(300, 4))))
N = np.array([[0.0, 1.0], [0.0, 0.0]])
check("M_2(R) has a nonzero nilpotent, so M_2(R) is not isomorphic to H", np.allclose(N @ N, 0) and np.any(N))
check("both complexify to M_2(C)", complex_rank(imgs) == 4 and complex_rank([np.eye(2), N, N.T, np.diag([1.0, -1.0])]) == 4)
note("whether L_H(F_5) and L_R(F_2) are isomorphic is open (Problem 2) and is not tested")

# ------------------------------------------------------------------ Conjecture 4.5
section("Conjecture 4.5  d_beta(N) and why the draft 'Theorem 4.3' is withdrawn")


def sa_basis(beta, N):
    units = {1: [1.0], 2: [1.0, 1j], 4: [phi(q) for q in QBASIS]}[beta]
    out = []
    for i in range(N):
        for j in range(N):
            for u in units:
                E = np.zeros((N, N)); E[i, j] = 1
                M = np.kron(E, u) if beta == 4 else E * u
                out.append(M + M.conj().T)
    return out


check("d_beta(N) = N + beta N(N-1)/2 from explicit bases, beta = 1, 2, 4, N = 1..5",
      all(real_rank(sa_basis(b, N)) == N + b * N * (N - 1) // 2 for b in (1, 2, 4) for N in range(1, 6)))
lg = lambda d, R: d / 2 * math.log(math.pi) - math.lgamma(d / 2 + 1) + d * math.log(R)
diffs = [abs(lg(N * (N + 1) // 2, math.sqrt(N)) - lg(N * N, math.sqrt(N))) for N in (10, 100, 1000, 10000)]
check("log-volumes in spaces of different dimension differ at order N^2 log N (not O(1/N))",
      all(diffs[i + 1] > diffs[i] for i in range(3)) and diffs[-1] / (10000 ** 2 * math.log(10000)) > 0.05)
eps = 1e-3
worst = max((1 + a) + (1 + b) - 2 * (1 - c) for a, b, c in it.product((-eps, 0, eps), repeat=3))
check("(S1,S2,S1) counterexample: moment errors <= eps force tr((Y1-Y3)^2) <= 4 eps", math.isclose(worst, 4 * eps))


def sample(beta, N):
    if beta == 1:
        A = rng.normal(size=(N, N)); return (A + A.T) / np.sqrt(2 * N)
    if beta == 2:
        A = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2 * N); return (A + A.conj().T) / np.sqrt(2)
    w = rng.normal(size=(N, N)); w = (w + w.T) / 2
    comps = [w] + [(lambda v: (v - v.T) / 2)(rng.normal(size=(N, N))) for _ in range(3)]
    return sum(np.kron(c, phi(q)) for c, q in zip(comps, QBASIS)) / np.sqrt(2 * N)


for beta in (1, 2, 4):
    N = 300 if beta < 4 else 150
    A, B = sample(beta, N), sample(beta, N)
    tr = lambda M: np.real(np.trace(M)) / M.shape[0]
    check(f"beta={beta}: independent ensembles are asymptotically free (tr A^2 ~ 1, tr A^2B^2 ~ 1, tr ABAB ~ 0)",
          abs(tr(A @ A) - 1) < 0.05 and abs(tr(A @ A @ B @ B) - 1) < 0.08 and abs(tr(A @ B @ A @ B)) < 0.05)
note("Conjecture 4.5 itself is open and is not tested")

# ------------------------------------------------------------------ Question 4.6
section("Question 4.6")
note("Question 4.6 is open; no computation applies")

raise SystemExit(summary("Section 4"))
