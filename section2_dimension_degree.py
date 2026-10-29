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

"""Section 2 -- Dimension and fermionic degree.

Lemma 2.1 (Exercise 3), Lemma 2.2, Proposition 2.3 (Exercise 7).
"""
import itertools as it
import numpy as np
from common import (check, note, section, summary, kron, SY, op, pauli_string, all_pauli_labels, weight,
                    J_apply_conj, mask, lie_closure_dim, is_abelian, label_of, majoranas_z_onsite,
                    majoranas_x_onsite, find_monomial, rot, zero_state, real_rank, pmask, expm_herm)

rng = np.random.default_rng(202)

# ------------------------------------------------------------------ Lemma 2.1
section("Lemma 2.1  dim k_n = (4^n - (-2)^n)/2 and sp/so type (Exercise 3)")
for n in range(1, 11):
    count = sum(len(list(it.combinations(range(n), w))) * 3 ** w for w in range(1, n + 1, 2))
    formula = (4 ** n - (-2) ** n) // 2
    lie = (2 ** (n - 1)) * (2 ** n + 1) if n % 2 else 2 ** (n - 1) * (2 ** n - 1)
    name = f"sp({2 ** (n - 1)})" if n % 2 else f"so({2 ** n})"
    check(f"n={n}: odd-weight count = (4^n-(-2)^n)/2 = dim {name} = {formula}", count == formula == lie)
check("binomial step: (1+3)^n - (1-3)^n = 2 * sum_odd C(n,w) 3^w, n = 1..12",
      all(4 ** n - (-2) ** n == 2 * sum(len(list(it.combinations(range(n), w))) * 3 ** w for w in range(1, n + 1, 2))
          for n in range(1, 13)))
table = {1: 3, 2: 6, 3: 36, 4: 120, 5: 528}
check("table in the Solution of Exercise 3 (n = 1..5: 3, 6, 36, 120, 528)",
      all((4 ** n - (-2) ** n) // 2 == v for n, v in table.items()))
for n in range(1, 6):
    odd = [mask(l) for l in all_pauli_labels(n) if weight(l) % 2 == 1]
    d, _ = lie_closure_dim(odd)
    check(f"n={n}: span of odd-weight iP is closed under brackets (it is a Lie algebra)", d == len(odd))
for n in range(1, 6):
    S = kron(*[SY] * n)
    labs = [l for l in all_pauli_labels(n) if weight(l) % 2 == 1]
    A = sum(rng.normal() * 1j * pauli_string(l) for l in labs[:30])
    U = expm_herm(1j * A, 1.0)
    kind = "orthogonal (S symmetric)" if n % 2 == 0 else "symplectic (S antisymmetric)"
    form_ok = np.allclose(S.T, S) if n % 2 == 0 else np.allclose(S.T, -S)
    check(f"n={n}: K_n preserves the bilinear form S = sigma_y^(x)n, which is {kind}",
          form_ok and np.allclose(U.T @ S @ U, S))

# ------------------------------------------------------------------ Lemma 2.2
section("Lemma 2.2  Majorana degrees (Z-on-site convention)")
for n in (3, 4, 5):
    g = majoranas_z_onsite(n)
    check(f"n={n}: {{g_r, g_s}} = 2 delta_rs I",
          all(np.allclose(g[r] @ g[s] + g[s] @ g[r], 2 * np.eye(2 ** n) * (r == s)) for r in g for s in g))
n = 6
g = majoranas_z_onsite(n)
check("(1) Z_j = -i g_(2j-1) g_(2j) for all j (n=6)",
      all(np.allclose(op(n, {j - 1: "Z"}), -1j * g[2 * j - 1] @ g[2 * j]) for j in range(1, n + 1)))
check("(1) proof step: g_(2j-1) g_(2j) = X_j Y_j = i Z_j",
      all(np.allclose(g[2 * j - 1] @ g[2 * j], 1j * op(n, {j - 1: "Z"})) for j in range(1, n + 1)))
check("(2) Z_j Z_(j+1) Z_(j+2) = i g_(2j-1) ... g_(2j+4) for all j (n=6)",
      all(np.allclose(op(n, {j - 1: "Z", j: "Z", j + 1: "Z"}),
                      1j * np.linalg.multi_dot([g[k] for k in range(2 * j - 1, 2 * j + 5)])) for j in range(1, n - 1)))
ok = True
for j in range(1, n + 1):
    string = {k: "Z" for k in range(j - 1)}
    ok &= np.allclose(op(n, {j - 1: "X"}), op(n, string) @ g[2 * j - 1])
check("(3) X_j = (prod_(k<j) Z_k) g_(2j-1), a monomial of degree 2j-1 (n=6)", ok)
f = find_monomial(op(5, {1: "Z", 2: "Z", 3: "Z"}), majoranas_x_onsite(5))
check("(4) X-on-site convention: Z2Z3Z4 = -g1 g2 g3 g6 g7 (degree 5)",
      f is not None and f[0] == (1, 2, 3, 6, 7) and f[1] == -1, str(f))
gx = majoranas_x_onsite(4)
check("(4) in the X-on-site convention X_j is quadratic",
      all(find_monomial(op(4, {j - 1: "X"}), gx, max_degree=2) is not None for j in range(1, 5)))

# ------------------------------------------------------------------ Proposition 2.3
section("Proposition 2.3  ZZZ circuits (Exercise 7, five parts)")
for n in range(3, 9):
    G = [pmask(n, {j: "Z", j + 1: "Z", j + 2: "Z"}) for j in range(n - 2)]
    d, _ = lie_closure_dim(G)
    check(f"n={n}: (1) all generators odd weight; (2) DLA abelian of dimension n-2 = {n - 2}",
          all(weight(label_of(p, n)) == 3 for p in G) and is_abelian(G) and d == n - 2)
n = 5
P0 = op(n, {0: "Z", 1: "Z", 2: "Z"})
check("(1) the rotations are J-fixed for random theta (n=5)",
      all(np.allclose(J_apply_conj(n, rot(P0, th)), rot(P0, th)) for th in rng.uniform(0, 6, 5)))
ok = True
for _ in range(10):
    psi = zero_state(n)
    tot = 0.0
    for _ in range(30):
        j = rng.integers(0, n - 2); th = rng.uniform(0, 6)
        psi = rot(op(n, {j: "Z", j + 1: "Z", j + 2: "Z"}), th) @ psi
        tot += th
    ok &= np.isclose(abs(psi[0]) ** 2, 1) and np.isclose(psi[0], np.exp(-1j * tot / 2))
check("(3) every circuit maps |0...0> to exp(-i sum theta/2)|0...0>: output 0...0 with probability 1", ok)
g = majoranas_z_onsite(n)
quad = [g[r] @ g[s] for r, s in it.combinations(range(1, 2 * n + 1), 2)]
target = op(n, {0: "Z", 1: "Z", 2: "Z"})
V = np.array([np.concatenate([q.real.ravel(), q.imag.ravel()]) for q in quad]).T
t = np.concatenate([target.real.ravel(), target.imag.ravel()])
coef, res, *_ = np.linalg.lstsq(V, t, rcond=None)
check("(4) Z1Z2Z3 is not in the real span of the quadratic monomials (least-squares residual > 0)",
      np.linalg.norm(V @ coef - t) > 1e-6, f"residual {np.linalg.norm(V @ coef - t):.3f}")
note("(5) a ℚ-structured family that is efficiently sampleable (part 3) but not quadratic (part 4) "
     "refutes 'free fermions are the only tractable case'")

raise SystemExit(summary("Section 2"))
