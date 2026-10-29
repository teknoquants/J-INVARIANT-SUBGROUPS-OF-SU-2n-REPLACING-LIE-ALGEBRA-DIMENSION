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

"""Section 3 -- The simulability dichotomy (Problem 3) and its answer.

Theorem 3.2 (known results, demonstrated), Statement 3.3, Lemma 3.4, Proposition 3.5, Theorem 3.6,
Theorem 3.7, Lemma 3.8, Theorem 3.9, Corollary 3.10, Theorem 3.11, Theorem 3.12, Question 3.13.
"""
import itertools as it
import numpy as np
from common import (check, note, section, summary, op, pauli_string, all_pauli_labels, weight, J_apply_conj,
                    mask, lie_closure_dim, is_abelian, label_of, majoranas_z_onsite, rot, expm_herm,
                    zero_state, pmask, pmat, module_dim)

rng = np.random.default_rng(303)

# ------------------------------------------------------------------ Pauli-propagation helpers (for Theorem 3.2(2))
_MUL = {("I", "I"): (1, "I"), ("I", "X"): (1, "X"), ("I", "Y"): (1, "Y"), ("I", "Z"): (1, "Z"),
        ("X", "I"): (1, "X"), ("X", "X"): (1, "I"), ("X", "Y"): (1j, "Z"), ("X", "Z"): (-1j, "Y"),
        ("Y", "I"): (1, "Y"), ("Y", "X"): (-1j, "Z"), ("Y", "Y"): (1, "I"), ("Y", "Z"): (1j, "X"),
        ("Z", "I"): (1, "Z"), ("Z", "X"): (1j, "Y"), ("Z", "Y"): (-1j, "X"), ("Z", "Z"): (1, "I")}


def pmul(a, b):
    ph, out = 1, ""
    for x, y in zip(a, b):
        p, c = _MUL[(x, y)]
        ph *= p; out += c
    return ph, out


def anticomm(a, b):
    return sum(1 for x, y in zip(a, b) if x != "I" and y != "I" and x != y) % 2 == 1


def heisenberg(obs, gates):
    """U^dag O U for U = prod gates (applied left to right in time), O = dict label->coef."""
    O = dict(obs)
    for P, th in reversed(gates):            # last gate acts first in the Heisenberg picture
        new = {}
        for Q, c in O.items():
            if anticomm(P, Q):
                ph, R = pmul(P, Q)
                new[Q] = new.get(Q, 0) + c * np.cos(th)
                new[R] = new.get(R, 0) + c * 1j * np.sin(th) * ph
            else:
                new[Q] = new.get(Q, 0) + c
        O = {k: v for k, v in new.items() if abs(v) > 1e-14}
    return O


def expval_zero(O):
    return sum(c for Q, c in O.items() if set(Q) <= {"I", "Z"})


# ------------------------------------------------------------------ Theorem 3.2
section("Theorem 3.2 (known results, demonstrated)")
# (1) matchgate circuit: covariance propagation (2n x 2n) reproduces the statevector
n = 4; g = majoranas_z_onsite(n); dim = 2 * n
psi = zero_state(n)
cov0 = np.array([[np.real(1j / 2 * np.vdot(psi, (g[a] @ g[b] - g[b] @ g[a]) @ psi)) for b in range(1, dim + 1)]
                 for a in range(1, dim + 1)])
R_total = np.eye(dim)
for _ in range(25):
    r, s = sorted(rng.choice(np.arange(1, dim + 1), 2, replace=False)); th = rng.uniform(0, 2 * np.pi)
    U = np.cos(th / 2) * np.eye(2 ** n) + np.sin(th / 2) * g[r] @ g[s]
    R = np.eye(dim); R[r - 1, r - 1] = R[s - 1, s - 1] = np.cos(th); R[r - 1, s - 1] = np.sin(th); R[s - 1, r - 1] = -np.sin(th)
    psi = U @ psi; R_total = R @ R_total
cov_sv = np.array([[np.real(1j / 2 * np.vdot(psi, (g[a] @ g[b] - g[b] @ g[a]) @ psi)) for b in range(1, dim + 1)]
                   for a in range(1, dim + 1)])
check("(1) matchgates: 25-gate circuit, 8x8 covariance propagation = 16-dim statevector",
      np.allclose(R_total @ cov0 @ R_total.T, cov_sv))
# (2) Lie-algebraic expectation values: Heisenberg evolution of O in g stays inside g
n = 6
gens = ["X" + "I" * (n - 1)] + ["I" * j + "ZZ" + "I" * (n - j - 2) for j in range(n - 1)] + \
       ["I" * j + "X" + "I" * (n - j - 1) for j in range(1, n)]
dimg, basis = lie_closure_dim([mask(l) for l in gens])
basis_labels = {label_of(b, n) for b in basis}
gates = [(gens[rng.integers(len(gens))], rng.uniform(0, 3)) for _ in range(40)]
O = {"I" * 2 + "ZZ" + "I" * (n - 4): 1.0}
OH = heisenberg(O, gates)
U = np.eye(2 ** n, dtype=complex)
for P, th in gates:
    U = expm_herm(pauli_string(P), th / 2) @ U
psi = U @ zero_state(n)
exact = np.vdot(psi, pauli_string(list(O)[0]) @ psi).real
check(f"(2) g-sim style: <O> from Heisenberg evolution inside g (dim {dimg}) matches statevector",
      np.isclose(expval_zero(OH).real, exact) and set(OH) <= basis_labels, f"{exact:.6f}")
note("Theorem 3.2 cites [3]-[6]; the checks above demonstrate the two mechanisms on small instances")

# ------------------------------------------------------------------ Statement 3.3
section("Statement 3.3  (Problem 3 as posed)")
note("Statement 3.3 is the question itself; it is answered by Theorems 3.7, 3.9 and 3.11 below")

# ------------------------------------------------------------------ Lemma 3.4
section("Lemma 3.4  degree rule for Majorana monomials")
n = 6; g = majoranas_z_onsite(n)
ok = True
for _ in range(300):
    p, q = rng.choice([1, 2, 3, 4, 5, 6], 2)
    A = tuple(sorted(rng.choice(np.arange(1, 2 * n + 1), p, replace=False)))
    B = tuple(sorted(rng.choice(np.arange(1, 2 * n + 1), q, replace=False)))
    s = len(set(A) & set(B))
    MA = np.linalg.multi_dot([g[a] for a in A]) if p > 1 else g[A[0]]
    MB = np.linalg.multi_dot([g[b] for b in B]) if q > 1 else g[B[0]]
    sign = (-1) ** ((p * q - s) % 2)
    ok &= np.allclose(MA @ MB, sign * MB @ MA)
    sym = tuple(sorted(set(A) ^ set(B)))
    G = (np.linalg.multi_dot([g[r] for r in sym]) if len(sym) > 1 else g[sym[0]]) if sym else np.eye(2 ** n)
    ok &= any(np.allclose(MA @ MB, ph * G) for ph in (1, -1, 1j, -1j)) and len(sym) == p + q - 2 * s
check("300 random pairs: g_A g_B = (-1)^(pq-s) g_B g_A and g_A g_B = ± g_(AΔB) (degree p+q-2s)", ok)
A = (1, 2, 3, 4, 5, 6); B = (6, 7, 8, 9, 10, 11)
MA = np.linalg.multi_dot([g[a] for a in A]); MB = np.linalg.multi_dot([g[b] for b in B])
C = MA @ MB - MB @ MA
G10 = np.linalg.multi_dot([g[r] for r in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)])
check("Corollary: two degree-6 monomials sharing one index have bracket ±2 g of degree 10",
      any(np.allclose(C, ph * G10) for ph in (2, -2, 2j, -2j)))
quad = [g[r] @ g[s] for r, s in it.combinations(range(1, 2 * n + 1), 2)]
V = np.array([np.concatenate([x.real.ravel(), x.imag.ravel()]) for x in quad])
br = [x @ y - y @ x for x, y in it.combinations(quad[:40], 2)]
W = np.vstack([V] + [np.concatenate([b.real.ravel(), b.imag.ravel()])[None] for b in br])
check("Corollary: quadratic monomials close under brackets (span dimension n(2n-1) unchanged)",
      np.linalg.matrix_rank(V) == np.linalg.matrix_rank(W) == n * (2 * n - 1))

# ------------------------------------------------------------------ Proposition 3.5
section("Proposition 3.5  |x><x| = 2^-n sum_S (-1)^(x.S) Z_S, all even degrees")
n = 4; g = majoranas_z_onsite(n)
for x in [(0, 0, 0, 0), (1, 0, 1, 1), (1, 1, 1, 1)]:
    idx = int("".join(map(str, x)), 2)
    proj = np.zeros((2 ** n, 2 ** n)); proj[idx, idx] = 1
    recon = sum((-1) ** sum(x[j] for j in S) * op(n, {j: "Z" for j in S})
                for r in range(n + 1) for S in it.combinations(range(n), r)) / 2 ** n
    check(f"x = {x}: projector equals the 2^n-term Z_S expansion", np.allclose(proj, recon))
ok = all(np.allclose(op(n, {j: "Z" for j in S}),
                     (-1j) ** len(S) * (np.linalg.multi_dot([g[2 * j + 1] @ g[2 * j + 2] for j in S]) if len(S) > 1
                                       else (g[2 * S[0] + 1] @ g[2 * S[0] + 2] if S else np.eye(2 ** n))))
         for r in range(n + 1) for S in it.combinations(range(n), r))
check("Z_S = (-i)^|S| prod_(j in S) g_(2j-1) g_(2j): degree 2|S|, all degrees 0..2n occur", ok)

# ------------------------------------------------------------------ Theorem 3.6
section("Theorem 3.6  block factorisation for disjoint supports")
blocks = [(0, 1), (2, 3, 4), (5,)]
n = 6
gens = []
for S in blocks:
    for c in "XYZ":
        gens.append({S[0]: c})
    if len(S) > 1:
        gens.append({S[0]: "Z", S[1]: "X", **({S[2]: "Z"} if len(S) > 2 else {})})
U = np.eye(2 ** n, dtype=complex); Us = {S: np.eye(2 ** len(S), dtype=complex) for S in blocks}
for _ in range(40):
    gd = gens[rng.integers(len(gens))]; th = rng.uniform(0, 6)
    U = rot(op(n, gd), th) @ U
    S = next(B for B in blocks if set(gd) <= set(B))
    Us[S] = rot(op(len(S), {S.index(k): v for k, v in gd.items()}), th) @ Us[S]
check("(1) U = tensor product of block unitaries", np.allclose(U, np.kron(np.kron(Us[blocks[0]], Us[blocks[1]]), Us[blocks[2]])))
P = np.abs(U[:, 0]) ** 2; Pf = np.ones(1)
for S in blocks:
    Pf = np.kron(Pf, np.abs(Us[S][:, 0]) ** 2)
check("(1) output distribution = product of block distributions", np.allclose(P, Pf))
dg, _ = lie_closure_dim([pmask(n, gd) for gd in gens])
db = sum(lie_closure_dim([pmask(len(S), {S.index(k): v for k, v in gd.items()}) for gd in gens if set(gd) <= set(S)])[0]
         for S in blocks)
check("(2) dim g = sum of block dimensions <= sum 4^|S_k|", dg == db <= sum(4 ** len(S) for S in blocks), f"{dg}")
XX, ZZ = op(2, {0: "X", 1: "X"}), op(2, {0: "Z", 1: "Z"})
U = rot(XX, 1.4) @ rot(ZZ, 2.2); P = np.abs(U[:, 0]) ** 2
p1 = np.array([P[0] + P[1], P[2] + P[3]]); p2 = np.array([P[0] + P[2], P[1] + P[3]])
check("Remark: XX and ZZ commute but their output does not factorise over qubits",
      np.allclose(XX @ ZZ, ZZ @ XX) and not np.allclose(P, np.kron(p1, p2)), str(np.round(P, 3)))

# ------------------------------------------------------------------ Theorem 3.7
section("Theorem 3.7  exponential DLA, exact O(L) sampling (necessity fails)")


def family_A(n):
    t = n - 1
    return [pmask(n, {t: "X"}), pmask(n, {t: "Y"})] + [pmask(n, {i: "Z", i + 1: "Z", t: "X"}) for i in range(n - 2)]


for n in range(3, 13):
    G = family_A(n)
    d, B = lie_closure_dim(G)
    check(f"n={n}: (1) {len(G)} odd-weight generators; (2) dim g = 3*2^(n-2) = {3 * 2 ** (n - 2)}",
          d == 3 * 2 ** (n - 2) and all(weight(label_of(p, n)) % 2 == 1 for p in G))
n = 8; d, B = lie_closure_dim(family_A(n))
cm = (1 << (n - 1)) - 1
check("(2) every DLA element is Z^s (x) sigma_target with s of even weight (n=8)",
      all((p[0] & cm) == 0 and bin(p[1] & cm).count("1") % 2 == 0 and ((p[0] >> (n - 1)) | (p[1] >> (n - 1))) for p in B))
n = 6; G = family_A(n); mats = [pmat(p, n) for p in G]
Zs = [pmat(p, n) for p in B if True][:0]
# structure: g = A (x) su(2), A commutative of dim |E| = 2^(n-2)
Es = {(p[1] & ((1 << (n - 1)) - 1)) for p in lie_closure_dim(G)[1]}
check("(2) structure: control patterns form a group E of size 2^(n-2) (so g ≅ su(2)^(⊕ 2^(n-2)))",
      len(Es) == 2 ** (n - 2) and all((a ^ b) in Es for a in Es for b in Es))
sig = {"X": pauli_string("X"), "Y": pauli_string("Y")}
ok = True
for _ in range(20):
    psi = zero_state(n); phi = np.array([1, 0], complex)
    for _ in range(60):
        k = rng.integers(len(G)); th = rng.uniform(0, 2 * np.pi)
        psi = rot(mats[k], th) @ psi
        phi = rot(sig["Y" if k == 1 else "X"], th) @ phi
    pred = np.zeros(2 ** n, complex); pred[0], pred[1] = phi[0], phi[1]
    ok &= np.allclose(psi, pred)
check("(3) 20 random 60-gate circuits: U|0..0> = |0..0>_controls (x) V|0>_target exactly (n=6)", ok)
check("(1) the generators' rotations are J-fixed (n=4)",
      all(np.allclose(J_apply_conj(4, rot(pmat(p, 4), 0.9)), rot(pmat(p, 4), 0.9)) for p in family_A(4)))

# ------------------------------------------------------------------ Lemma 3.8
section("Lemma 3.8  parity padding")
ok_odd = ok_dist = ok_dim = True
for _ in range(40):
    n = int(rng.integers(2, 5)); k = int(rng.integers(2, 6))
    labs = [l for l in ("".join(rng.choice(list("IXYZ"), n)) for _ in range(k)) if l != "I" * n] or ["X" + "I" * (n - 1)]
    pad = [l + ("Z" if weight(l) % 2 == 0 else "I") for l in labs]
    ok_odd &= all(weight(l) % 2 == 1 for l in pad)
    d0, _ = lie_closure_dim([mask(l) for l in labs]); d1, _ = lie_closure_dim([mask(l) for l in pad])
    ok_dim &= d0 <= d1 <= 2 * d0
    a, b = zero_state(n), zero_state(n + 1)
    for _ in range(25):
        j = rng.integers(len(labs)); th = rng.uniform(0, 6)
        a = rot(pauli_string(labs[j]), th) @ a; b = rot(pauli_string(pad[j]), th) @ b
    Pb = (np.abs(b) ** 2).reshape(2 ** n, 2)
    ok_dist &= np.allclose(Pb[:, 1], 0) and np.allclose(Pb[:, 0], np.abs(a) ** 2)
check("(1) 40 random sets: every padded generator has odd weight", ok_odd)
check("(2) padded circuit reproduces the original distribution; ancilla always 0", ok_dist)
check("(3) dim g(G) <= dim g(G') <= 2 dim g(G)", ok_dim)

# ------------------------------------------------------------------ Theorem 3.9
section("Theorem 3.9  padded IQP: abelian DLA n(n+1)/2, identical output")
for n in (3, 4, 5):
    terms = [{i: "X"} for i in range(n)] + [{i: "X", j: "X"} for i, j in it.combinations(range(n), 2)]
    padded = [{**t, **({n: "Z"} if len(t) == 2 else {})} for t in terms]
    G = [pmask(n + 1, t) for t in padded]
    d, _ = lie_closure_dim(G)
    check(f"n={n}: (1) odd weights, commuting, abelian DLA of dim n(n+1)/2 = {n * (n + 1) // 2}",
          all(len(t) % 2 == 1 for t in padded) and is_abelian(G) and d == n * (n + 1) // 2)
    w = [int(rng.integers(0, 8)) for _ in terms]
    Hx = sum(w[k] * np.pi / 8 * op(n, terms[k]) for k in range(len(terms)))
    Hp = sum(w[k] * np.pi / 8 * op(n + 1, padded[k]) for k in range(len(terms)))
    U, Up = expm_herm(-Hx, 1.0), expm_herm(-Hp, 1.0)
    P = np.abs(U[:, 0]) ** 2; Pp = (np.abs(Up[:, 0]) ** 2).reshape(2 ** n, 2)
    check(f"n={n}: (2) padded circuit = Ising IQP distribution, ancilla = 0", np.allclose(Pp[:, 0], P) and np.allclose(Pp[:, 1], 0))
    Zs = sum(np.exp(1j * np.pi / 8 * sum(w[k] * np.prod([z[i] for i in terms[k]]) for k in range(len(terms))))
             for z in it.product([1, -1], repeat=n))
    check(f"n={n}: (2) <0|C|0> = Z(e^(i pi/8))/2^n, the Ising partition function of [13]", np.isclose(U[0, 0], Zs / 2 ** n))
# Walsh expansion of an arbitrary 2-qubit X-diagonal gate
H2 = np.kron(np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.array([[1, 1], [1, -1]]) / np.sqrt(2))
ok = True
for _ in range(20):
    ph = rng.uniform(0, 2 * np.pi, 4)
    D = H2 @ np.diag(np.exp(1j * ph)) @ H2
    zz = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    coef = np.linalg.solve(np.column_stack([np.ones(4), zz[:, 0], zz[:, 1], zz[:, 0] * zz[:, 1]]), ph)
    a, b, c, d = coef
    D2 = np.exp(1j * a) * expm_herm(-(b * op(2, {0: "X"}) + c * op(2, {1: "X"}) + d * op(2, {0: "X", 1: "X"})), 1.0)
    ok &= np.allclose(D, D2)
check("(2) every 2-qubit gate diagonal in the X basis = e^(i a) exp(i(b X_j + c X_k + d X_jX_k)) (20 random)", ok)
note("(3)-(4) hardness: Theorem 2 / Corollary 1 of [12] and Theorem 1 of [13] (Conjecture 2); cited, not computable")

# ------------------------------------------------------------------ Corollary 3.10
section("Corollary 3.10  conjugate DLAs, opposite sampling behaviour")
n = 4
GX = [pmask(n + 1, {**{i: "X"}}) for i in range(n)] + [pmask(n + 1, {i: "X", j: "X", n: "Z"}) for i, j in it.combinations(range(n), 2)]
GZ = [pmask(n + 1, {**{i: "Z"}}) for i in range(n)] + [pmask(n + 1, {i: "Z", j: "Z", n: "Z"}) for i, j in it.combinations(range(n), 2)]
check("both sets abelian with equal DLA dimension", is_abelian(GX) and is_abelian(GZ)
      and lie_closure_dim(GX)[0] == lie_closure_dim(GZ)[0])
Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
Hn = np.kron(np.kron(np.kron(np.kron(Hd, Hd), Hd), Hd), np.eye(2))
check("H^(x)n (on the first n qubits) conjugates every X-type generator to its Z-type partner",
      all(np.allclose(Hn @ pmat(a, n + 1) @ Hn, pmat(b, n + 1)) for a, b in zip(GX, GZ)))
psi = zero_state(n + 1)
for p in GZ:
    psi = rot(pmat(p, n + 1), rng.uniform(0, 6)) @ psi
check("Z-type circuits output 0...0 with probability 1", np.isclose(abs(psi[0]) ** 2, 1))

# ------------------------------------------------------------------ Theorem 3.11
section("Theorem 3.11  answer to Problem 3")
note("part 1 = Theorem 3.7 (checked above, unconditional); part 2 = Theorem 3.9 (output equality checked "
     "above; hardness cited from [12], [13])")

# ------------------------------------------------------------------ Theorem 3.12
section("Theorem 3.12  code criterion")
for trial in range(5):
    n = 7
    c1, c2 = rng.integers(0, 2, n), rng.integers(0, 2, n)
    labs = []
    for xv in (c1, c2, (c1 + c2) % 2):
        for _ in range(3):
            zv = rng.integers(0, 2, n)
            labs.append("".join({(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[(int(a), int(b))] for a, b in zip(xv, zv)))
    labs = [l for l in labs if l != "I" * n]
    psi = zero_state(n)
    for _ in range(40):
        psi = rot(pauli_string(labs[rng.integers(len(labs))]), rng.uniform(0, 6)) @ psi
    code = {tuple(np.zeros(n, int)), tuple(c1), tuple(c2), tuple((c1 + c2) % 2)}
    support = {tuple(int(b) for b in format(i, f"0{n}b")) for i in np.nonzero(np.abs(psi) > 1e-12)[0]}
    d, _ = lie_closure_dim([mask(l) for l in labs])
    check(f"trial {trial}: dim g = {d}, yet the state stays in span{{|c>: c in C}} with |C| = 4", support <= code)
check("Theorem 3.7 is the case C = span{e_target}: X-parts of its generators all equal e_target",
      len({p[0] for p in family_A(6)}) == 1)

# ------------------------------------------------------------------ Question 3.13
section("Question 3.13  data for the module generated by rho_0")
for n in (3, 4, 5):
    dA = module_dim(family_A(n), n)
    dZ = module_dim([pmask(n, {j: "Z", j + 1: "Z", j + 2: "Z"}) for j in range(n - 2)], n)
    dX = module_dim([pmask(n, {i: "X"}) for i in range(n)] +
                    [pmask(n, {i: "X", j: "X", k: "X"}) for i, j, k in it.combinations(range(n), 3)], n)
    dP = module_dim([pmask(n, {i: "X"}) for i in range(n)], n)
    check(f"n={n}: table row values: Theorem 3.7 = 4, ZZZ = 1, odd X-program = {2 ** (2 * n - 1) + 1}, product = 3^n = {3 ** n}",
          dA == 4 and dZ == 1 and dX == 2 ** (2 * n - 1) + 1 and dP == 3 ** n, f"{dA}, {dZ}, {dX}, {dP}")
note("Question 3.13 itself is open; these numbers are the evidence table in Section 3.2")

raise SystemExit(summary("Section 3"))
