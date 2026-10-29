# Validation scripts: J-Invariant Subgroups of SU(2ⁿ): Replacing Lie-Algebra Dimension

Roger Cooperman and A.E. Petrunina V.V. · Vienna · 29 October 2026 · MIT License (see `LICENSE`; every script also carries the full licence text in its header)

Python 3.9+ and NumPy only.

```
python run_all.py                    # every script, with a PASS/FAIL summary
python section3_simulability.py      # any single section
```

There is one script per section of the paper, plus one for Appendix A. Every line prints `[PASS]`, `[FAIL]` or `[NOTE]`. A `[NOTE]` marks a step that is cited or open and cannot be computed; it is not counted as a check. The `*_output.txt` files are logs of a complete run.

## Coverage: every numbered result

| Result | Script | How it is checked |
|---|---|---|
| Definition 1.1 | section1 | Used throughout; antilinearity and antiunitarity of J_n checked |
| Theorem 1.2 | section1 | J_n² = (−1)ⁿ I for n = 1–7; the proof step S_n·conj(S_n) = (−1)ⁿ I; involution only for even n |
| Definition 1.3 | section1 | K_n closed under products and inverses; exp(𝔨_n) ⊆ K_n |
| Lemma 1.4 | section1 | J P J⁻¹ = (−1)^w P for every Pauli string (n ≤ 3); J-fixed for all θ ⇔ w odd; θ = 2π counterexample showing the qualifier is needed; one-qubit identity |
| Corollary 1.5 | section1 | P commutes with J ⇔ w even; random even/odd Hamiltonians; dimension of the J-fixed part of su(2ⁿ) computed directly |
| Corollary 1.6 (and Exercise 10) | section1 | Kramers pairing for n = 1, 3, 5, with J v ⊥ v; simple spectrum for n = 4; −Z₁Z₂ degenerate; E₁ − E₀ = 0 for odd n |
| Lemma 2.1 (Exercise 3) | section2 | Count = formula = dim sp/so for n ≤ 10; binomial step; solution table; Lie closure; invariant form symmetric (so) or antisymmetric (sp) |
| Lemma 2.2 (1)–(4) | section2 | Majorana anticommutation; Z_j = −iγγ and the proof step; ZZZ has degree 6; X_j has degree 2j−1; X-on-site ZZZ has degree 5; X_j quadratic there |
| Proposition 2.3 (Exercise 7) | section2 | (1) odd weight, J-fixed; (2) abelian DLA of dimension n−2 for n = 3–8; (3) exact phase output from \|0…0⟩; (4) ZZZ not in the span of the quadratic monomials; (5) logical consequence (`[NOTE]`) |
| Definition 3.1 | section3 | Used in the checks below |
| Theorem 3.2 (cited) | section3 | Demonstrated: (1) matchgate covariance propagation = statevector; (2) Heisenberg evolution of an observable inside 𝔤 gives the exact expectation value |
| Statement 3.3 | section3 | `[NOTE]`: the question itself, answered by Theorems 3.7, 3.9, 3.11 |
| Lemma 3.4 and its Corollary | section3 | Sign and product rule on 300 random monomial pairs; degree-10 bracket of two degree-6 monomials; quadratic closure |
| Proposition 3.5 | section3 | Projector = 2ⁿ-term Z_S expansion; Z_S has Majorana degree 2\|S\| |
| Theorem 3.6 | section3 | Tensor factorisation of U and of the distribution; DLA = sum of the block DLAs; XX/ZZ remark |
| Theorem 3.7 | section3 | dim 𝔤 = 3·2ⁿ⁻² for n = 3–12; structure Z^s ⊗ σ with E a group of size 2ⁿ⁻²; 20 random 60-gate circuits match the single-qubit prediction exactly; J-fixed |
| Lemma 3.8 | section3 | 40 random generator sets: odd weight, identical distribution, dim 𝔤 ≤ dim 𝔤′ ≤ 2 dim 𝔤 |
| Theorem 3.9 | section3 | (1) abelian, dimension n(n+1)/2; (2) padded circuit = Ising IQP distribution; ⟨0\|C\|0⟩ = Ising partition function; Walsh expansion of X-diagonal 2-qubit gates. (3)–(4) hardness is cited from Bremner–Jozsa–Shepherd and Bremner–Montanaro–Shepherd (`[NOTE]`) |
| Corollary 3.10 | section3 | Both families abelian with equal dimension; conjugate by H^⊗n; Z-type output deterministic |
| Theorem 3.11 | section3 | `[NOTE]`: combines Theorems 3.7 and 3.9 |
| Theorem 3.12 | section3 | State supported on the code C (5 random sets); Theorem 3.7 is the case dim C = 1 |
| Question 3.13 (open) | section3 | Evidence table reproduced exactly: 4; 1; 2²ⁿ⁻¹+1; 3ⁿ |
| Definition 4.1 | section4 | Used below |
| Theorem 4.2 (cited) | section4 | Exact-rational consistency: group action, inverse, admissibility, Voiculescu case |
| Proposition 4.3 (Exercise 6) | section4 | (k+3)/4 exactly; ℍ ⊗ ℂ = M₂(ℂ); complexification identity on a finite-dimensional real group algebra |
| Remark 4.4 | section4 | ℍ is a division algebra; M₂(ℝ) has a nilpotent; both complexify to M₂(ℂ). The isomorphism question is open (`[NOTE]`) |
| Conjecture 4.5 (open) | section4 | d_β(N) from explicit bases; reasons the draft theorem is withdrawn (N² log N gap, (S₁,S₂,S₁) counterexample); asymptotic freeness for β = 1, 2, 4 |
| Question 4.6 (open) | section4 | `[NOTE]` |
| Definition 5.1 | section5 | Projection and encoding isometry |
| Proposition 5.2 | section5 | Exact expansion and positivity, 5 random trials |
| Question 5.3 (open) | section5 | `[NOTE]` |
| Proposition 5.4 | section5 | (1) J-symmetrisation removes exactly the odd-weight terms; (2) odd n: degenerate ground level, positive gap to the next level; (3) even n: both unique and degenerate ground levels occur |
| Conjecture 5.5 (open) | section5 | `[NOTE]` |
| Proposition 5.6 and Remark | section5 | Intertwiners J_n → J_{n+1}: only 0; J_n → J_{n+2}: nonzero; x ⊗ φ^⊗m is an isometric intertwiner for steps of 2 and 4; compatible forms constant along the chain |
| Conjecture 5.7 (open) | section5 | `[NOTE]` |
| Appendix A | appendixA_errata | Each error of the earlier draft reproduced (21 checks) |

## What computation cannot establish

- **Cited theorems:** the amplification formula (Theorem 4.2), the matchgate and g-sim results (Theorem 3.2), and the IQP hardness theorems behind Theorem 3.9(3)–(4). The scripts check their consistency and the exact reductions to them.
- **Open questions:** Questions 3.13, 4.6 and 5.3, and Conjectures 4.5, 5.5 and 5.7. The scripts reproduce only the supporting data stated in the paper.
- **Finite checks:** numerical checks at small n confirm identities and catch errors, but the general-n statements rest on the proofs in the paper.
