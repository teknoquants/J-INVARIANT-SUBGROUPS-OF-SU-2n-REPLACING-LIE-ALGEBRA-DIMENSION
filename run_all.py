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

"""Run every validation script for
'J-Invariant Subgroups of SU(2^n): Replacing Lie-Algebra Dimension' and print one summary.

Usage:  python run_all.py      (Python 3.9+, NumPy)
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    ("Section 1  Antiunitary framework", "section1_antiunitary.py"),
    ("Section 2  Dimension and fermionic degree", "section2_dimension_degree.py"),
    ("Section 3  Simulability and Problem 3", "section3_simulability.py"),
    ("Section 4  Free group factors", "section4_free_group_factors.py"),
    ("Section 5  Truncation and embeddings", "section5_truncation_embeddings.py"),
    ("Appendix A Errata of the earlier draft", "appendixA_errata.py"),
]
results = []
for title, script in SCRIPTS:
    t0 = time.time()
    p = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, capture_output=True, text=True)
    print(p.stdout)
    if p.stderr:
        print(p.stderr, file=sys.stderr)
    last = [l for l in p.stdout.splitlines() if "checks passed" in l]
    results.append((title, p.returncode == 0, last[-1].split(": ", 1)[-1] if last else "crashed", time.time() - t0))
print("=" * 78)
print("SUMMARY")
print("=" * 78)
for title, ok, line, dt in results:
    print(f"{'PASS' if ok else 'FAIL'}  {title:<42} {line:<26} {dt:6.1f}s")
sys.exit(0 if all(r[1] for r in results) else 1)
