"""Compare generated SMILES between two platforms (case<TAB>smiles lists).

Usage:
    python sdaa/scripts/compare_smiles.py sdaa/scripts/smiles_sdaa.txt sdaa/scripts/smiles_cuda.txt
"""

import os
import sys
from collections import defaultdict


def load(path):
    per_case = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            case, smi = line.split("\t", 1)
            per_case[case].append(smi)
    return per_case


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = load(a_path), load(b_path)
    cases = sorted(set(a) | set(b))
    tot_same = tot_a = tot_b = 0
    print("%-8s %6s %6s %6s %8s" % ("case", "A", "B", "same", "rate"))
    for c in cases:
        la, lb = a.get(c, []), b.get(c, [])
        same = sum(1 for s in la if s in lb)
        tot_same += same
        tot_a += len(la)
        tot_b += len(lb)
        rate = same / max(len(la), 1)
        print("%-8s %6d %6d %6d %7.1f%%" % (c, len(la), len(lb), same, rate * 100))
    print("\nA = %s" % os.path.basename(a_path))
    print("B = %s" % os.path.basename(b_path))
    print("identical molecules: %d / %d (A) = %.1f%%" % (tot_same, tot_a, tot_same / max(tot_a, 1) * 100))


if __name__ == "__main__":
    main()
