"""Compare two forward dumps produced by forward_check.py (element-wise).

Also reports how close the sampling decisions are: the gap between the top-1
and top-2 probability tells how often a 1e-6 numerical difference can flip an
argmax / multinomial draw.

Usage:
    python sdaa/scripts/compare_forward.py a.npz b.npz
"""

import sys

import numpy as np


def bitwise_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool((np.ascontiguousarray(a).view(np.uint8)
                 == np.ascontiguousarray(b).view(np.uint8)).all())


def main():
    a = np.load(sys.argv[1])
    b = np.load(sys.argv[2])
    keys = [k for k in a.files if k in b.files]
    print("%-14s %-16s %12s %12s %12s %8s" % ("tensor", "shape", "max_abs", "mean_abs", "rel", "bitwise"))
    for k in keys:
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        if x.shape != y.shape:
            print("%-14s %-16s SHAPE MISMATCH %s vs %s" % (k, str(x.shape), y.shape))
            continue
        d = np.abs(x - y)
        scale = max(float(np.abs(x).max()), 1e-12)
        print("%-14s %-16s %12.3e %12.3e %12.3e %8s"
              % (k, str(x.shape), d.max(), d.mean(), d.max() / scale,
                 "YES" if bitwise_equal(a[k], b[k]) else "no"))

    # how often is the top-2 probability gap small enough for a 1e-6 difference to flip the pick?
    for name, key in (("pivotal_prob", "pivotal_prob"), ("pos_pi", "pos_pi")):
        if key not in keys:
            continue
        p = np.sort(a[key].astype(np.float64).ravel())[::-1]
        if len(p) < 2:
            continue
        gap = p[0] - p[1]
        print("\n%s: top1=%.6f top2=%.6f gap=%.3e" % (name, p[0], p[1], gap))
        for thr in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            frac = float((gap < thr).sum()) / len(gap) if len(gap) else 0.0
            print("  gap < %-6.0e : %6.2f%% of items" % (thr, frac * 100))


if __name__ == "__main__":
    main()
