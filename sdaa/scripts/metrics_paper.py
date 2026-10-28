"""Reproduce the paper's official evaluation metrics for EPMolGen.

Table 1 (validity / QED / LogP / SA) and Table 2 (uncommon ring proportions)
as reported in Nature Communications 2026, "Deep learning-assisted discovery of
a potent and cell-active inhibitor of RNA N6-methyladenosine recognition
protein YTHDC2" (doi 10.1038/s41467-025-65542-0).

Usage:
  python metrics_paper.py --sdf  <file.sdf>            # molecules with 3D coords
  python metrics_paper.py --smiles <file.smi>          # one SMILES per line
  python metrics_paper.py --extra <file.smi>           # extra SMILES (the ones the
                                                      # repo rejected) are parsed
                                                      # and counted as chemically valid
"""
import argparse
import os
import statistics
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


def _load_sascorer():
    """Ertl & Schufenhauer SA score.

    RDKit ships it under Contrib/SA_Score, but that folder has no __init__.py,
    so it must be imported as a top-level module with its own directory on
    sys.path.  The container build of RDKit omits Contrib entirely, so a
    vendored copy can sit next to this script.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "SA_Score"),
                  os.path.join(here, "..", "SA_Score")]
    try:
        from rdkit.Chem import RDConfig
        contrib = RDConfig.RDContribDir
        candidates += [os.path.join(contrib, "SA_Score"),
                       os.path.join(contrib, "Contrib", "SA_Score")]
    except Exception:
        pass
    for base in candidates:
        if not os.path.isdir(base):
            continue
        if base not in sys.path:
            sys.path.insert(0, base)
        try:
            import sascorer
            return sascorer
        except Exception:
            continue
    return None


def read_sdf(path):
    mols = []
    for mol in Chem.SDMolSupplier(path, removeHs=False, sanitize=True):
        if mol is not None:
            mols.append(mol)
    return mols


def read_smiles(path):
    """Read SMILES, one per line.

    Lines may be "<case>\\t<SMILES>" (the format eval_dataset.py writes) or a
    bare SMILES.
    """
    mols = []
    n_bad = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            line = line.split("\t", 1)[1].strip()
        m = Chem.MolFromSmiles(line)
        if m is None:
            n_bad += 1
        else:
            mols.append(m)
    return mols, n_bad


def ring_stats(mols):
    """Table 2 style statistics.

    three-membered rings : a ring of size 3
    eight-membered rings  : a ring of size 8
    condensed rings      : >= 4 fused rings in one connected ring system
    """
    three = eight = condensed = 0
    size_hist = {}
    for m in mols:
        ri = m.GetRingInfo()
        sizes = [len(r) for r in ri.AtomRings()]
        for s in sizes:
            size_hist[s] = size_hist.get(s, 0) + 1
        if 3 in sizes:
            three += 1
        if 8 in sizes:
            eight += 1
        # fused ring systems: count rings per connected ring-system
        systems = []
        for r in ri.AtomRings():
            aset = set(r)
            placed = False
            for sysset in systems:
                if sysset & aset:
                    sysset |= aset
                    placed = True
                    break
            if not placed:
                systems.append(set(aset))
        # merge systems that became adjacent
        merged = True
        while merged:
            merged = False
            for i in range(len(systems)):
                for j in range(i + 1, len(systems)):
                    if systems[i] & systems[j]:
                        systems[i] |= systems[j]
                        systems.pop(j)
                        merged = True
                        break
                if merged:
                    break
        # a fused system of >=4 rings == >=4 rings sharing atoms
        fused_counts = []
        for sysset in systems:
            cnt = sum(1 for r in ri.AtomRings() if set(r) <= sysset)
            fused_counts.append(cnt)
        if fused_counts and max(fused_counts) >= 4:
            condensed += 1
    n = len(mols)
    return {
        "n": n,
        "three_pct": 100.0 * three / n if n else 0.0,
        "eight_pct": 100.0 * eight / n if n else 0.0,
        "condensed_pct": 100.0 * condensed / n if n else 0.0,
        "size_hist": size_hist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", default="")
    ap.add_argument("--smiles", default="")
    ap.add_argument("--extra", default="", help="SMILES the repo rejected; "
                                               "counted as chemically valid if RDKit parses them")
    ap.add_argument("--label", default="")
    ap.add_argument("--requested", type=int, default=0,
                    help="number of molecules requested from the model")
    args = ap.parse_args()

    sascorer = _load_sascorer()
    mols = []
    n_extra_bad = 0
    if args.sdf:
        mols = read_sdf(args.sdf)
    if args.smiles:
        m, bad = read_smiles(args.smiles)
        mols += m
    if args.extra:
        m, bad = read_smiles(args.extra)
        mols += m
        n_extra_bad = bad

    n = len(mols)
    if n == 0:
        print("no molecules")
        return

    qed = [QED.qed(m) for m in mols]
    logp = [Crippen.MolLogP(m) for m in mols]
    heavy = [Descriptors.HeavyAtomCount(m) for m in mols]
    sa = None
    if sascorer is not None:
        sa = [sascorer.calculateScore(m) for m in mols]

    def stat(v):
        return (statistics.mean(v), statistics.pstdev(v)) if len(v) > 1 else (v[0], 0.0)

    print("label            : %s" % (args.label or "-"))
    print("molecules        : %d" % n)
    if args.requested:
        print("requested        : %d" % args.requested)
        print("Validity (%%)      : %.2f" % (100.0 * n / args.requested))
    q_m, q_s = stat(qed)
    l_m, l_s = stat(logp)
    print("QED              : %.2f +- %.2f" % (q_m, q_s))
    print("LogP             : %.2f +- %.2f" % (l_m, l_s))
    if sa is not None:
        s_m, s_s = stat(sa)
        print("SA               : %.2f +- %.2f" % (s_m, s_s))
    else:
        print("SA               : (sascorer unavailable)")
    h_m, h_s = stat(heavy)
    print("heavy atoms      : %.2f +- %.2f" % (h_m, h_s))

    rs = ring_stats(mols)
    print("--- Table 2 style ring statistics (%% of molecules) ---")
    print("three-membered   : %.2f" % rs["three_pct"])
    print("eight-membered   : %.2f" % rs["eight_pct"])
    print("condensed (>=4)  : %.2f" % rs["condensed_pct"])
    print("ring size hist   : %s" % dict(sorted(rs["size_hist"].items())))


if __name__ == "__main__":
    main()
