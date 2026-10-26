"""Concrete molecular descriptors for the generated molecules (auxiliary script).

Reads the "case<TAB>smiles" files written by eval_dataset.py and reports
standard RDKit drug-likeness descriptors plus the repository's own ring
statistics (EPMolGen/tools/static_ring.py: substructure).

Usage:
    python sdaa/scripts/metrics_desc.py sdaa/scripts/smiles_sdaa.txt sdaa/scripts/smiles_cuda.txt
"""

import os
import statistics
import sys
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
try:
    from EPMolGen.tools.static_ring import substructure
except Exception:
    substructure = None

try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    from SA_Score import sascorer
except Exception:
    sascorer = None


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


def describe(mol):
    d = {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "MW": Descriptors.MolWt(mol),
        "logP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "rings": Lipinski.RingCount(mol),
        "arom_rings": Lipinski.NumAromaticRings(mol),
        "Fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "QED": QED.qed(mol),
    }
    if sascorer is not None:
        d["SA"] = sascorer.calculateScore(mol)
    return d


def main():
    sides = [(os.path.basename(p).replace("smiles_", "").replace(".txt", ""), load(p))
             for p in sys.argv[1:]]
    keys = ["QED", "SA", "logP", "TPSA", "MW", "heavy_atoms", "HBD", "HBA", "RotB",
            "rings", "arom_rings", "Fsp3"]

    table = {}
    for name, per_case in sides:
        mols, rows = [], []
        for case in sorted(per_case):
            for smi in per_case[case]:
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                mols.append(m)
                r = describe(m)
                r["case"] = case
                r["smiles"] = smi
                rows.append(r)
        table[name] = (rows, mols)
        print("%s: %d molecules, %d cases" % (name, len(rows), len(per_case)))

    names = [n for n, _ in sides]
    print("\n%-14s %16s %16s %8s" % ("metric", names[0], names[1] if len(names) > 1 else "-", "diff%"))
    for k in keys:
        vals, detail = [], []
        for n in names:
            v = [r[k] for r in table[n][0] if k in r]
            if v:
                vals.append(statistics.mean(v))
                detail.append("%8.3f+-%.3f" % (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0))
            else:
                vals.append(float("nan"))
                detail.append("       n/a")
        diff = (vals[0] - vals[1]) / vals[1] * 100 if len(vals) > 1 and vals[1] == vals[1] else float("nan")
        if len(vals) > 1:
            print("%-14s %16s %16s %7.1f%%" % (k, detail[0], detail[1], diff))
        else:
            print("%-14s %16s" % (k, detail[0]))

    if substructure is not None:
        for n in names:
            rows, mols = table[n]
            try:
                stat = substructure([mols])
            except Exception as e:
                print("\n%s: ring statistics unavailable (%s)" % (n, e))
                continue
            rings = {k: v for k, v in stat.items() if k != "sssr"}
            sssr = stat.get("sssr", {})
            sssr_num = [v["num"] for v in sssr.values()] if isinstance(sssr, dict) else []
            sssr_rate = [v["rate"] for v in sssr.values()] if isinstance(sssr, dict) else []
            print("\n%s ring statistics (repository metric)" % n)
            for k, v in rings.items():
                print("  %-16s num=%-4s rate=%.3f" % (k, v.get("num"), v.get("rate", 0.0)))
            if sssr_num:
                print("  %-16s mean=%.2f max=%d (molecules with that SSSR size: %d)"
                      % ("sssr", statistics.mean(sssr_num), max(sssr_num), len(sssr_num)))


if __name__ == "__main__":
    main()
