"""Dataset-level sampling statistics for EPMolGen (auxiliary script).

Runs the repository's own sampling entry point (Generate.generate) on every
pocket of a case directory and collects the metrics the repository itself
computes (metrics.dir: Validity / Unique / ring_size).

The same script runs on CUDA and on SDAA; with a fixed seed the generated
SMILES lists can be compared across platforms.

Usage:
    python sdaa/scripts/eval_dataset.py --device sdaa:0 -n 10
    python sdaa/scripts/eval_dataset.py --device cuda:0 -n 10
"""

import argparse
import csv
import os
import re
import sys
import time

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from EPMolGen import epmolgen, Generate  # noqa: E402
from bench_infer import ATOMIC_NUMBERS, build_complex, peak_mem, reset_peak  # noqa: E402


def collect_cases(cases_dir):
    cases = []
    if "," in cases_dir:
        for one in cases_dir.split(","):
            cases.extend(collect_cases(one.strip()))
        return cases
    files = os.listdir(cases_dir)
    direct_pdb = [f for f in files if f.endswith("_pocket10.pdb")]
    direct_topo = [f for f in files if f.endswith("_topo.dict")]
    if direct_pdb and direct_topo:
        name = os.path.basename(os.path.normpath(cases_dir))
        return [(name, os.path.join(cases_dir, direct_pdb[0]), os.path.join(cases_dir, direct_topo[0]))]
    for name in sorted(files):
        d = os.path.join(cases_dir, name)
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        pdb = [f for f in files if f.endswith("_pocket10.pdb")]
        topo = [f for f in files if f.endswith("_topo.dict")]
        if pdb and topo:
            cases.append((name, os.path.join(d, pdb[0]), os.path.join(d, topo[0])))
    return cases


def latest_dir(root, name):
    d = os.path.join(root, name)
    subs = sorted(os.path.join(d, x) for x in os.listdir(d))
    return subs[-1] if subs else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("-n", "--molecule_number", type=int, default=10)
    ap.add_argument("--cases", default="./Ten_test_cases")
    ap.add_argument("--ckpt", default="./model_ckpt.pt")
    ap.add_argument("--root_path", default="gen_results_dataset")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    cases = collect_cases(args.cases)
    print("device=%s  cases=%d  molecules/case=%d  seed=%d"
          % (args.device, len(cases), args.molecule_number, args.seed), flush=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = epmolgen(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    print("model device: %s" % next(model.parameters()).device, flush=True)
    try:
        import torch_sdaa
        print("rng align_nv_device: %s  (TORCH_SDAA_ALIGN_NV_DEVICE=%s)"
              % (torch_sdaa.backend.random.random_align_nv_device(),
                 os.environ.get("TORCH_SDAA_ALIGN_NV_DEVICE")), flush=True)
    except Exception:
        pass

    rows = []
    smiles_all = []
    for idx, (name, pdb, topo) in enumerate(cases):
        torch.manual_seed(args.seed + idx)
        # the per-molecule atom count comes from numpy's global RNG (EPMolGen/sample.py:27),
        # so numpy has to be seeded as well, otherwise runs are not reproducible at all
        np.random.seed(args.seed + idx)
        t0 = time.time()
        cpx, graph_compose = build_complex(pdb, topo)
        sample = Generate(
            model, graph_compose.run, temperature=[1.0, 1.0], atom_type_map=ATOMIC_NUMBERS,
            num_bond_type=4, num_atom_max=35, pivotal_threshold=0.5, max_double_in_6ring=0,
            min_dist_inter_mol=3.0, bond_length_range=(1.0, 2.0), choose_max=1,
            device=args.device,
        )
        reset_peak(device)
        sample.generate(cpx, molecule_number=args.molecule_number, rec_name=name,
                        print_SMILES=False, root_path=args.root_path)
        wall = time.time() - t0

        out_dir = latest_dir(args.root_path, name)
        with open(os.path.join(out_dir, "metrics.dir")) as f:
            raw = f.read()
        # metrics.dir is a str(dict) whose ring_size block holds rdkit reprs -> not literal_eval-able
        validity = float(re.search(r"'Validity':\s*([0-9.]+)", raw).group(1))
        unique = float(re.search(r"'Unique':\s*([0-9.]+)", raw).group(1))
        metrics = {"Validity": validity, "Unique": unique}
        smi_path = os.path.join(out_dir, "generated.smi")
        smiles = [s.strip() for s in open(smi_path)] if os.path.exists(smi_path) else []
        heavy = []
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                heavy.append(m.GetNumHeavyAtoms())

        row = {
            "case": name,
            "requested": args.molecule_number,
            "valid": int(round(metrics["Validity"] * args.molecule_number)),
            "unique": int(round(metrics["Unique"] * max(metrics["Validity"] * args.molecule_number, 1))),
            "validity": round(metrics["Validity"], 4),
            "unique_rate": round(metrics["Unique"], 4),
            "heavy_atoms_avg": round(sum(heavy) / len(heavy), 1) if heavy else 0,
            "wall_s": round(wall, 1),
            "s_per_mol": round(wall / args.molecule_number, 2),
            "peak_mem_MiB": round(peak_mem(device) / 1024 ** 2, 1),
            "out_dir": out_dir,
        }
        rows.append(row)
        for s in smiles:
            smiles_all.append("%s\t%s" % (name, s))
        print("[%2d/%2d] %-6s valid=%d/%d unique=%d heavy_atoms=%.1f wall=%.1fs (%.2fs/mol) peak=%.1fMiB"
              % (idx + 1, len(cases), name, row["valid"], row["requested"], row["unique"],
                 row["heavy_atoms_avg"], row["wall_s"], row["s_per_mol"], row["peak_mem_MiB"]),
              flush=True)

    n = len(rows)
    tot_valid = sum(r["valid"] for r in rows)
    tot_req = sum(r["requested"] for r in rows)
    tot_unique = sum(r["unique"] for r in rows)
    tot_wall = sum(r["wall_s"] for r in rows)
    print("\n--- dataset summary (%s) ---" % args.device)
    print("cases            : %d" % n)
    print("molecules        : %d (valid %d, overall validity %.4f)"
          % (tot_req, tot_valid, tot_valid / tot_req))
    print("unique           : %d (overall unique rate %.4f)" % (tot_unique, tot_unique / max(tot_valid, 1)))
    print("total wall       : %.1f s (%.2f s/molecule)" % (tot_wall, tot_wall / tot_req))
    print("peak mem (max)   : %.1f MiB" % max(r["peak_mem_MiB"] for r in rows))

    csv_path = args.csv or os.path.join(HERE, "results_dataset_%s.csv"
                                         % torch.device(args.device).type)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("csv              : %s" % csv_path)

    smi_path = os.path.join(HERE, "smiles_%s.txt" % torch.device(args.device).type)
    with open(smi_path, "w") as f:
        f.write("\n".join(smiles_all) + "\n")
    print("smiles           : %s (%d lines)" % (smi_path, len(smiles_all)))


if __name__ == "__main__":
    main()
