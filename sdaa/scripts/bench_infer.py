"""Single-sample inference timing and peak device memory for EPMolGen.

Auxiliary script (not part of upstream). Mirrors the data preparation in
sample_main.py, then times molecule generation one molecule at a time.

Usage:
    python sdaa/bench_infer.py --device sdaa:0 -n 10
    python sdaa/bench_infer.py --device cpu  -n 3
"""

import argparse
import os
import sys
import time

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EPMolGen import epmolgen, Generate  # noqa: E402
from EPMolGen.tools import *  # noqa: F401,F403,E402
from EPMolGen.tools import mask_node, ComplexData  # noqa: E402
from EPMolGen.tools.parse_pdb import Protein  # noqa: E402
from guassian_atom_number import sample_atom_num  # noqa: E402

ATOMIC_NUMBERS = [6, 7, 8, 9, 15, 16, 17, 35, 53]


def peak_mem(device):
    """Peak allocated bytes on the device, 0 if the backend has no counter."""
    if device.type == "cuda" or device.type == "sdaa":
        mod = torch.sdaa if device.type == "sdaa" else torch.cuda
        fn = getattr(mod, "max_memory_allocated", None)
        if fn is not None:
            return fn(device)
    return 0


def reset_peak(device):
    if device.type == "cuda" or device.type == "sdaa":
        mod = torch.sdaa if device.type == "sdaa" else torch.cuda
        fn = getattr(mod, "reset_peak_memory_stats", None)
        if fn is not None:
            fn(device)


def build_complex(pocket, topo):
    pro_dict = Protein(pocket, topo_info=topo).get_atom_dict(removeHs=False)
    lig_dict = Ligand.empty_dict()
    cpx = ComplexData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pro_dict),
        ligand_dict=torchify_dict(lig_dict),
    )
    graph_compose = GraphCompose(knn=16, num_workers=16, for_gen=True, use_protein_bond=True)
    cpx = RefineData()(cpx)
    cpx = LigandCountNeighbors()(cpx)
    cpx = FeaturizeProteinAtom()(cpx)
    cpx = FeaturizeLigandAtom(atomic_numbers=ATOMIC_NUMBERS)(cpx)
    node4mask = torch.arange(cpx.ligand_pos.size(0))
    cpx = mask_node(cpx, torch.empty([0], dtype=torch.long), node4mask, num_atom_type=9, y_pos_std=0.)
    cpx = graph_compose.run(cpx)
    return cpx, graph_compose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("-n", "--molecule_number", type=int, default=10)
    ap.add_argument("--pocket", default="./Ten_test_cases/1bvr/1bvr_pocket10.pdb")
    ap.add_argument("--topo", default="./Ten_test_cases/1bvr/1bvr_topo.dict")
    ap.add_argument("--ckpt", default="./model_ckpt.pt")
    ap.add_argument("--root_path", default="gen_results_bench")
    ap.add_argument("--name", default="1bvr")
    args = ap.parse_args()

    device = torch.device(args.device)

    t0 = time.time()
    cpx, graph_compose = build_complex(args.pocket, args.topo)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = epmolgen(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    torch_dev_sync = getattr(getattr(torch, device.type, None), "synchronize", None)
    if callable(torch_dev_sync):
        torch_dev_sync()
    t_prep = time.time() - t0

    sample = Generate(
        model, graph_compose.run, temperature=[1.0, 1.0], atom_type_map=ATOMIC_NUMBERS,
        num_bond_type=4, num_atom_max=35, pivotal_threshold=0.5, max_double_in_6ring=0,
        min_dist_inter_mol=3.0, bond_length_range=(1.0, 2.0), choose_max=1, device=args.device,
    )

    reset_peak(device)
    times, valid, n_atoms = [], 0, []
    out_dir = os.path.join(args.root_path, args.name, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    sdf = os.path.join(out_dir, "generated.sdf")

    for i in range(args.molecule_number):
        sample.num_atom_max = sample_atom_num([10, 28], distribution="normal")
        t1 = time.time()
        out = sample.run(cpx.clone().detach())
        if callable(torch_dev_sync):
            torch_dev_sync()
        times.append(time.time() - t1)

        mol = out[0] if out else None
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            ok = Chem.MolFromSmiles(smi) is not None
            valid += int(ok)
            n_atoms.append(mol.GetNumAtoms())
            mol.SetProp("_Name", "No_%d-%s" % (i, args.name))
            with open(sdf, "a") as f:
                f.write(Chem.MolToMolBlock(mol) + "\n$$$$\n")
            print("%2d  %6.1f s  %2d atoms  %s  %s" % (i, times[-1], mol.GetNumAtoms(), "OK  " if ok else "FAIL", smi))
        else:
            print("%2d  %6.1f s  run() returned None" % (i, times[-1]))

    n = len(times)
    print("\n--- summary ---")
    print("device           : %s (%s)" % (args.device, torch.device(args.device).type))
    print("prep+load        : %.1f s" % t_prep)
    print("molecules        : %d" % n)
    print("total (generate) : %.1f s" % sum(times))
    print("per molecule     : %.1f s (min %.1f / max %.1f)" % (sum(times) / n, min(times), max(times)))
    print("atoms per mol    : %.1f avg" % (sum(n_atoms) / len(n_atoms) if n_atoms else 0))
    print("valid SMILES     : %d/%d" % (valid, n))
    print("peak device mem  : %.1f MiB" % (peak_mem(device) / 1024 ** 2))
    print("sdf              : %s" % sdf)


if __name__ == "__main__":
    main()
