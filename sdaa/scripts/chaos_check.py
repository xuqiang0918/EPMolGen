"""Chaos sensitivity check: how many generated molecules change when the model
weights are perturbed by ~1e-6 (the fp32 noise level between two platforms).

Runs the repository sampling twice with identical seeds, the second time with
perturbed weights. A high change rate means the end-to-end molecule comparison
is dominated by discrete-decision chaos, not by a platform defect.

Usage:
    python sdaa/scripts/chaos_check.py --device cuda:0 -n 10
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from EPMolGen import epmolgen, Generate  # noqa: E402
from EPMolGen.tools import *  # noqa: F401,F403,E402
from guassian_atom_number import sample_atom_num  # noqa: E402
from bench_infer import ATOMIC_NUMBERS, build_complex  # noqa: E402


def generate_smiles(model, cpx, graph_compose, device, n, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    sample = Generate(
        model, graph_compose.run, temperature=[1.0, 1.0], atom_type_map=ATOMIC_NUMBERS,
        num_bond_type=4, num_atom_max=35, pivotal_threshold=0.5, max_double_in_6ring=0,
        min_dist_inter_mol=3.0, bond_length_range=(1.0, 2.0), choose_max=1, device=device,
    )
    out = []
    for _ in range(n):
        sample.num_atom_max = sample_atom_num([10, 28], distribution="normal")
        res = sample.run(cpx.clone().detach())
        mol = res[0] if res else None
        if mol is not None:
            from rdkit import Chem
            out.append(Chem.MolToSmiles(mol))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--pocket", default="./Ten_test_cases/1bvr/1bvr_pocket10.pdb")
    ap.add_argument("--topo", default="./Ten_test_cases/1bvr/1bvr_topo.dict")
    ap.add_argument("--ckpt", default="./model_ckpt.pt")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = torch.device(args.device)
    cpx, graph_compose = build_complex(args.pocket, args.topo)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = epmolgen(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    base = generate_smiles(model, cpx, graph_compose, args.device, args.n, args.seed)

    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(args.seed + 7)
        for p in model.parameters():
            noise = torch.randn(p.shape, generator=g).to(p.device)
            p.add_(noise * args.eps)

    pert = generate_smiles(model, cpx, graph_compose, args.device, args.n, args.seed)

    common = set(base) & set(pert)
    print("device=%s  eps=%g  molecules: base=%d perturbed=%d" % (args.device, args.eps, len(base), len(pert)))
    print("identical molecules: %d / %d" % (len(common), max(len(base), len(pert))))
    for i, (a, b) in enumerate(zip(base, pert)):
        print("  %2d  %-52s %s" % (i, a, "SAME" if a == b else "DIFF -> " + b))


if __name__ == "__main__":
    main()
