"""Memory decomposition probe: weights vs activations vs allocator granularity.

Auxiliary script (not part of upstream). Mirrors bench_infer.py data prep,
then reports allocator stats at each stage of one molecule generation.
"""
import argparse
import os
import sys

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from EPMolGen import epmolgen, Generate  # noqa: E402
from EPMolGen.tools import *  # noqa: F401,F403,E402
from EPMolGen.tools import mask_node, ComplexData  # noqa: E402
from EPMolGen.tools.parse_pdb import Protein  # noqa: E402
from guassian_atom_number import sample_atom_num  # noqa: E402

ATOMIC_NUMBERS = [6, 7, 8, 9, 15, 16, 17, 35, 53]


def backend(device):
    if device.type == "sdaa":
        return torch.sdaa
    if device.type == "cuda":
        return torch.cuda
    return None


def idx(device):
    return device.index if device.index is not None else 0


def reset_peak(device):
    mod = backend(device)
    if mod is not None and getattr(mod, "reset_peak_memory_stats", None):
        mod.reset_peak_memory_stats(idx(device))


def mi(b):
    return b / 2**20


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
    ap.add_argument("--pocket", default="./Ten_test_cases/1bvr/1bvr_pocket10.pdb")
    ap.add_argument("--topo", default="./Ten_test_cases/1bvr/1bvr_topo.dict")
    ap.add_argument("--ckpt", default="./model_ckpt.pt")
    args = ap.parse_args()
    device = torch.device(args.device)
    mod = backend(device)

    # --- 0) allocator granularity probe (no model involved) ---
    torch.zeros(1, device=device)  # force backend init before memory APIs
    reset_peak(device)
    base = mod.memory_allocated(idx(device))
    big = torch.empty(262144, dtype=torch.float32, device=device)  # 1 MiB
    print("[granularity] 1 x 1MiB tensor  -> delta %.1f KiB (ideal 1024)" % ((mod.memory_allocated(idx(device)) - base) / 1024), flush=True)
    del big
    base = mod.memory_allocated(idx(device))
    small = [torch.empty(256, dtype=torch.float32, device=device) for _ in range(1000)]  # 1000 KiB ideal
    print("[granularity] 1000 x 1KiB     -> delta %.1f KiB (ideal 1000)" % ((mod.memory_allocated(idx(device)) - base) / 1024), flush=True)
    del small

    # --- 1) weights ---
    cpx, graph_compose = build_complex(args.pocket, args.topo)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = epmolgen(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    n_par = sum(p.numel() for p in model.parameters())
    print("[weights]     params %.1f M (%.1f MiB fp32)  | allocator after load: %.1f MiB"
          % (n_par / 1e6, n_par * 4 / 2**20, mi(mod.memory_allocated(idx(device)))), flush=True)

    # --- 2) one molecule ---
    reset_peak(device)
    sample = Generate(
        model, graph_compose.run, temperature=[1.0, 1.0], atom_type_map=ATOMIC_NUMBERS,
        num_bond_type=4, num_atom_max=35, pivotal_threshold=0.5, max_double_in_6ring=0,
        min_dist_inter_mol=3.0, bond_length_range=(1.0, 2.0), choose_max=1, device=args.device,
    )
    sample.num_atom_max = sample_atom_num([10, 28], distribution="normal")
    out = sample.run(cpx.clone().detach())
    peak_a = mod.max_memory_allocated(idx(device))
    print("[1 molecule]  peak allocated %.1f MiB  | live now %.1f MiB" % (mi(peak_a), mi(mod.memory_allocated(idx(device)))), flush=True)
    try:
        print("[1 molecule]  peak reserved  %.1f MiB" % mi(mod.max_memory_reserved(idx(device))), flush=True)
    except Exception:
        pass
    print("[1 molecule]  activations+temp peak (minus weights): %.1f MiB"
          % (mi(peak_a) - mi(mod.memory_allocated(idx(device)))), flush=True)


main()
