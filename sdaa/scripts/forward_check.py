"""Deterministic forward-pass dump for cross-platform numerical comparison.

Builds the complex for one pocket, runs the deterministic part of the model
(embed_compose -> encoder -> pivotal_net -> pos_predictor) and stores the
intermediate tensors so that two platforms can be compared element-wise.

No sampling is involved, so the output is a pure function of the input.

Usage:
    python sdaa/scripts/forward_check.py --device sdaa:0 --out /tmp/fwd_sdaa.npz
    python sdaa/scripts/forward_check.py --device cuda:0 --out /tmp/fwd_cuda.npz
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
from EPMolGen.EPMolGen_model import embed_compose  # noqa: E402
from EPMolGen.tools import *  # noqa: F401,F403,E402
from bench_infer import ATOMIC_NUMBERS, build_complex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pocket", default="./Ten_test_cases/1bvr/1bvr_pocket10.pdb")
    ap.add_argument("--topo", default="./Ten_test_cases/1bvr/1bvr_topo.dict")
    ap.add_argument("--ckpt", default="./model_ckpt.pt")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--out", default="/tmp/fwd.npz")
    args = ap.parse_args()

    dt = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cpx, _ = build_complex(args.pocket, args.topo)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = epmolgen(ckpt["config"]).to(device=device, dtype=dt)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = cpx.to(device)
    with torch.no_grad():
        h_cpx = embed_compose(
            data.cpx_feature.to(dt), data.cpx_pos.to(dt), data.idx_ligand_ctx_in_cpx,
            data.idx_protein_in_cpx, model.ligand_atom_emb, model.protein_atom_emb,
            model.emb_dim)
        h_enc = model.encoder(
            node_attr=h_cpx, pos=data.cpx_pos.to(dt), edge_index=data.cpx_edge_index,
            edge_feature=data.cpx_edge_feature.to(dt), annealing=model.msg_annealing)
        pivotal_pred = model.pivotal_net(h_enc, data.idx_protein_in_cpx)
        pivotal_prob = torch.sigmoid(pivotal_pred).view(-1)
        pivot = data.idx_protein_in_cpx[pivotal_prob.argmax()].view(-1)
        rel, abs_pos, sigma, pi = model.pos_predictor(
            h_enc, pivot, data.cpx_pos.to(dt), atom_type_emb=None)

    def flat(x):
        if isinstance(x, (list, tuple)):
            return torch.cat([t.reshape(t.shape[0], -1) for t in x], dim=1)
        return x

    out = {
        "cpx_pos": data.cpx_pos.detach().cpu().numpy(),
        "cpx_feature": data.cpx_feature.detach().cpu().numpy(),
        "cpx_edge_index": data.cpx_edge_index.detach().cpu().numpy(),
        "h_cpx": flat(h_cpx).detach().cpu().numpy(),
        "h_enc": flat(h_enc).detach().cpu().numpy(),
        "pivotal_prob": pivotal_prob.detach().cpu().numpy(),
        "pos_abs": abs_pos.detach().cpu().numpy(),
        "pos_pi": pi.detach().cpu().numpy(),
        "pos_sigma": sigma.detach().cpu().numpy(),
    }
    np.savez(args.out, **out)
    print("device=%s  out=%s" % (args.device, args.out))
    for k, v in out.items():
        print("  %-14s %-18s sum=%.6f" % (k, str(v.shape), float(v.sum())))


if __name__ == "__main__":
    main()
