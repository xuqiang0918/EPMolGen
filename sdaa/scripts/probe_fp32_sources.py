"""Locate which op / layer introduces the first fp32 divergence.

Two modes:
  ops    - micro probes over candidate ops (GEMM/linear/scatter/softmax/exp/...),
           deterministic np-built inputs, each op run twice to check self-determinism.
  layers - deterministic forward (same chain as forward_check.py) with hooks on
           every encoder interaction block + stage outputs + full weight dump.

Outputs an .npz of float32 arrays; compare bitwise on any machine.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from EPMolGen import epmolgen  # noqa: E402
from EPMolGen.EPMolGen_model import embed_compose  # noqa: E402
from EPMolGen.tools import *  # noqa: F401,F403,E402
from bench_infer import ATOMIC_NUMBERS, build_complex  # noqa: F401,E402


def det32(a64):
    return a64.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", default="ops", choices=["ops", "layers"])
    ap.add_argument("--pocket", default="./Ten_test_cases/1bvr/1bvr_pocket10.pdb")
    ap.add_argument("--topo", default="./Ten_test_cases/1bvr/1bvr_topo.dict")
    ap.add_argument("--ckpt", default="./model_ckpt.pt")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="/tmp/probe.npz")
    args = ap.parse_args()
    device = torch.device(args.device)
    out = {}

    if args.mode == "ops":
        # deterministic inputs (bitwise identical by construction)
        rng64 = (np.arange(2000 * 256, dtype=np.float64) % 97) / 97 - 0.5
        x = torch.from_numpy(det32(rng64).reshape(2000, 256)).to(device)
        w64 = (np.arange(256 * 256, dtype=np.float64) % 61) / 61 - 0.5
        w = torch.from_numpy(det32(w64).reshape(256, 256)).to(device)
        b = torch.from_numpy(det32((np.arange(256, dtype=np.float64) % 23) / 23)).to(device)
        a64m = (np.arange(1024 * 512, dtype=np.float64) % 53) / 53 - 0.5
        A = torch.from_numpy(det32(a64m).reshape(1024, 512)).to(device)
        b64m = (np.arange(512 * 1024, dtype=np.float64) % 47) / 47 - 0.5
        B = torch.from_numpy(det32(b64m).reshape(512, 1024)).to(device)
        src64 = (np.arange(20000, dtype=np.float64) % 89) / 89 - 0.5
        src = torch.from_numpy(det32(src64)).to(device)
        index = torch.from_numpy(((np.arange(20000) * 7919) % 500).astype(np.int64)).to(device)
        s2_64 = (np.arange(1000 * 128, dtype=np.float64) % 83) / 83 - 0.5
        s2 = torch.from_numpy(det32(s2_64).reshape(1000, 128)).to(device)
        e64 = (np.arange(4096, dtype=np.float64) % 71) / 71 * 16 - 8
        e = torch.from_numpy(det32(e64)).to(device)
        v3 = torch.from_numpy(det32((np.arange(20000 * 3, dtype=np.float64) % 37) / 37 - 0.5).reshape(20000, 3)).to(device)
        w_pt = torch.from_numpy(det32(w64).reshape(256, 256)).to(device)

        def run_all(tag):
            r = {}
            r["linear_2000x256"] = F.linear(x, w, b).reshape(-1)
            r["gemm_1024x512x1024"] = (A @ B).reshape(-1)
            r["scatter_add_500"] = scatter_add(src, index, dim=0, dim_size=500)
            r["scatter_softmax_500"] = scatter_softmax(src, index, dim=0)
            r["softmax_1000x128"] = F.softmax(s2, dim=-1).reshape(-1)
            r["exp_4096"] = torch.exp(e)
            r["tanh_4096"] = torch.tanh(e)
            r["gelu_4096"] = F.gelu(e)
            r["layernorm_200x256"] = F.layer_norm(x[:200], (256,)).reshape(-1)
            r["norm_20000x3"] = torch.norm(v3, dim=-1)
            r["mean_2000x256"] = x.mean().reshape(1)
            r["sigmoid_4096"] = torch.sigmoid(e)
            return {k: v.detach().float().cpu().numpy() for k, v in r.items()}

        r1 = run_all("det1")
        r2 = run_all("det2")
        for k in r1:
            out[k] = r1[k]
            out[k + "__run2"] = r2[k]

    else:  # layers
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cpx, _ = build_complex(args.pocket, args.topo)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = epmolgen(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        # full weight dump (bitwise identity check of the loaded parameters)
        wcat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        out["weights_flat"] = wcat.float().cpu().numpy()

        rec = {}

        def flat(x):
            if isinstance(x, (list, tuple)):
                return torch.cat([t.reshape(t.shape[0], -1) for t in x], dim=1)
            return x

        def mkhook(name):
            def h(mod, inp, o):
                rec[name] = flat(o).detach()
            return h

        for i, blk in enumerate(model.encoder.interactions):
            blk.register_forward_hook(mkhook("enc%d" % i))

        data = cpx.to(device)
        with torch.no_grad():
            h_cpx = embed_compose(
                data.cpx_feature.float(), data.cpx_pos, data.idx_ligand_ctx_in_cpx,
                data.idx_protein_in_cpx, model.ligand_atom_emb, model.protein_atom_emb,
                model.emb_dim)
            rec["embed"] = flat(h_cpx).detach()
            h_enc = model.encoder(
                node_attr=h_cpx, pos=data.cpx_pos, edge_index=data.cpx_edge_index,
                edge_feature=data.cpx_edge_feature.float(), annealing=model.msg_annealing)
            rec["enc_final"] = flat(h_enc).detach()
            pivotal_pred = model.pivotal_net(h_enc, data.idx_protein_in_cpx)
            pivotal_prob = torch.sigmoid(pivotal_pred).view(-1)
            rec["pivotal_prob"] = pivotal_prob.detach()
            rel, abs_pos, sigma, pi = model.pos_predictor(
                h_enc, data.idx_protein_in_cpx[pivotal_prob.argmax()].view(-1),
                data.cpx_pos, atom_type_emb=None)
            rec["pos_abs"] = abs_pos.detach()
            rec["pos_sigma"] = sigma.detach()

        # inputs for reference
        out["cpx_pos"] = data.cpx_pos.cpu().numpy()
        out["cpx_feature"] = data.cpx_feature.cpu().numpy()
        out["cpx_edge_index"] = data.cpx_edge_index.cpu().numpy()
        for k, v in rec.items():
            out[k] = v.float().cpu().numpy()

    np.savez(args.out, **out)
    print("device=%s mode=%s out=%s tensors=%d" % (args.device, args.mode, args.out, len(out)), flush=True)


if __name__ == "__main__":
    main()
