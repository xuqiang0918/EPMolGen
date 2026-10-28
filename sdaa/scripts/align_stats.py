"""SDAA / CUDA 生成结果对齐统计（辅助脚本）。

对同一批 pocket、同样分子数、同样 seed 的两次生成结果，做分布级对照：
论文官方指标（Validity / Unique / QED / LogP / SA）+ 环统计 + 逐 case 明细。

设计口径：
  * 只做「分布级」比较。EPMolGen 是自回归逐原子采样，float32 下的
    1e-5 级前向差异就会被 multinomial 放大成不同分子（同一 CUDA 上仅
    扰动权重 1e-5 就会有 4/10 分子改变），所以「逐分子一致」不是
    有效的对齐判据。
  * 两边必须在同一 RDKit 版本下算。本脚本在跑的一侧直接读两边的
    generated.smi，避免跨机器的 RDKit 版本差污染指标。

用法：
    python sdaa/scripts/align_stats.py \
        --ref gen_results_dataset_cuda_50 \
        --test gen_results_dataset_sdaa_50 \
        --cases 1bvr,1u0f,1zyu,2ah9
"""
import argparse
import os
import statistics
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from metrics_paper import _load_sascorer, ring_stats  # noqa: E402

from rdkit.Chem import Crippen, Descriptors, QED  # noqa: E402


def latest_dir(root, case):
    d = os.path.join(root, case)
    subs = sorted(os.path.join(d, x) for x in os.listdir(d))
    return subs[-1] if subs else d


def load_smiles(root, case):
    p = os.path.join(latest_dir(root, case), "generated.smi")
    if not os.path.exists(p):
        return None
    return [l.strip() for l in open(p) if l.strip()]


def load_repo_metrics(root, case):
    """Validity / Unique 取仓库自己写的 metrics.dir，不自己重算。"""
    p = os.path.join(latest_dir(root, case), "metrics.dir")
    if not os.path.exists(p):
        return None, None
    raw = open(p).read()
    import re
    try:
        v = float(re.search(r"'Validity':\s*([0-9.]+)", raw).group(1))
        u = float(re.search(r"'Unique':\s*([0-9.]+)", raw).group(1))
    except AttributeError:
        return None, None
    return v, u


def metrics_of(smiles, sascorer):
    mols = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
    qed = [QED.qed(m) for m in mols]
    logp = [Crippen.MolLogP(m) for m in mols]
    heavy = [Descriptors.HeavyAtomCount(m) for m in mols]
    sa = [sascorer.calculateScore(m) for m in mols] if sascorer else None
    return {
        "n": len(mols),
        "qed": qed,
        "logp": logp,
        "heavy": heavy,
        "sa": sa,
        "ring": ring_stats(mols),
    }


def fmt(v):
    if v is None:
        return "     -   "
    return "%5.2f" % v


def mean_sd(v):
    if not v:
        return None, None
    return statistics.mean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="参照平台结果根目录（CUDA）")
    ap.add_argument("--test", required=True, help="被测平台结果根目录（SDAA）")
    ap.add_argument("--cases", default="", help="逗号分隔；空 = 取 test 目录下全部")
    ap.add_argument("--ref-label", default="cuda")
    ap.add_argument("--test-label", default="sdaa")
    args = ap.parse_args()

    sascorer = _load_sascorer()
    if sascorer is None:
        print("SA score unavailable (sascorer import failed)")

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = sorted(os.listdir(args.test))

    rows = []
    for case in cases:
        s_ref = load_smiles(args.ref, case)
        s_test = load_smiles(args.test, case)
        if not s_test:
            print("[skip] %s: no test set" % case)
            continue
        if s_ref is None:
            print("[skip] %s: no ref set" % case)
            continue
        v_ref, u_ref = load_repo_metrics(args.ref, case)
        v_test, u_test = load_repo_metrics(args.test, case)
        m_ref = metrics_of(s_ref, sascorer)
        m_test = metrics_of(s_test, sascorer)
        inter = len(set(s_ref) & set(s_test))
        rows.append((case, v_ref, u_ref, m_ref, v_test, u_test, m_test, inter,
                     len(s_ref), len(s_test)))

    print("\n=== per-case: %s (ref) vs %s (test) ===" % (args.ref_label, args.test_label))
    hdr = ("case", "n_ref", "n_test", "Val%", "Uniq%", "QED_ref", "QED_test",
           "logP_ref", "logP_test", "SA_ref", "SA_test", "HA_ref", "HA_test",
           "SMILES∩")
    print("%-6s %5s %6s %6s %6s %7s %8s %8s %9s %7s %7s %7s %8s %7s" % hdr)
    for (case, v_r, u_r, mr, v_t, u_t, mt, inter, nr, nt) in rows:
        qr, qs = mean_sd(mr["qed"])
        qt, qts = mean_sd(mt["qed"])
        lr, lrs = mean_sd(mr["logp"])
        lt, lts = mean_sd(mt["logp"])
        sr, srs = mean_sd(mr["sa"])
        st, sts = mean_sd(mt["sa"])
        hr, hrs = mean_sd(mr["heavy"])
        ht, hts = mean_sd(mt["heavy"])
        print("%-6s %5d %6d %6s %6s %7s %8s %8s %9s %7s %7s %7s %8s %7d" % (
            case, nr, nt,
            "%.0f" % (v_r * 100) if v_r is not None else "-",
            "%.0f" % (u_r * 100) if u_r is not None else "-",
            "%.2f+-%.2f" % (qr, qs) if qr is not None else "-",
            "%.2f+-%.2f" % (qt, qts) if qt is not None else "-",
            "%.2f+-%.2f" % (lr, lrs) if lr is not None else "-",
            "%.2f+-%.2f" % (lt, lts) if lt is not None else "-",
            "%.2f+-%.2f" % (sr, srs) if sr is not None else "-",
            "%.2f+-%.2f" % (st, sts) if st is not None else "-",
            "%.1f+-%.1f" % (hr, hrs) if hr is not None else "-",
            "%.1f+-%.1f" % (ht, hts) if ht is not None else "-",
            inter))

    # 汇总（把两边的分子合在一起算分布）
    all_ref, all_test = [], []
    for (case, v_r, u_r, mr, v_t, u_t, mt, inter, nr, nt) in rows:
        all_ref.append(mr)
        all_test.append(mt)

    def pool(ms, key):
        out = []
        for m in ms:
            out.extend(m[key] or [])
        return out

    print("\n=== pooled over %d cases ===" % len(rows))
    for key, name in (("qed", "QED"), ("logp", "LogP"), ("sa", "SA"),
                      ("heavy", "heavy atoms")):
        vr = pool(all_ref, key)
        vt = pool(all_test, key)
        mr, sr = mean_sd(vr)
        mt, st = mean_sd(vt)
        if mr is None or mt is None:
            continue
        # 标准化均值差：|mean_test - mean_ref| / pooled_sd
        pooled = ((sr ** 2 + st ** 2) / 2) ** 0.5
        smd = (mt - mr) / pooled if pooled else 0.0
        print("%-12s ref %6.2f +- %-5.2f   test %6.2f +- %-5.2f   diff %+6.2f   SMD %+.3f"
              % (name, mr, sr, mt, st, mt - mr, smd))

    print("\n--- ring statistics (%% of molecules, pooled) ---")
    for label, ms in ((args.ref_label, all_ref), (args.test_label, all_test)):
        tot = sum(m["ring"]["n"] for m in ms)
        if not tot:
            continue
        three = sum(m["ring"]["three_pct"] * m["ring"]["n"] for m in ms) / tot
        eight = sum(m["ring"]["eight_pct"] * m["ring"]["n"] for m in ms) / tot
        cond = sum(m["ring"]["condensed_pct"] * m["ring"]["n"] for m in ms) / tot
        print("%-6s three %.2f  eight %.2f  condensed %.2f" % (label, three, eight, cond))

    tot_r = sum(m["n"] for m in all_ref)
    tot_t = sum(m["n"] for m in all_test)
    inter = sum(r[7] for r in rows)
    print("\nmolecules        : %s %d / %s %d" % (args.ref_label, tot_r,
                                                 args.test_label, tot_t))
    print("SMILES overlap   : %d (%.1f%% of %s set)"
          % (inter, 100.0 * inter / max(tot_r, 1), args.ref_label))


if __name__ == "__main__":
    main()
