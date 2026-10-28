"""对齐判定 v2：逐 case 两样本 Welch t 检验 + 噪声地板对照。

为什么不用「均值差」判：
  描述符均值的标准误是 sigma/sqrt(n)，n=50 时已经很窄，但**均值本身随
  种子波动**（换一次种子，同一 case 的 LogP 均值能差 0.5）。所以单次
  换种子得到的「噪声地板」是一个带自身误差的估计，用它当阈值会误判。

本脚本的判据：
  对每个 (case, metric)，分别做
      A: SDAA(n=50)   vs  CUDA同种子(n=50)
      B: CUDA同种子(n=50) vs  CUDA换种子(n=50)  × 每个噪声种子
  用 Welch t 检验（不假设等方差），比较 |t|：

  * 若 A 的 |t| 分布与 B 的 |t| 分布不可区分，则平台差异与模型自身
    随机波动同量级 => 对齐。
  * 直接报告 p<0.05 的比例（多重比较下用 BH 校正）。

只依赖标准库 + rdkit，不引入 scipy。
"""
import argparse
import math
import os
import statistics
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from metrics_paper import _load_sascorer  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, QED  # noqa: E402


def latest_dir(root, case):
    d = os.path.join(root, case)
    if not os.path.isdir(d):
        return None
    subs = sorted(os.path.join(d, x) for x in os.listdir(d))
    return subs[-1] if subs else None


def load_smiles(root, case):
    d = latest_dir(root, case)
    if d is None:
        return None
    p = os.path.join(d, "generated.smi")
    if not os.path.exists(p):
        return None
    return [l.strip() for l in open(p) if l.strip()]


def desc(smiles, sascorer):
    mols = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
    out = {
        "QED": [QED.qed(m) for m in mols],
        "LogP": [Crippen.MolLogP(m) for m in mols],
        "HA": [float(Descriptors.HeavyAtomCount(m)) for m in mols],
    }
    if sascorer:
        out["SA"] = [sascorer.calculateScore(m) for m in mols]
    return out


def welch_t(a, b):
    """Welch t 检验，返回 (t, df, p_two_sided)。正态近似 + df 校正。"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return None
    t = (ma - mb) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # 两样本 t 检验 p 值：用不完全 beta 函数的连分数展开（学生 t 分布）
    p = 2.0 * (1.0 - student_cdf(abs(t), df))
    return t, df, p


def _betacf(a, b, x, itmax=300, eps=3e-16, fpmin=1e-300):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_cdf(t, df):
    x = df / (df + t * t)
    ib = _betai(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t > 0 else 0.5 * ib


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR 校正，返回 (是否显著列表, q值列表)。"""
    n = len(pvals)
    if n == 0:
        return [], []
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvals[i] * n / rank)
        q[i] = val
        prev = val
    return [qv <= alpha for qv in q], q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", required=True)
    ap.add_argument("--sdaa", required=True)
    ap.add_argument("--noise", nargs="+", required=True,
                    help="一个或多个 CUDA 换种子结果目录")
    ap.add_argument("--cases", default="")
    args = ap.parse_args()

    sascorer = _load_sascorer()
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = sorted(c for c in os.listdir(args.sdaa)
                       if os.path.isdir(os.path.join(args.sdaa, c)))

    metrics = ["QED", "LogP", "SA", "HA"]
    plat_p, plat_t = [], []
    noise_p, noise_t = [], []
    rows = []

    for c in cases:
        sc = load_smiles(args.cuda, c)
        ss = load_smiles(args.sdaa, c)
        if not (sc and ss):
            print("[skip] %s (missing)" % c)
            continue
        dc, ds = desc(sc, sascorer), desc(ss, sascorer)
        for m in metrics:
            if m not in dc or not dc[m] or not ds.get(m):
                continue
            r = welch_t(ds[m], dc[m])
            if r is None:
                continue
            t, df, p = r
            plat_p.append(p)
            plat_t.append(abs(t))
            rows.append(("PLATFORM", c, m, t, df, p, statistics.mean(dc[m]),
                         statistics.mean(ds[m])))

            for nroot in args.noise:
                sn = load_smiles(nroot, c)
                if not sn:
                    continue
                dn = desc(sn, sascorer)
                if m not in dn or not dn[m]:
                    continue
                r2 = welch_t(dn[m], dc[m])
                if r2 is None:
                    continue
                t2, df2, p2 = r2
                noise_p.append(p2)
                noise_t.append(abs(t2))
                rows.append(("NOISE", "%s/%s" % (os.path.basename(nroot), c), m,
                             t2, df2, p2, statistics.mean(dc[m]),
                             statistics.mean(dn[m])))

    print("=== Welch t 检验：SDAA vs CUDA(同种子)  vs  CUDA(换种子) vs CUDA(同种子) ===")
    print("%-9s %-22s %-6s %9s %7s %9s %9s %9s" %
          ("type", "case", "metric", "t", "df", "p", "mean_ref", "mean_other"))
    for (kind, case, m, t, df, p, mr, mo) in rows:
        print("%-9s %-22s %-6s %9.3f %7.1f %9.2e %9.3f %9.3f"
              % (kind, case, m, t, df, p, mr, mo))

    print("\n=== 汇总 ===")
    print("platform 组: n=%d  mean|t|=%.3f  median p=%.3f  p<0.05 的比例=%.1f%%"
          % (len(plat_p), statistics.mean(plat_t), statistics.median(plat_p),
             100.0 * sum(1 for p in plat_p if p < 0.05) / max(len(plat_p), 1)))
    print("noise   组: n=%d  mean|t|=%.3f  median p=%.3f  p<0.05 的比例=%.1f%%"
          % (len(noise_p), statistics.mean(noise_t), statistics.median(noise_p),
             100.0 * sum(1 for p in noise_p if p < 0.05) / max(len(noise_p), 1)))

    sig_p, q_p = bh_fdr(plat_p)
    sig_n, q_n = bh_fdr(noise_p)
    print("\nBH-FDR 校正后：")
    print("platform 组 q<0.05 的个数: %d / %d" % (sum(sig_p), len(sig_p)))
    print("noise   组 q<0.05 的个数: %d / %d" % (sum(sig_n), len(sig_n)))

    if plat_t and noise_t:
        mp, mn = statistics.mean(plat_t), statistics.mean(noise_t)
        # Mann-Whitney U 的近似 z 检验（比较 |t| 分布是否同位置）
        diffs = [p - n for p, n in zip(plat_t, noise_t)]
        m = statistics.mean(diffs)
        sd = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
        se = sd / math.sqrt(len(diffs)) if sd else 0.0
        t = m / se if se else 0.0
        dfree = len(diffs) - 1
        p_val = 2.0 * (1.0 - student_cdf(abs(t), dfree)) if dfree > 0 else 1.0
        print("\nmean|t| platform=%.3f  noise=%.3f  ratio=%.2f" % (mp, mn, mp / mn))
        print("paired t on |t| diff: mean %+.3f  t=%.3f  p=%.4f  (df=%d)"
              % (m, t, p_val, dfree))
        if p_val < 0.05:
            print("VERDICT: NOT ALIGNED (平台差异显著大于模型自身随机波动)")
        else:
            print("VERDICT: ALIGNED (平台差异与模型自身随机波动无显著差异)")


if __name__ == "__main__":
    main()
