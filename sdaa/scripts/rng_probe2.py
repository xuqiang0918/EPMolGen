"""RNG 对齐探针：只测推理路径真正用到的取样函数。

要点：每个取样函数必须在**独立的进程/独立的手动种子**下测，
否则前一个调用消耗掉的随机流会污染后一个的结果（此前探针误报
"normal 不对齐" 就是这个原因：它先调了不对齐的 int64 randint）。

用法:
    python rng_probe2.py --device sdaa:0
    python rng_probe2.py --device cuda:0
"""
import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    dev = torch.device(args.device)
    out = {}

    def probe(name, fn):
        torch.manual_seed(args.seed)
        out[name] = [float(x) for x in fn()]

    probe("rand", lambda: torch.rand(8, device=dev))
    probe("randn", lambda: torch.randn(8, device=dev))
    probe("normal", lambda: torch.normal(0.0, 1.0, (8,), device=dev))
    probe("randint", lambda: torch.randint(0, 100, (8,), device=dev))

    # multinomial：推理路径真正用的抽样
    torch.manual_seed(args.seed)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]],
                          device=dev)
    out["multinomial"] = [int(x) for x in torch.multinomial(logits, 3).flatten()]

    # distributions.Normal 的 sample：EPMolGen 先验采样走这条
    torch.manual_seed(args.seed)
    d = torch.distributions.Normal(0.0, 1.0)
    out["dist_normal"] = [float(x) for x in d.sample((8,)).to(dev)]

    print(json.dumps(out))


if __name__ == "__main__":
    main()
