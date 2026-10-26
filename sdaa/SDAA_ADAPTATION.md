# EPMolGen SDAA 适配

## 结论

官方推理脚本 `sample_main.py` 在 SDAA 上直接可跑，5 个生成分子全部有效（`Validity 1.0000` / `Unique 1.0000`）。

与 CUDA（A100）对照：SDAA 慢 **12.9×**（16.13 s/分子 对 1.25 s/分子），峰值显存 153.6 MiB 对 68.1 MiB，生成质量（仓库自带的 Validity / Unique）处于同一量级。

性能差距的主因不在卡算力：每加一个原子都要在 CPU 侧重建 KNN 图（`GraphCompose`，16 进程）+ RDKit 建分子，真正上卡的 forward 只有几百个节点；SDAA 与本机 CPU（27.1 s/分子）也基本同速。

## 环境

| 项 | SDAA 侧 | CUDA 对照侧 |
| --- | --- | --- |
| 宿主 | 10.71.15.115（Loongson-3C6000/D，32 张 SDAA） | 10.10.6.21（AMD EPYC 7453，A100-PCIE-40GB） |
| 容器 | `sdaa`（镜像 `3.2.1-torch_sdaa3.2.1`） | 无（直接用宿主 conda env） |
| Python | `/home/py312/bin/python3`（3.12.13） | `anaconda3/envs/surf/bin/python`（3.10.21） |
| torch | 2.12.0 / Torch-SDAA 3.2.1 | 2.4.1+cu121 |
| 设备 | `sdaa:0`（单核组 16 GiB） | `cuda:0`（A100 40 GB） |
| 仓库 | `/data/application/xuqiang/EPMolGen`，分支 `adapt/sdaa` | `/data/application/xuqiang/EPMolGen_cuda`（同一份工作树） |

SDAA 侧依赖全部用容器自带，**未额外安装任何包**：`torch_scatter 2.1.2`、`torch_cluster 1.6.3`、`torch_sparse 0.6.18`、`pyg_lib 0.7.0`、`pymol`、`rdkit`，PyG 2.8.0.post1（仓库 `requirements.txt` 钉 2.5.3）。

CUDA 侧环境已有 torch 2.4.1+cu121 / PyG 2.6.1 / torch_scatter 2.1.2+pt24cu121 / torch_cluster / rdkit / pymol，只隔离补了 `easydict` 与 `tensorboard`（`--target` 到 `/data/application/xuqiang/pylibs_epmolgen`，不改动 env 本体，110 MB）。

## 改动

| 归属 | 位置 | 改动 |
| --- | --- | --- |
| `[sdaa-adapt]` | `EPMolGen/sample.py:276-281` | 4 处 `.cuda(data.cpx_pos.device)` → `.to(data.cpx_pos.device)`；`.cuda()` 是 CUDA 专有 API |
| `[sdaa-adapt]` | `EPMolGen/sample.py:342,344` | 3 处 `.to('cuda:0')` → `.to(self.device)`（含 2 行注释） |
| `[upstream]` | `EPMolGen/sample.py:433` | `if out: mol,_ = out` → `mol = out[0] if out else None`；`run()` 返回 None 时 `mol` 未绑定，多分子必崩 |
| `[upstream]` | `EPMolGen/sample.py:342` | `step_batch` 补 `dtype=torch.long`；PyG ≥ 2.8 的 `index2ptr` 要求整型 batch |
| `[upstream]` | `sample_main.py:80` | `torch.load(..., weights_only=False)`；torch ≥ 2.6 默认 True，而 ckpt 内含 `easydict.EasyDict` |
| `[upstream]` | `sample_main.py:43-58` | 作者本机绝对路径的调试块改为「CLI 未给才生效」的仓库内相对路径 |

patch 归档在 `sdaa/patches/`，3 个 commit 按平台适配 / 通用修复分开。

## 运行

```bash
# SDAA
docker exec -u 1007:1007 -e USER=xuqiang -e HOME=/data/application/xuqiang sdaa bash -c \
  'source /opt/tecoai/setvars.sh; cd /data/application/xuqiang/EPMolGen && \
   /home/py312/bin/python3 -u sample_main.py --device sdaa:0 --molecule_number 5'

# CUDA（对照）
cd /data/application/xuqiang/EPMolGen_cuda && \
  PYTHONPATH=/data/application/xuqiang/pylibs_epmolgen CUDA_VISIBLE_DEVICES=0 \
  /data/application/xuqiang/anaconda3/envs/surf/bin/python -u sample_main.py \
  --device cuda:0 --molecule_number 5 --root_path gen_results_sdaa_cuda
```

SDAA 输出（节选，日志 `sdaa/logs/2026-10-26_sample_main_sdaa.log`）：

```
CCC=CC=CC(C)CCc1ccccc1OC
C=CC=CCCCCC(COP(=O)(O)O)C(=O)O
...
5
5
Validity: 1.0000
Unique: 1.0000
```

CUDA 同样 5 个分子、`Validity 1.0000` / `Unique 1.0000`，耗时 13 s（SDAA 112 s）。采样为逐原子随机过程，未设随机种子，每次生成的分子不同。

## 性能与显存

| 指标 | SDAA | CUDA (A100) | 倍数 |
| --- | --- | --- | --- |
| 官方 CLI（5 分子，含准备与加载） | 112 s | 13 s | 8.6× |
| 生成阶段（`sdaa/scripts/bench_infer.py`，5 分子） | 22.9 s/分子 | 1.3 s/分子 | 17.6× |
| 数据集 10 口袋 × 10 分子 | 16.13 s/分子 | 1.25 s/分子 | 12.9× |
| 峰值显存（10 口袋最大） | 153.6 MiB | 68.1 MiB | 2.3× |
| 同机 CPU 对照 | 27.1 s/分子 | — | — |

单分子耗时随原子数变化（SDAA 侧 10 原子 10.6 s、28 原子 30.1 s）。SDAA 相对本机 CPU 仅 1.18×，说明瓶颈在 CPU 侧。论文报告 RTX 4090 上 0.7 s/分子。

## 精度

生成是随机过程，没有预测值可对；两平台即使同种子也生成不同分子（RNG 实现不同），无法逐分子比对，只能做分布级统计。指标用仓库自己写的 `Generate.generate()`，即 `metrics.dir` 里的 Validity / Unique。

数据集用仓库自带的 `Ten_test_cases`（10 个口袋），每个口袋生成 10 个分子，共 100 个：

| 口袋 | SDAA 有效 | CUDA 有效 | SDAA s/分子 | CUDA s/分子 | SDAA 峰值 MiB | CUDA 峰值 MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1bvr | 8/10 | 9/10 | 19.77 | 1.24 | 123.3 | 56.6 |
| 1u0f | 10/10 | 10/10 | 14.46 | 1.18 | 90.1 | 45.0 |
| 1zyu | 9/10 | 10/10 | 18.58 | 1.25 | 123.1 | 56.4 |
| 2ah9 | 9/10 | 7/10 | 23.34 | 1.59 | 153.6 | 68.1 |
| 2ati | 8/10 | 4/10 | 15.39 | 1.25 | 97.9 | 47.7 |
| 2hw1 | 8/10 | 7/10 | 15.57 | 1.20 | 96.5 | 46.6 |
| 4bnw | 7/10 | 2/10 | 17.17 | 1.21 | 107.7 | 50.9 |
| 4i91 | 8/10 | 3/10 | 8.31 | 1.10 | 77.1 | 40.1 |
| 5g3n | 4/10 | 8/10 | 14.49 | 1.29 | 87.3 | 43.0 |
| 5lvq | 9/10 | 8/10 | 14.22 | 1.15 | 80.0 | 40.7 |
| **合计** | **80/100（0.80）** | **68/100（0.68）** | **16.13** | **1.25** | **153.6** | **68.1** |

Unique：SDAA 78/80（0.975）、CUDA 68/68（1.000）。

两个差异最大的口袋把样本加大到每案 30 个分子：

| 口袋 | SDAA | CUDA |
| --- | --- | --- |
| 2ati | 30/30（1.000） | 23/30（0.767） |
| 4bnw | 12/30（0.400） | 13/30（0.433） |
| 合计 | 42/60（0.700） | 36/60（0.600） |

**判定**：n=10 时单个口袋的 Validity 自身波动就很大（同一口袋 CUDA 两次跑出 0.4 与 0.767）；n=30 时两平台差异方向不一致（2ati SDAA 高、4bnw SDAA 低）；n=100 的整体差 0.80 对 0.68 落在 1.4σ 内。因此**没有证据表明 SDAA 与 CUDA 存在系统性质量差异**，本次对照不足以判定更小的偏差。

原始数据：`sdaa/scripts/results_dataset_sdaa.csv`、`results_dataset_sdaa30.csv`、`results_dataset_cuda.csv`、`results_dataset_cuda30.csv`；日志在 `sdaa/logs/`。

## 已知限制

- **加速有限**：SDAA 比 CUDA 慢 12.9×，比本机 CPU 仅快 1.18×。要真正吃到加速需把逐分子串行改成批量生成（一次 forward 出 N 个候选），属采样逻辑改造，不是适配能解决的。
- **训练路径未适配**：`pretraining.py` / `finetuning.py` 里的 `torch.cuda.amp` 与硬编码 `cuda:0` 未动。
- **PyG 版本**：仓库钉 2.5.3，SDAA 侧是 2.8.0.post1、CUDA 侧 2.6.1；`knn` 的 batch 索引类型要求已按 2.8 修正（2.6 同样接受整型）。
- `sample_main.py` 顶部的 `CUDA_LAUNCH_BLOCKING` / `CUDA_VISIBLE_DEVICES` 未改：torch 在其之前已 import，两个变量本就不生效。
- 无官方评测脚本，本文的精度对照用的是仓库 `Generate.generate()` 自带的 `metrics.dir`（Validity / Unique / ring_size）。
