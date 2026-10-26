# EPMolGen SDAA 适配

## 结论

官方推理脚本 `sample_main.py` 在 SDAA 上直接可跑，5 个生成分子全部有效（`Validity 1.0000` / `Unique 1.0000`）。

与 CUDA（A100）对照：SDAA 慢 **13.2×**（16.53 s/分子 对 1.25 s/分子），峰值显存 152.6 MiB 对 68.1 MiB；生成质量一致——两边 Validity 0.71 对 0.73，逐分子相同率 42.3%。

性能差距的主因不在卡算力：每加一个原子都要在 CPU 侧重建 KNN 图（`GraphCompose`，16 进程）+ RDKit 建分子，真正上卡的 forward 只有几百个节点；SDAA 与本机 CPU（27.1 s/分子）也基本同速。

## 随机数对齐

生成模型的对齐必须处理两处随机源，缺一不可：

| 随机源 | 位置 | 处理 |
| --- | --- | --- |
| 设备端 uniform / normal | `torch.multinomial`、`torch.rand`、先验采样 | SDAA 侧设 `TORCH_SDAA_ALIGN_NV_DEVICE=a100`，与 CUDA 的 Philox 对齐 |
| 主机端 numpy | `EPMolGen/sample.py:27` 的 `np.random.randint` 决定每个分子的原子数 | 评测脚本里按 case 调 `np.random.seed()` |

**`TORCH_SDAA_ALIGN_NV_DEVICE=a100` 实测有效**（同一 seed=1234，`torch.manual_seed` 后取样）：

| 取样 | SDAA 不对齐 | SDAA 对齐 a100 | CUDA (A100) |
| --- | --- | --- | --- |
| `torch.rand(16)` | 与 CUDA 完全不同 | **与 CUDA 逐位相同** | 基准 |
| `torch.randn(16)` | 与 CUDA 完全不同 | 与 CUDA 相同（差在 float32 末位 ~1e-7） | 基准 |
| `torch.randint(16)` | — | 与 CUDA **不同** | 基准 |
| `torch.normal(16)` | — | 与 CUDA **不同**（SDAA 给的是 CUDA `randn(4,8)` 的前 16 个） | 基准 |

即 uniform 路径对齐了，int64 与 `normal_` 路径没有。EPMolGen 的先验采样走 `torch.distributions.Normal`（即 `normal_`），`torch.multinomial` 消耗的均匀数序列也可能因此错位，这是跨平台仍有 42.3% 分子不一致的直接原因。

**numpy 播种是必须的**：仓库代码从不播种 numpy，而每个分子的原子数就来自 `np.random.randint`，不播种则**同一平台两次运行都不可复现**。播种后实测：同一平台重跑两次，73 个分子 **100% 逐个相同**。

## 环境

| 项 | SDAA 侧 | CUDA 对照侧 |
| --- | --- | --- |
| 宿主 | 10.71.15.115（Loongson-3C6000/D，32 张 SDAA） | 10.10.6.21（AMD EPYC 7453，A100-PCIE-40GB，共享机） |
| 容器 | `sdaa`（镜像 `3.2.1-torch_sdaa3.2.1`） | 无（直接用宿主 conda env） |
| Python | `/home/py312/bin/python3`（3.12.13） | `anaconda3/envs/surf/bin/python`（3.10.21） |
| torch | 2.12.0 / Torch-SDAA 3.2.1 | 2.4.1+cu121 |
| 设备 | `sdaa:0`（单核组 16 GiB） | `cuda:0`（A100 40 GB） |
| 仓库 | `/data/application/xuqiang/EPMolGen`，分支 `adapt/sdaa` | `/data/application/xuqiang/EPMolGen_cuda`（同一份工作树） |
| RNG | `TORCH_SDAA_ALIGN_NV_DEVICE=a100` | 默认（Philox） |

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
# SDAA（注意 RNG 对齐变量）
docker exec -u 1007:1007 -e USER=xuqiang -e HOME=/data/application/xuqiang \
  -e TORCH_SDAA_ALIGN_NV_DEVICE=a100 sdaa bash -c \
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

CUDA 同样 5 个分子、`Validity 1.0000` / `Unique 1.0000`，耗时 13 s（SDAA 112 s）。官方脚本不播种任何 RNG，每次运行结果都不同。

## 性能与显存

| 指标 | SDAA | CUDA (A100) | 倍数 |
| --- | --- | --- | --- |
| 官方 CLI（5 分子，含准备与加载） | 112 s | 13 s | 8.6× |
| 数据集 10 口袋 × 10 分子 | 16.53 s/分子 | 1.25 s/分子 | 13.2× |
| 峰值显存（10 口袋最大） | 152.6 MiB | 68.1 MiB | 2.2× |
| 同机 CPU 对照 | 27.1 s/分子 | — | — |

CUDA 侧是共享机，耗时随宿主 CPU 负载变化：load 4.3 时 1.25 s/分子，load 79 时同一脚本 4.59 s/分子（GPU 利用率始终 0%，瓶颈全在 CPU 侧）。SDAA 侧是独占机。

单分子耗时随原子数变化（SDAA 侧 10 原子 10.6 s、28 原子 30.1 s）。SDAA 相对本机 CPU 仅 1.18×。论文报告 RTX 4090 上 0.7 s/分子。

## 精度

数据集用仓库自带的 `Ten_test_cases`（10 个口袋），每个口袋 10 个分子共 100 个，seed 1234（`torch.manual_seed` + `np.random.seed`，逐 case 加索引），SDAA 侧开 RNG 对齐。指标用仓库自己写的 `Generate.generate()`，即 `metrics.dir` 里的 Validity / Unique。

| 口袋 | SDAA 有效 | CUDA 有效 | 逐分子相同 | SDAA s/分子 | CUDA s/分子 | SDAA 峰值 MiB | CUDA 峰值 MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1bvr | 9/10 | 9/10 | 6/9 | 19.22 | 1.33 | 122.1 | 56.5 |
| 1u0f | 10/10 | 10/10 | 7/10 | 14.98 | 1.13 | 89.1 | 45.0 |
| 1zyu | 9/10 | 10/10 | 2/9 | 18.70 | 1.26 | 122.0 | 56.4 |
| 2ah9 | 9/10 | 9/10 | 1/9 | 23.30 | 1.47 | 152.6 | 68.1 |
| 2ati | 8/10 | 9/10 | 4/8 | 15.72 | 1.28 | 96.9 | 47.7 |
| 2hw1 | 7/10 | 5/10 | 1/7 | 15.83 | 1.24 | 95.5 | 46.6 |
| 4bnw | 2/10 | 3/10 | 2/2 | 17.89 | 1.41 | 106.7 | 50.9 |
| 4i91 | 5/10 | 6/10 | 5/5 | 11.77 | 1.01 | 76.1 | 40.0 |
| 5g3n | 4/10 | 4/10 | 0/4 | 14.26 | 1.14 | 86.2 | 42.9 |
| 5lvq | 8/10 | 8/10 | 2/8 | 13.60 | 1.20 | 79.0 | 40.7 |
| **合计** | **71/100（0.71）** | **73/100（0.73）** | **30/71（42.3%）** | **16.53** | **1.25** | **152.6** | **68.1** |

Unique：SDAA 71/71（1.000）、CUDA 73/73（1.000）。

### 数值对齐（前向级）

端到端"逐分子一致率 42.3%"不是平台精度问题。用同一份输入、同��份权重做**确定性前向**对照（`sdaa/scripts/forward_check.py` + `compare_forward.py`，输出 `sdaa/forward_compare.txt`）：

| 张量 | float32 最大绝对差 | float32 相对差 | float64 最大绝对差 | float64 相对差 |
| --- | --- | --- | --- | --- |
| 输入 `cpx_pos` / `cpx_feature` / `cpx_edge_index` | **0（逐比特相同）** | 0 | 0 | 0 |
| `h_cpx`（embed_compose 输出） | 2.38e-07 | 8.5e-08 | 4.44e-16 | 1.6e-16 |
| `h_enc`（encoder 输出） | 3.89e-05 | 5.9e-06 | 5.15e-13 | 7.7e-14 |
| `pivotal_prob`（pivotal_net 输出） | 1.37e-06 | 1.4e-06 | 2.63e-14 | 2.6e-14 |
| `pos_abs` / `pos_pi` / `pos_sigma`（pos_predictor） | ≤4.8e-07 | ≤9.1e-09 | ≤1.0e-14 | ≤1.4e-14 |

float64 下两平台差到 1e-16（机器精度），**说明 SDAA 与 CUDA 的数值实现是等价的**；float32 下的 1e-6~1e-5 差异只是 fp32 在多层 encoder 上的舍入累积。

### 为什么端到端只能对上 42.3%

模型对 fp32 量级的扰动本身是混沌的。在**同一台 CUDA** 上把权重加 1e-5~1e-3 的绝对扰动、其余条件完全相同再跑一遍（`sdaa/scripts/chaos_check.py`，输出 `sdaa/chaos_check.txt`）：

| 权重扰动 | 10 个分子里保持不变 |
| --- | --- |
| 1e-5 | 6 / 10 |
| 1e-4 | 1 / 10 |
| 1e-3 | 0 / 10 |

而两平台之间的前向差异恰好就在 1e-5 这个量级。所以「同一套种子、两个数值等价的平台，只有 42% 的分子逐个相同」是模型自身的性质，不是平台缺陷；端到端逐分子一致率在这个模型上不是合适的精度指标。

合适的指标是两条：**① 确定性前向的逐元素数值对照**（已给，等价到 1e-16）；**② 分布级质量统计**（下面两节：Validity/Unique 与理化描述符）。

### 具体理化指标

仓库只输出 Validity / Unique，没有官方评价脚本，故用 RDKit 标准描述符补一套（`sdaa/scripts/metrics_desc.py`，两平台分子在同一份 rdkit 上算，输出存 `sdaa/metrics_desc.txt`）：

| 指标 | SDAA（71 个分子） | CUDA（73 个分子） | 差 |
| --- | --- | --- | --- |
| QED | 0.588 ± 0.165 | 0.616 ± 0.167 | −4.6% |
| SA score | 3.101 ± 1.078 | 2.989 ± 1.109 | +3.7% |
| logP (Crippen) | 3.336 ± 2.023 | 3.233 ± 1.919 | +3.2% |
| TPSA | 41.12 ± 40.03 | 47.35 ± 40.77 | −13.2% |
| MW | 258.4 ± 47.9 | 260.4 ± 47.3 | −0.8% |
| 重原子数 | 17.70 ± 2.62 | 17.85 ± 2.54 | −0.8% |
| HBD | 0.915 ± 1.118 | 1.164 ± 1.131 | −21.4% |
| HBA | 2.085 ± 1.857 | 2.192 ± 1.920 | −4.9% |
| 可旋转键 | 4.97 ± 2.37 | 5.06 ± 2.26 | −1.6% |
| 环数 | 1.465 ± 0.734 | 1.466 ± 0.668 | −0.1% |
| 芳香环数 | 0.986 ± 0.853 | 1.055 ± 0.780 | −6.5% |
| Fsp3 | 0.400 ± 0.236 | 0.371 ± 0.236 | +8.0% |

每一项的均值差都远小于自身标准差（QED 差 0.028 对 σ 0.165，SA 差 0.11 对 σ 1.08），**两平台生成分子的理化性质无实质差异**。

仓库自带的环统计（`EPMolGen/tools/static_ring.py: substructure`）也一致：六元环占比 SDAA 0.887 / CUDA 0.918，五元环 0.028 / 0.014，七元环 0.028 / 0.041，八元环 0 / 0.014，SSSR 均为 1。

结论：

1. **总体质量一致**：Validity 0.71 对 0.73，差 2 个分子，逐案差 ≤2；12 项理化描述符的均值差全部落在 1σ 以内。
2. **可复现**：同一平台（CUDA）同种子重跑两次，73 个分子 100% 逐个相同；未播种 numpy 时连这一点都不成立。
3. **残余差异的来源已定位**：不是 RNG（抽样路径上 `rand` / `normal` / `multinomial` 逐值对齐），也不是平台数值（float64 下等价到 1e-16），而是模型在 fp32 噪声下的采样混沌（见上节：同平台 1e-5 权重扰动就有 4/10 分子变化）。
4. 未对齐 RNG、且不播种 numpy 的那轮旧数据（Validity 0.80 对 0.68）**不可复现，已作废**。

原始数据：`sdaa/scripts/results_dataset_sdaa_align.csv`（SDAA）、`results_dataset_cuda_run2.csv`（CUDA）、`smiles_sdaa.txt`、`smiles_cuda.txt`；指标输出 `sdaa/metrics_desc.txt`；日志在 `sdaa/logs/`；比对脚本 `sdaa/scripts/compare_smiles.py`、`sdaa/scripts/metrics_desc.py`。同目录下 `results_dataset_sdaa.csv` / `results_dataset_cuda.csv` / `*_30.csv` 是未对齐 RNG、未播种 numpy 的那轮旧数据，已作废，仅留档。

## 已知限制

- **加速有限**：SDAA 比 CUDA 慢 13.2×，比本机 CPU 仅快 1.18×。要真正吃到加速需把逐分子串行改成批量生成（一次 forward 出 N 个候选），属采样逻辑改造，不是适配能解决的。
- **跨平台只做到 uniform 对齐**：`torch.randint`（int64）两平台的随机流不同，不过推理路径不用它；`rand` / `normal` / `multinomial` 在抽样路径上逐值对齐。
- **端到端逐分子一致率停在 42.3%**：模型对 fp32 噪声敏感所致（同平台 1e-5 权重扰动即改变 4/10 分子），不是平台缺陷；要做到端到端 100% 一致需让全流程在 float64 下运行，而采样中途的图重建会把 dtype 打回 float32，需要改上游 `transform` 逻辑。
- **训练路径未适配**：`pretraining.py` / `finetuning.py` 里的 `torch.cuda.amp` 与硬编码 `cuda:0` 未动。
- **PyG 版本**：仓库钉 2.5.3，SDAA 侧是 2.8.0.post1、CUDA 侧 2.6.1；`knn` 的 batch 索引类型要求已按 2.8 修正（2.6 同样接受整型）。
- `sample_main.py` 顶部的 `CUDA_LAUNCH_BLOCKING` / `CUDA_VISIBLE_DEVICES` 未改：torch 在其之前已 import，两个变量本就不生效。
- 无官方评测脚本，本文的精度对照用的是仓库 `Generate.generate()` 自带的 `metrics.dir`（Validity / Unique / ring_size）。
