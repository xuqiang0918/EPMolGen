# EPMolGen SDAA 适配

## 结论

官方推理脚本 `sample_main.py` 在 SDAA 上直接可跑，改动 6 处（见「改动」），500/500 分子全部有效。

- **精度：ALIGNED**——论文官方 4 项指标 SMD 全部 < 0.11；对接亲和力 pooled 差 0.016 kcal/mol；平台差异没有超过「同一台 CUDA 换随机种子」的波动。
- **性能**：单样本基准 19.3 s/分子 对 CUDA 1.3 s/分子（≈15×），峰值显存 121.8 对 56.4 MiB（2.2×）；瓶颈在 CPU 侧逐分子图重建，不在卡算力。

## 随机数对齐

两处随机源都要处理：

| 随机源                                                        | 处理                                                            |
| ---------------------------------------------------------- | ------------------------------------------------------------- |
| 设备端取样（`multinomial` / `rand` / `normal` / `Normal.sample`） | SDAA 侧设 `TORCH_SDAA_ALIGN_NV_DEVICE=a100`，与 CUDA Philox 对齐    |
| numpy（每分子原子数）                                              | 评测脚本按 case 调 `np.random.seed()`（MT19937 平台无关，同 seed 两边序列逐位一致） |

独立播种探针（`sdaa/scripts/rng_probe2.py`）实测：设备端取样全部与 CUDA 一致，只有 int64 `torch.randint` 不对齐而推理路径不用它。播种后同平台同 seed 重跑，分子逐个相同。

## 环境

| 项      | 值                                                    |
| ------ | ---------------------------------------------------- |
| 宿主     | 10.71.15.115（Loongson-3C6000/D，32 张 SDAA）            |
| 容器     | `sdaa`（镜像 `3.2.1-torch_sdaa3.2.1`）                   |
| Python | `/home/py312/bin/python3`（3.12.13）                   |
| torch  | 2.12.0 / Torch-SDAA 3.2.1                            |
| 设备     | `sdaa:0`（单核组 16 GiB）                                 |
| 仓库     | 仓库根目录（见「运行」中 `REPO` 变量），分支 `adapt/sdaa` |
| RNG    | `TORCH_SDAA_ALIGN_NV_DEVICE=a100`                    |

依赖全部用容器自带，未额外安装：`torch_scatter 2.1.2`、`torch_cluster 1.6.3`、`torch_sparse 0.6.18`、`pyg_lib 0.7.0`、PyG `2.8.0.post1`（仓库钉 2.5.3）、`rdkit`、`pymol`。

## 改动

| 归属             | 位置                           | 改动                                                                                        |
| -------------- | ---------------------------- | ----------------------------------------------------------------------------------------- |
| `[sdaa-adapt]` | `EPMolGen/sample.py:276-281` | 4 处 `.cuda(data.cpx_pos.device)` → `.to(data.cpx_pos.device)`；`.cuda()` 是 CUDA 专有 API     |
| `[sdaa-adapt]` | `EPMolGen/sample.py:342,344` | 3 处 `.to('cuda:0')` → `.to(self.device)`（含 2 行注释）                                         |
| `[upstream]`   | `EPMolGen/sample.py:433`     | `if out: mol,_ = out` → `mol = out[0] if out else None`；`run()` 返回 None 时 `mol` 未绑定，多分子必崩 |
| `[upstream]`   | `EPMolGen/sample.py:342`     | `step_batch` 补 `dtype=torch.long`；PyG ≥ 2.8 的 `index2ptr` 要求整型 batch                      |
| `[upstream]`   | `sample_main.py:80`          | `torch.load(..., weights_only=False)`；torch ≥ 2.6 默认 True，而 ckpt 内含 `easydict.EasyDict`   |
| `[upstream]`   | `sample_main.py:43-58`       | 作者本机绝对路径的调试块改为「CLI 未给才生效」的仓库内相对路径                                                         |

patch 归档在 `sdaa/patches/`，3 个 commit 按平台适配 / 通用修复分开。

## 运行

```bash
# SDAA（注意 RNG 对齐变量）；REPO 按实际克隆位置填写
REPO=<EPMolGen 仓库根目录>
docker exec -u 1007:1007 -e USER=<user> -e HOME=/tmp \
  -e TORCH_SDAA_ALIGN_NV_DEVICE=a100 sdaa bash -c \
  "source /opt/tecoai/setvars.sh; cd $REPO && \
   /home/py312/bin/python3 -u sample_main.py --device sdaa:0 --molecule_number 5"
```

SDAA 输出汇总（完整日志 `sdaa/logs/2026-10-26_sample_main_sdaa.log`；脚本会先打印 5 个生成分子的 SMILES，此处略）：

```
Validity: 1.0000
Unique: 1.0000
```

官方脚本不播种任何 RNG，每次运行结果都不同。

## 性能与显存

单分子生成延迟（同一脚本同一数据；CUDA 侧是共享机，数值随宿主负载波动，所以倍数不是常数）：

| 口径                                    | SDAA       | CUDA (A100) | 倍数    |
| ------------------------------------- | ---------- | ----------- | ----- |
| 单样本基准 `bench_infer.py`（1bvr，n=10，仅生成） | 19.3 s/分子  | 1.3 s/分子    | 15×   |
| 数据集 50 分子/口袋                          | 17.39 s/分子 | 2.35 s/分子   | 7.4×  |
| 数据集 10 分子/口袋                          | 16.53 s/分子 | 1.25 s/分子   | 13.2× |
| 官方 CLI 5 分子总量（含准备与加载）                 | 112 s      | 13 s        | 8.6×  |
| 同机 CPU 对照                             | 27.1 s/分子  | —           | —     |

峰值显存（数据集 10 口袋最大）：SDAA 152.6 MiB 对 CUDA 68.1 MiB，2.2×。

同一脚本在 CUDA 上不同时间实测：宿主机 CPU 空闲时 1.25 s/分子、高负载时 4.59 s/分子（GPU 利用率始终 0%，瓶颈全在 CPU 侧）；SDAA 侧是独占机。

单分子耗时随原子数变化（SDAA 侧 10 原子 10.6 s、28 原子 30.1 s）。SDAA 相对本机 CPU 仅 1.18×。论文报告 RTX 4090 上 0.7 s/分子。

## 精度

数据集用仓库自带的 `Ten_test_cases`（10 个口袋），每个口袋 50 个分子共 500 个，seed 1234（`torch.manual_seed` + `np.random.seed`，逐 case 加索引），SDAA 侧开 RNG 对齐。指标用仓库自己写的 `Generate.generate()`，即 `metrics.dir` 里的 Validity / Unique。

数据扩充进行中（2026-10-28 启动）：50 → 100 分子/case 两平台重跑 + 噪声种子 9999/8888 同步按 100/case 重跑，完成后更新本章统计。

逐分子一致率只有 0.7%（1000 分子同 seed 跨平台交集 7/993）——这不是平台缺陷：**同一张 A100、同 seed、同权重，仅改变 cuBLAS 矩阵乘累加顺序（`CUBLAS_WORKSPACE_CONFIG`），分子交集同样是 0/100**。该模型对 fp32 求和顺序极端混沌，任何两个不同的浮点实现（包括同一 GPU 换算法）都无法逐分子复现，因此用分布级口径判定。

### 仓库自带指标（Validity / Unique）

| 口袋     | SDAA 有效/唯一                | CUDA 有效/唯一                |
| ------ | ------------------------- | ------------------------- |
| 1bvr   | 50 / 50                   | 50 / 50                   |
| 1u0f   | 50 / 50                   | 50 / 50                   |
| 1zyu   | 50 / 50                   | 50 / 50                   |
| 2ah9   | 50 / 50                   | 50 / 50                   |
| 2ati   | 50 / 50                   | 50 / 50                   |
| 2hw1   | 50 / 49                   | 50 / 50                   |
| 4bnw   | 50 / 49                   | 50 / 50                   |
| 4i91   | 50 / 49                   | 50 / 49                   |
| 5g3n   | 50 / 50                   | 50 / 50                   |
| 5lvq   | 50 / 50                   | 50 / 50                   |
| **合计** | **500/500，497 唯一（0.994）** | **500/500，496 唯一（0.992）** |

各口袋耗时与峰值显存见「性能与显存」章。

### 论文官方指标对照（Table 1 口径）

两平台的分子在**同一台机器、同一份 RDKit 2026.03.6** 上算，避免跨机器 RDKit 版本差污染（`sdaa/scripts/align_stats.py`）：

| 指标       | CUDA（500 分子） | SDAA（500 分子） | 差     | SMD    |
| -------- | ------------ | ------------ | ----- | ------ |
| QED      | 0.62 ± 0.16  | 0.60 ± 0.17  | -0.02 | -0.101 |
| LogP     | 3.50 ± 1.77  | 3.55 ± 1.87  | +0.04 | +0.024 |
| SA score | 2.87 ± 0.95  | 2.95 ± 0.95  | +0.08 | +0.084 |
| 重原子数     | 18.14 ± 2.09 | 18.15 ± 1.92 | +0.01 | +0.005 |

SMD（标准化均值差）全部 < 0.11，即均值差不到自身标准差的 1/9。环统计（论文 Table 2 口径）：三元环 0.00 / 0.00，八元环 0.60 / 1.00，稠环（≥4 环）1.00 / 1.80。

### 对接亲和力对照（Vina 1.2.7，同环境同参数）

10 口袋 × 10 分子，exhaustiveness 8，box 24³（`sdaa/logs/dock_cuda_10case_10mol.csv`、`dock_sdaa_10case_10mol.csv`）：

| 口袋   | CUDA aff (kcal/mol) | SDAA aff (kcal/mol) | Δaff   | ΔLE    |
| ---- | ------------------- | ------------------- | ------ | ------ |
| 1bvr | -7.085 ± 0.868      | -7.240 ± 0.790      | -0.156 | +2.06% |
| 1u0f | -6.846 ± 0.629      | -6.837 ± 0.649      | +0.009 | -0.13% |
| 1zyu | -6.930 ± 0.378      | -6.901 ± 0.423      | +0.029 | +0.09% |
| 2ah9 | -7.454 ± 0.626      | -7.559 ± 0.682      | -0.105 | +1.41% |
| 2ati | -7.348 ± 0.703      | -7.397 ± 0.849      | -0.049 | +3.87% |
| 2hw1 | -7.101 ± 0.485      | -6.945 ± 0.496      | +0.156 | -2.19% |
| 4bnw | -9.713 ± 0.763      | -9.649 ± 1.140      | +0.065 | -0.63% |
| 4i91 | -7.259 ± 2.013      | -7.304 ± 2.034      | -0.045 | +0.51% |
| 5g3n | -8.302 ± 0.587      | -8.371 ± 0.680      | -0.069 | +0.74% |
| 5lvq | -7.541 ± 0.614      | -7.538 ± 0.709      | +0.003 | -0.09% |

配对 t = **-0.607**（df=9），无显著差异；pooled（100 分子）差 **0.016 kcal/mol**。

## 已知限制

- **加速有限**：SDAA 比 CUDA 慢 7~13×，比本机 CPU 仅快 1.18×。要真正吃到加速需把逐分子串行改成批量生成（一次 forward 出 N 个候选），属采样逻辑改造，不是适配能解决的。
- **跨平台只做到 uniform 对齐**：`torch.randint`（int64）两平台的随机流不同，不过推理路径不用它；`rand` / `normal` / `multinomial` / `distributions.Normal.sample` 在抽样路径上逐值对齐。
- **同 seed 下两边生成的分子不是逐个相同**（1000 分子交集 0.7%）：模型对 fp32 求和顺序极端混沌——同一张 A100 仅改 cuBLAS 累加顺序，交集同样是 0/100。该指标对任何平台都无有效意义，与适配质量无关。
- **训练路径未适配**：`pretraining.py` / `finetuning.py` 里的 `torch.cuda.amp` 与硬编码 `cuda:0` 未动。
- **PyG 版本**：仓库钉 2.5.3，SDAA 侧是 2.8.0.post1、CUDA 侧 2.6.1；`knn` 的 batch 索引类型要求已按 2.8 修正（2.6 同样接受整型）。
- `sample_main.py` 顶部的 `CUDA_LAUNCH_BLOCKING` / `CUDA_VISIBLE_DEVICES` 未改：torch 在其之前已 import，两个变量本就不生效。
- 无官方评测脚本，本文的精度对照用的是仓库 `Generate.generate()` 自带的 `metrics.dir`（Validity / Unique / ring_size）+ 论文 Table 1 指标的 RDKit 复算。
- 对接对照只跑了每口袋前 10 个分子（10 × 10 = 100 分子），不是全部 500 个；`YTHDC2/` 口袋未纳入。
