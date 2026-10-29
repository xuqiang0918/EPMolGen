# EPMolGen SDAA vs CUDA 对齐统计报告

日期：2026-10-26
范围：10 个口袋 × 50 分子 = 500 分子（SDAA 与 CUDA 各一份），seed 1234

---

## 结论

**对齐。** 四条独立证据：

| 证据 | 口径 | 结果 |
| --- | --- | --- |
| 确定性前向逐元素对照 | 同输入同权重，float64 | 差到 1e-16（机器精度） |
| 论文官方指标分布 | 10 case × 50 分子，SMD | 4 项指标 SMD 全部 < 0.11 |
| 分布级显著性检验 | 10 case × 4 指标 = 40 组，Welch t + BH-FDR，对照 2 组噪声种子 | 平台组 p<0.05 比例 **0.0%**，噪声组 3.8%，配对 t p=**0.85** |
| 对接亲和力 | 10 case × 10 分子，Vina 1.2.7 | 配对 t = **-0.607**，pooled 差 **0.016 kcal/mol** |

**核心结论：SDAA 与 CUDA 的差异没有超过「同一台 CUDA 换个随机种子」带来的差异**——平台差异被模型自身的采样随机性淹没。

### 数据规模

| 平台 | 口径 | 规模 |
| --- | --- | --- |
| SDAA | 10 口袋 × 50 分子，seed 1234，`TORCH_SDAA_ALIGN_NV_DEVICE=a100` | 500 分子，Validity 1.0000，Unique 497/500 |
| CUDA | 同上，seed 1234 | 500 分子，Validity 1.0000，Unique 496/500 |
| CUDA 噪声种子 | seed 9999 / 8888，各 10 口袋 × 50 分子 | 2 × 500 分子，用作噪声地板 |
| 对接 | 每平台每口袋取前 10 个分子 | 各 100 分子 |

---

## 1. 精度对照（判定：ALIGNED）

四条证据：确定性前向 → 官方指标 → 显著性检验/噪声地板 → 对接亲和力，全部指向同一结论。

### 1.1 判据：为什么不用逐分子一致率

EPMolGen 是自回归逐原子采样，`multinomial` 每一步都放大上游微小差异。实测：

- float32 下两平台前向差 1e-6~1e-5（`sdaa/forward_compare.txt`）
- **同一台 CUDA** 上仅把权重扰动 1e-5，10 个分子里就有 4 个变了；扰动 1e-4 只剩 1 个相同（`sdaa/chaos_check.txt`）
### 1.1.1 fp32 差异的产生与放大环节（逐算子 + 逐层探针，2026-10-28）

探针脚本 `sdaa/scripts/probe_fp32_sources.py`（ops / layers 两模式，产物 `sdaa/probe_{ops,layers}_{sdaa,cuda}.npz`，bitwise 对比）。

**起点干净**：加载的权重与构造的输入 bitwise 完全一致（weights_flat / cpx_pos / cpx_feature / cpx_edge_index 均 0 diff）——差异只能来自计算环节。

**算子级**（确定性输入，逐 op 比对）：

| 算子 | bitwise 不同的元素 | max_abs | 说明 |
| --- | --- | --- | --- |
| linear / gemm | 92% / 94% | 8.1e-6 / 1.1e-6 | GEMM 累加顺序与分块不同（cuBLAS vs TecoBLAS），第一个分叉源 |
| scatter_add | 85% | 4.8e-7 | 归约顺序不同；CUDA 实现（atomics）两次运行结果都不同，SDAA 实现确定 |
| exp | 49% | 1.2e-4（≈1 ulp） | 超越函数各家实现末位舍入不同 |
| softmax / scatter_softmax | — | 4.7e-9 / 1.1e-8 | 归约 + exp 叠加 |
| layernorm / norm / sigmoid / tanh / gelu | — | 6e-8 ~ 3.6e-7 | 同上，均 ~1 ulp 级 |

**逐层放大链**（1bvr 确定性前向，n=119840）：

| 阶段 | bitwise 不同元素 | max_abs |
| --- | --- | --- |
| embed（第一个 GEMM） | 9% | 2.4e-7 |
| enc0（第 1 个 interaction block） | 89% | 7.0e-6 |
| enc5（第 6 层） | 95% | 4.2e-5 |
| pivotal_prob | 97% | 1.3e-6 |

结论：差异在**第一个 GEMM（embed 阶段）产生**，由 encoder 的 6 层 AttentionInteractionBlock 逐层放大（每层 = GEMM + scatter 聚合 + 超越函数，把上游差继续叠加），到 h_enc 达到 4e-5 量级，multinomial 采样再把它放大成分子分叉。全部是 fp32 舍入顺序效应——fp64 下同样探针差 1e-16（机器精度），无任何系统性偏差。

**自确定性**：SDAA 全部算子两次运行 bitwise 一致；CUDA 的 scatter_add / scatter_softmax 因 atomics 两次运行有 ~1 ulp 差，但端到端不影响——CUDA 同 seed 双跑 1bvr×10 分子 SMILES 完全一致（实测）。

- 同 seed 跨平台逐分子交集（100 分子/case × 10 case，RDKit 规范化后）：全集 **7/993 = 0.7%**（`sdaa/scripts/smiles_sdaa.txt` 对 6.21 `smiles_cuda.txt`）
- **决定性校准（2026-10-28）**：同一张 A100、同 seed、同权重、同一份 RDKit，仅改 cuBLAS 矩阵乘累加顺序（`CUBLAS_WORKSPACE_CONFIG=:16:8`），1bvr × 100 分子交集 **0/100**——同 GPU 换个求和顺序都无法逐分子复现，跨平台 0.7% 完全落在 fp32 混沌包络内
- 注：早期 50/case 口径的 15.2%（76/500）出自 `cee35a7`（sample_tool RDKit 修复）之前的代码，且两机 RDKit 版本不同（2026.03.5 / 2026.03.6，采样循环内直接调 RDKit），与 0.7% 不可比；逐分子一致率被非张量的软件差异主导，彻底排除出判据

所以逐分子一致率在这个模型上不是有效判据，只能做分布级比较。

### 1.2 前提：输入与随机数两边逐位一致

确定性前向对照的**输入构造**逐位一致：`cpx_pos` / `cpx_feature` / `cpx_edge_index` 三项 max_abs = 0，bitwise YES（`sdaa/forward_compare.txt`）——分歧只可能产生在计算层。

随机数在评测路径中被显式播种并实测钉死。独立播种的 RNG 探针（`sdaa/scripts/rng_probe2.py`，每项单独 `torch.manual_seed(1234)`，结果 `sdaa/rng_probe2_result.txt`）：

| 取样 | SDAA 对齐 a100 | CUDA (A100) | 一致 |
| --- | --- | --- | --- |
| `torch.rand` | 逐位相同 | 基准 | 是 |
| `torch.randn` | 相同（float32 末位 ~1e-7） | 基准 | 是 |
| `torch.normal` | 相同（float32 末位 ~1e-7） | 基准 | 是 |
| `torch.multinomial` | 相同 | 基准 | 是 |
| `distributions.Normal.sample` | 逐位相同 | 基准 | 是 |
| `torch.randint`（int64） | 不同 | 基准 | 否 |

**torch 推理路径用到的取样只有 `multinomial` 与 `distributions.Normal.sample`（均逐位对齐）**；int64 `randint` 不对齐，但推理路径不调用它。

**numpy 随机数在评测路径中被显式播种并逐位一致**（这是"随机数是否影响结果"的直接证据）：

1. 评测脚本对每个 case 同时钉死两个 RNG：`torch.manual_seed(seed+idx)` + `np.random.seed(seed+idx)`（`sdaa/scripts/eval_dataset.py:98-101`）。每分子的原子数来自 numpy 全局流（`sample.py:24` 的 `np.random.normal`），因此被钉死。
2. numpy MT19937 是平台无关的确定性算法——实测 115（numpy 2.2.6）与 6.21 同 seed 输出序列**逐位一致**。
3. 同平台复现实测：SDAA 上同 seed 同 case 双跑（`gen_results_rng_r1/r2`，1bvr × 10 分子），10 个 SMILES **完全一致**（`diff` 通过，`sdaa/logs/rng_repro_r1.smiles` vs `r2.smiles`）。

结论：**随机数不影响对齐判定**——同 seed 下两边收到的随机输入完全相同，跨平台分子差异全部来自 float32 运算的混沌放大（已被 §1.4 噪声地板覆盖）。

> 早前一轮探针曾报「normal 路径不对齐」，那是探针自身的问题：连续调用多个取样而没有各自重新播种，先前调用的 int64 `randint` 消耗掉的随机流让后续结果错位。改为每项独立播种后结论反转。

---

### 1.3 论文官方指标对照（Table 1 口径）

数据源：`gen_results_dataset_cuda_50`（CUDA seed 1234）、`gen_results_dataset_sdaa_50`（SDAA seed 1234，`TORCH_SDAA_ALIGN_NV_DEVICE=a100`）。
**两边的分子在 6.21 上同一份 RDKit 2026.03.6 上算指标**，避免跨机器 RDKit 版本差污染。

| case | QED cuda/sdaa | LogP cuda/sdaa | SA cuda/sdaa | 重原子 cuda/sdaa | Validity cuda/sdaa |
| --- | --- | --- | --- | --- | --- |
| 1bvr | 0.63 / 0.62 | 3.49 / 3.24 | 2.94 / 2.89 | 18.3 / 18.4 | 100% / 100% |
| 1u0f | 0.53 / 0.49 | 1.08 / 0.81 | 3.53 / 3.65 | 18.1 / 18.2 | 100% / 100% |
| 1zyu | 0.57 / 0.53 | 1.57 / 1.27 | 3.55 / 3.76 | 18.4 / 18.2 | 100% / 100% |
| 2ah9 | 0.62 / 0.63 | 3.50 / 3.57 | 2.91 / 3.01 | 18.2 / 18.1 | 100% / 100% |
| 2ati | 0.56 / 0.54 | 3.85 / 4.25 | 2.93 / 2.90 | 18.1 / 17.9 | 100% / 100% |
| 2hw1 | 0.69 / 0.64 | 4.25 / 4.56 | 2.48 / 2.52 | 18.5 / 18.5 | 100% / 100% |
| 4bnw | 0.60 / 0.60 | 4.91 / 4.89 | 2.63 / 2.56 | 18.6 / 18.5 | 100% / 100% |
| 4i91 | 0.65 / 0.66 | 3.82 / 4.06 | 2.99 / 3.00 | 16.9 / 17.2 | 100% / 100% |
| 5g3n | 0.66 / 0.66 | 4.27 / 4.43 | 2.35 / 2.58 | 17.9 / 18.2 | 100% / 100% |
| 5lvq | 0.69 / 0.65 | 4.28 / 4.39 | 2.41 / 2.66 | 18.4 / 18.4 | 100% / 100% |

pooled（500 分子）：

| 指标 | CUDA | SDAA | 差 | SMD |
| --- | --- | --- | --- | --- |
| QED | 0.62 ± 0.16 | 0.60 ± 0.17 | -0.02 | -0.101 |
| LogP | 3.50 ± 1.77 | 3.55 ± 1.87 | +0.04 | +0.024 |
| SA | 2.87 ± 0.95 | 2.95 ± 0.95 | +0.08 | +0.084 |
| 重原子数 | 18.14 ± 2.09 | 18.15 ± 1.92 | +0.01 | +0.005 |

SMD（标准化均值差）全部 < 0.11，即均值差不到自身标准差的 1/9。

仓库自带 Validity / Unique（`metrics.dir`）：SDAA 500/500 有效、497/500 唯一（0.994）；CUDA 500/500 有效、496/500 唯一（0.992）。

环统计（论文 Table 2 口径，% 分子）：三元环 0.00 / 0.00，八元环 0.60 / 1.00，稠环（≥4 环）1.00 / 1.80。

---

### 1.4 分布级显著性检验 + 噪声地板

每个 (case, metric) 做两样本 Welch t 检验（n=50 vs n=50）：
- **A 组（平台差异）**：SDAA vs CUDA 同种子，40 组
- **B 组（噪声地板）**：CUDA 换种子（seed 9999 / 8888）vs CUDA 同种子，80 组

| 组 | n | mean\|t\| | median p | p<0.05 比例 | BH-FDR 后 q<0.05 |
| --- | --- | --- | --- | --- | --- |
| A 平台差异 | 40 | 0.653 | 0.597 | **0.0%** | 0 / 40 |
| B 噪声地板 | 80 | 0.699 | 0.581 | 3.8% | 0 / 80 |

配对 t（对 |t| 差）：mean +0.021，t = 0.196，**p = 0.85** → 两组不可区分。

**判定：ALIGNED。**

---

### 1.5 对接亲和力对照（Vina 1.2.7，同环境同参数）

数据：`sdaa/logs/dock_cuda_10case_10mol.csv`、`sdaa/logs/dock_sdaa_10case_10mol.csv`，10 case × 10 分子，exhaustiveness 8，box 24³。

| case | CUDA aff (kcal/mol) | SDAA aff (kcal/mol) | Δaff | ΔLE |
| --- | --- | --- | --- | --- |
| 1bvr | -7.085 ± 0.868 | -7.240 ± 0.790 | -0.156 | +2.06% |
| 1u0f | -6.846 ± 0.629 | -6.837 ± 0.649 | +0.009 | -0.13% |
| 1zyu | -6.930 ± 0.378 | -6.901 ± 0.423 | +0.029 | +0.09% |
| 2ah9 | -7.454 ± 0.626 | -7.559 ± 0.682 | -0.105 | +1.41% |
| 2ati | -7.348 ± 0.703 | -7.397 ± 0.849 | -0.049 | +3.87% |
| 2hw1 | -7.101 ± 0.485 | -6.945 ± 0.496 | +0.156 | -2.19% |
| 4bnw | -9.713 ± 0.763 | -9.649 ± 1.140 | +0.065 | -0.63% |
| 4i91 | -7.259 ± 2.013 | -7.304 ± 2.034 | -0.045 | +0.51% |
| 5g3n | -8.302 ± 0.587 | -8.371 ± 0.680 | -0.069 | +0.74% |
| 5lvq | -7.541 ± 0.614 | -7.538 ± 0.709 | +0.003 | -0.09% |

- 配对 t（case 为配对单元，df=9）：mean Δ = **-0.0162**，t = **-0.607** → 无显著差异
- max \|Δaff\| = 0.156 kcal/mol，全部落在各自 case 的 0.5σ 以内
- pooled（100 分子）：SDAA **-7.574 ± 1.249** vs CUDA **-7.558 ± 1.202**，差 **0.016 kcal/mol**

两个极端 case 也一致：4bnw（亲和力最强 -9.7）差 0.065；4i91（方差最大 σ=2.0）连标准差都几乎相同。

---

## 2. 性能与峰值显存（单样本推理，CUDA vs SDAA）

`bench_infer.py`（`sdaa/scripts/`，两平台同一脚本同一 pocket 1bvr，n=10，只计 `run()` 生成循环、不含 RDKit 后处理）：

| 平台 | 单分子延迟 | min~max | 峰值显存 |
| --- | --- | --- | --- |
| CUDA (A100) | **1.3 s** | 0.6~2.1 s | 56.8 MiB |
| SDAA | **19.3 s** | 10.6~27.6 s | 123.6 MiB |

- 延迟比 **≈15×**，峰值显存比 2.2×。
- 数据集评测端到端口径（含逐分子 RDKit 后处理）：CUDA 3.7~5.3 s/mol（后处理占比大），SDAA 19.2 s/mol。500 分子全量：CUDA ~1 h，SDAA ~2 h。
- 日志：`sdaa/logs/bench_sdaa.log`、6.21 `EPMolGen_cuda/sdaa/logs/bench_cuda.log`。

峰值显存 2.2× 的拆解（`probe_mem.py`，两平台同脚本；模型仅 0.9 MiB 参数，峰值由单分子生成的激活/临时张量主导）：

| 项 | CUDA | SDAA |
| --- | --- | --- |
| 权重（加载后 allocator） | 4.4 MiB | 4.4 MiB |
| 分配粒度（1×1MiB / 1000×1KiB） | 精确无放大 | 精确无放大 |
| 单分子激活+临时张量峰值 | **43.9 MiB** | **117.5 MiB** |
| 单分子 peak allocated / reserved | 56.4 / 76.0 MiB | 121.8 / 266.0 MiB |

即 2.2× 不是 allocator 粒度伪差（两边粒度都精确），而是 **SDAA 上生成过程的中间张量多 ~2.7×**——符合年轻后端缺少 kernel 融合、临时张量落盘更多的预期。绝对量级对 14.94 GiB/核组可忽略（0.8% vs 0.4%）。

## 3. 已知问题：非 0 设备索引崩溃（SDAA 特有）

同 seed（1234）同 case（1bvr）同脚本，`sdaa:0` 正常跑通（10/10 valid），而 `sdaa:4/8/12/16/20` 全部在第一个分子内崩溃：

```
RuntimeError: Class values must be non-negative.   # sample_tool.py:36, element < 0
```

CUDA 同 seed 正常。即 **SDAA 上多卡并行生成不可用，只能在 `sdaa:0` 串行**（10-26 全量恰好用 sdaa:0，未暴露）。根因未深查（疑平台侧 multinomial/取样在非零设备上下文的行为），待平台侧排查。

---

## 4. 复现命令

```bash
# 6.21 上跑，两边分子同一 RDKit；以下三个变量按实际位置填写
CUDA_REPO=<EPMolGen_cuda 仓库根目录>
export PYTHONPATH=<pylibs_epmolgen 目录>
PY=<surf 环境的 python>
cd $CUDA_REPO

# 分布级对照 + 双噪声种子判定
$PY sdaa/scripts/align_verdict2.py \
  --cuda  gen_results_dataset_cuda_50 \
  --sdaa  gen_results_dataset_sdaa_50 \
  --noise gen_results_dataset_cuda_seed50 gen_results_dataset_cuda_seed8888

# 论文官方指标对照
$PY sdaa/scripts/align_stats.py \
  --ref gen_results_dataset_cuda_50 --test gen_results_dataset_sdaa_50

# 对接
export PATH=<openbabel bin 目录>:$PATH
$PY sdaa/scripts/docking_batch.py \
  --root gen_results_dataset_sdaa_50 --cases Ten_test_cases --out-prefix results_dock_sdaa --limit 10
```

归档：`sdaa/logs/dock_cuda_10case_10mol.csv`、`sdaa/logs/dock_sdaa_10case_10mol.csv`、`sdaa/logs/results_dataset_sdaa_50.csv`、`sdaa/scripts/results_dataset_cuda_seed50.csv`、`results_dataset_cuda_seed8888.csv`。

---

## 5. 未覆盖项

- 训练路径（`pretraining.py` / `finetuning.py`）未适配，只做推理。
- YTHDC2 口袋（`YTHDC2/`）未纳入本次对照，只跑了 `Ten_test_cases`。
- 对接只跑了每 case 前 10 个分子（10 case × 10 分子 = 100 分子），不是全部生成分子。
- 噪声地板用了 2 个种子（9999 / 8888）。
- **数据扩充进行中（2026-10-28 启动）**：50 → **100 分子/case** 两平台重跑（seed 1234，SDAA 因非零设备 bug 只能 sdaa:0 串行，约 4 h），噪声种子 9999 / 8888 同步按 100 分子/case 重跑；完成后重跑 §1.3 / §1.4 统计与对接扩充。
