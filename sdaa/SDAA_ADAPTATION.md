# EPMolGen SDAA 适配

## 结论

官方推理脚本 `sample_main.py` 在 SDAA 上直接可跑，5 个生成分子全部有效（脚本自带统计 `Validity 1.0000` / `Unique 1.0000`）。

但 SDAA 相对本机 CPU 基本没有加速（22.9 s/分子 vs 27.1 s/分子）：瓶颈在 CPU 侧的逐原子图重建与 RDKit 采样，卡上只占 124.7 MiB 显存。

## 环境

| 项 | 值 |
| --- | --- |
| 宿主 | 10.71.15.115（Loongson-3C6000/D，32 张 SDAA） |
| 容器 | `sdaa`（镜像 `3.2.1-torch_sdaa3.2.1`） |
| Python | `/home/py312/bin/python3`（3.12.13） |
| torch | 2.12.0 / Torch-SDAA 3.2.1 |
| 设备 | `sdaa:0`（单核组 16 GiB） |
| 仓库 | `/data/application/xuqiang/EPMolGen`，分支 `adapt/sdaa` |

依赖全部用容器自带，**未额外安装任何包**：`torch_scatter 2.1.2`、`torch_cluster 1.6.3`、`torch_sparse 0.6.18`、`pyg_lib 0.7.0`、`pymol`、`rdkit`。PyG 实装 2.8.0.post1（仓库 `requirements.txt` 钉 2.5.3）。

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
docker exec -u 1007:1007 -e USER=xuqiang -e HOME=/data/application/xuqiang sdaa bash -c \
  'source /opt/tecoai/setvars.sh; cd /data/application/xuqiang/EPMolGen && \
   /home/py312/bin/python3 -u sample_main.py --device sdaa:0 --molecule_number 5'
```

输出（节选）：

```
CCC=CC=CC(C)CCc1ccccc1OC
C=CC=CCCCCC(COP(=O)(O)O)C(=O)O
...
5
5
Validity: 1.0000
Unique: 1.0000
```

采样为逐原子随机过程，未设随机种子；每次生成的分子不同。产物在 `gen_results_sdaa/1bvr/<时间戳>/`（`generated.sdf` / `generated.smi`）。

## 性能与显存

| 场景 | 结果 |
| --- | --- |
| 官方 CLI（5 分子，含数据准备与加载） | 112 s |
| 生成阶段（`sdaa/scripts/bench_infer.py`，5 分子） | 22.9 s/分子（10.6 ~ 30.1） |
| 同脚本 CPU 对照（3 分子） | 27.1 s/分子（21.6 ~ 31.5） |
| 峰值显存 | 124.7 MiB |

单分子耗时随生成分子数变化（10 原子 10.6 s，28 原子 30.1 s）：每加一个原子都要跑一次 `self.run()`，即重建 KNN 图（`GraphCompose`，CPU 侧 16 进程）+ RDKit 建分子，真正上卡的 forward 只有几百个节点。

## 精度

- 本任务没有预测值可对，采样又是随机的，不存在逐比特对照基准。
- 可用的对照是仓库脚本自带的生成分布统计：5 个分子 `Validity 1.0000`、`Unique 1.0000`（`sdaa/scripts/bench_infer.py` 另测 5/5 SMILES 可解析）。
- **与 CUDA 的同权重对照未做**（6.21 上需另装 torch 2.2.1+cu118 与 PyG 2.5.3），因此 SDAA 与 CUDA 的数值一致性属未验证项。

## 已知限制

- **加速有限**：22.9 s/分子 对本机 CPU 27.1 s/分子，约 1.18×。论文报告 RTX 4090 上 0.7 s/分子，但那部分差距主要在 CPU 侧 Python/RDKit，不是卡算力差距。要真正吃到加速需把逐分子串行改成批量生成，属采样逻辑改造。
- **训练路径未适配**：`pretraining.py` / `finetuning.py` 里的 `torch.cuda.amp` 与硬编码 `cuda:0` 未动。
- **PyG 版本**：仓库钉 2.5.3，容器是 2.8.0.post1，`knn` 的 batch 索引类型要求已按 2.8 修正。
- `sample_main.py` 顶部的 `CUDA_LAUNCH_BLOCKING` / `CUDA_VISIBLE_DEVICES` 未改：torch 在其之前已 import，两个变量本就不生效。
