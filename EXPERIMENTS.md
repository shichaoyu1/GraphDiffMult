# UTSW-Glioma 图约束扩散实验入口

本项目现在保留三条路线：

- `train.py`：原始 BraTS/UTSW 单病例 patch 演示，用于可视化和快速检查数据兼容性。
- `train_utsw.py`：根据实验方案新增的病人级主实验入口，用于 IDH、MGMT、1p/19q 和 WHO grade 预测。
- `train_semantic_alignment.py`：当前推荐主线，用于病理锚定的多模态语义单元对齐；分级只作为后续下游验证。

## 当前推荐主线：语义单元对齐

当前阶段不把分级作为主任务，而是先验证 MRI 区域节点、病理锚点和分子语义节点是否能在同一个 shared semantic space 中对齐。

```powershell
python train_semantic_alignment.py `
  --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" `
  --variant full `
  --graph_type learnable `
  --epochs 30 `
  --batch_size 4 `
  --augment
```

快速 smoke test：

```powershell
python train_semantic_alignment.py `
  --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" `
  --variant full `
  --max_cases 8 `
  --epochs 1 `
  --batch_size 2 `
  --roi_size 32 `
  --z_slices 3 `
  --cpu
```

输出目录默认是 `output/semantic_alignment_experiment`，包含：

- `config.json`
- `anchor_vocab.json`
- `splits.json`
- `history.json`
- `best_semantic_alignment.pt`
- `test_metrics.json`
- `semantic_alignment_metrics.json`
- `semantic_unit_alignment_space.png`
- `semantic_unit_graph_50patients.png`
- `semantic_unit_adjacency.png`
- `semantic_unit_laplacian.png`

核心指标：

- `recall@1` / `recall@5` / `recall@10`
- `mrr`
- `pair_auc`
- `positive_negative_distance_gap`
- `anchor_consistency`
- `edge_precision@10` / `edge_precision@25` / `edge_precision@50`
- `pathology_unavailable`：测试时移除病理锚点后的对齐结果
- `molecular_unavailable`：测试时移除分子锚点后的对齐结果

当前语义节点：

| 来源 | 节点 |
|---|---|
| MRI 区域 | `Necrotic/Core`、`Edema`、`Enhancing` |
| 病理锚点 | `Tumor Grade`、`Tumor Type` |
| 分子锚点 | `IDH`、`MGMT`、`1p19Q CODEL` |
| 临床锚点 | 默认关闭，可用 `--include_clinical_anchors` 加入 |

语义对齐消融：

```powershell
python train_semantic_alignment.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --variant clip
python train_semantic_alignment.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --variant no_anchor
python train_semantic_alignment.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --variant graph_only
python train_semantic_alignment.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --variant modality_vector
python train_semantic_alignment.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --variant no_private
python train_semantic_alignment.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --variant no_graph
```

这些变体分别对应：

| 变体 | 检验点 |
|---|---|
| `full` | 语义单元 graph + 病理锚点 + shared/private + diffusion |
| `clip` | 普通 CLIP-style 对比学习基线 |
| `no_anchor` | 去除病理锚点，仅保留非病理语义锚点 |
| `graph_only` | 保留 graph，去除 diffusion |
| `modality_vector` | 回退为模态大向量节点 |
| `no_private` | 去除 private branch 与 diffusion |
| `no_graph` | 去除图结构约束 |

## AutoDL 一键调度

服务器上推荐使用 `run_all_pro.sh` 跑完整语义对齐消融。默认路径按 AutoDL 数据盘设计：

```bash
chmod +x run_all_pro.sh utils/gpu_logger.sh

DATA_ROOT=/root/autodl-tmp/UTSW-Glioma \
METADATA_TSV=/root/autodl-tmp/UTSW_Glioma_Metadata-2-1.tsv \
EPOCHS=30 \
BATCH_SIZE=4 \
SEEDS="42" \
nohup ./run_all_pro.sh > logs/nohup_master.log 2>&1 &
```

如果先做 smoke test：

```bash
DATA_ROOT=/root/autodl-tmp/UTSW-Glioma \
EPOCHS=1 \
VARIANTS="full clip" \
SEEDS="42" \
nohup ./run_all_pro.sh > logs/nohup_smoke.log 2>&1 &
```

调度脚本会自动生成：

- `logs/master_*.log`：总控日志
- `logs/gpu_*.csv`：GPU 利用率、显存、功耗、温度时间序列
- `output/semantic_*`：每个变体的模型、指标和可视化
- `results/summary_*.csv`：汇总后的指标表
- `results/table_*.tex`：可复制到论文的 LaTeX 消融表

默认消融包括：

```text
full clip no_anchor graph_only modality_vector no_private no_graph
```

可通过环境变量调整：

```bash
VARIANTS="full no_anchor graph_only" SEEDS="42 43 44" ./run_all_pro.sh
```

如果要顺带跑 `train_utsw.py` 的 IDH 下游 sanity check：

```bash
RUN_UTSW_SANITY=1 ./run_all_pro.sh
```

## 主实验定位

`train_utsw.py` 保留为分级/分子预测下游路线。当前实现采用最小可验证路线：

1. 使用 UTSW-Glioma 的分割 mask 裁剪 tumor ROI。
2. 从 T1、T1ce、T2、FLAIR 提取 2.5D ROI 输入。
3. 默认把 MRI 节点细化为三个最小语义单元：坏死/非增强核心、水肿区、增强区。
4. 编码后分解为 shared representation 和 private representation。
5. 图只用于 shared representation 的 Laplacian consistency regularization，不做 GNN message passing。
6. UTSW metadata 中的分子/分级标签作为 pathology-derived semantic anchors，目前只进入可视化图，不作为预测输入。
7. latent diffusion 只作用在 private latent 上，用于互补语义恢复。

## 快速 smoke test

```powershell
python train_utsw.py `
  --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" `
  --task idh `
  --variant full `
  --max_cases 8 `
  --epochs 1 `
  --batch_size 2 `
  --roi_size 32 `
  --z_slices 3 `
  --cpu
```

## 推荐主任务

```powershell
python train_utsw.py `
  --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" `
  --task idh `
  --variant full `
  --graph_type learnable `
  --epochs 30 `
  --batch_size 4 `
  --class_weight `
  --augment
```

输出目录默认是 `output/utsw_experiment`，包含：

- `config.json`
- `history.json`
- `best_model.pt`
- `test_metrics.json`
- `semantic_graph_initial.png`
- `semantic_laplacian_initial.png`
- `semantic_graph_trained.png`
- `semantic_adjacency_trained.png`
- `semantic_laplacian_trained.png`
- `semantic_alignment_trained.png`
- `semantic_alignment_metrics.json`

## 当前支持的任务

| 参数 | 标签来源 | 类别 |
|---|---|---|
| `--task idh` | `IDH` | wildtype / mutant |
| `--task mgmt` | `MGMT` | unmethylated / methylated |
| `--task 1p19q` | `1p19Q CODEL` | non-codeleted / codeleted |
| `--task grade` | `Tumor Grade` | WHO grade 2 / 3 / 4 |

## 消融路线

### Shared/private baseline

```powershell
python train_utsw.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --task idh --variant shared_private
```

### Graph consistency

```powershell
python train_utsw.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --task idh --variant graph --graph_type learnable
```

### Graph + anchor

```powershell
python train_utsw.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --task idh --variant anchor --graph_type learnable
```

### Full model

```powershell
python train_utsw.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --task idh --variant full --graph_type learnable
```

## 图构建对照

`--graph_type` 支持：

- `no_graph`
- `fixed`
- `similarity`
- `learnable`
- `random`

建议先跑：

```powershell
python train_utsw.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --task idh --variant graph --graph_type fixed
python train_utsw.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --task idh --variant graph --graph_type similarity
python train_utsw.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --task idh --variant graph --graph_type learnable
python train_utsw.py --data_root "D:\dataset\脑肿瘤数据集\公共数据集\UTSW-Glioma" --task idh --variant graph --graph_type random
```

## 语义节点图

`train_utsw.py` 默认使用：

```powershell
--node_mode regions
```

此时训练图节点为：

- `Necrotic/Core`
- `Edema`
- `Enhancing`

如果需要回退到旧的模态级节点，可显式使用：

```powershell
--node_mode modalities
```

初始语义图额外画出 pathology/gene anchor 节点，例如 WHO grade、IDH、MGMT、1p/19q。它们只用于解释图，不进入模型预测输入，避免标签泄漏。

训练结束后还会生成 `semantic_alignment_trained.png`。这张图默认从 train/val/test 合并抽取最多 50 个病人，把训练后的 MRI 语义单元和病理/基因监督原型投影到同一个 shared semantic space：

- 颜色表示来源：MRI、Pathology 或 Gene。
- 形状表示语义单元，例如坏死/核心、水肿、增强区、病理锚点、基因锚点。
- 灰色连线表示 MRI 病灶语义单元与对应 pathology/gene anchor 的对齐关系。

`semantic_alignment_metrics.json` 会保存各 MRI 语义单元到监督锚点的平均距离，便于后续做表格或训练前后比较。

可用下面两个参数控制图上的病人数量和来源：

```powershell
--align_split all --align_max_cases 50
```

## 仍待扩展

- 单模态和 concat baseline。
- cross-attention baseline。
- modality availability variation 测试入口。
- BraTS 预训练到 UTSW 微调流程。
- UMAP/t-SNE shared space 可视化和 IDH 分组邻接矩阵热力图。
