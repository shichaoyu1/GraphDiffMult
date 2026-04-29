# 多模态脑肿瘤融合：扩散模型 + 图学习

BraTS MRI 数据上的最小可运行演示，适用于组会汇报展示。

## 核心思路

```
T1 ──┐                        ┌─ 扩散模型（DDPM）
     ├─▶ Encoder ─▶ GAT图网络 ─┤    捕捉模态间互补生成关系
FLAIR┤                        └─ 图对比损失
     │                             建模模态间拓扑关系
T1ce┘
         └──── 融合分类 ────▶ 背景 / 水肿 / 增强肿瘤
```

| 模块 | 作用 |
|------|------|
| `ModalityEncoder` | 轻量 CNN，提取单模态 patch 特征 |
| `GraphModalityEncoder` | 两层 GAT，建模三模态拓扑关系，输出注意力权重 |
| `MultimodalDiffusion` | 条件 DDPM，用其余模态预测/生成目标模态特征 |
| `MultimodalFusionNet` | 整体网络，联合三路损失端到端训练 |

## 文件结构

```
brats_fusion/
├── model.py        # 模型定义（编码器 / GAT / 扩散 / 融合网络）
├── dataset.py      # BraTS 数据加载 & Patch 采样
├── visualize.py    # 全套可视化（切片/训练结果/扩散/图拓扑）
├── train.py        # 主入口（CLI）
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

下载 BraTS2021 任意病人文件夹，解压后包含：
```
BraTS2021_00060/
  BraTS2021_00060_t1.nii.gz
  BraTS2021_00060_t2.nii.gz
  BraTS2021_00060_flair.nii.gz
  BraTS2021_00060_t1ce.nii.gz
  BraTS2021_00060_seg.nii.gz
```

### 3. 运行

```bash
# 最简运行（默认参数）
python train.py --patient_dir /path/to/BraTS2021_00060

# 自定义参数
python train.py \
    --patient_dir /path/to/BraTS2021_00060 \
    --out_dir     ./results \
    --epochs      100       \
    --batch_size  16        \
    --n_patches   300       \
    --patch_size  32        \
    --feat_dim    64        \
    --lr          3e-4      \
    --lambda_diff  0.1      \
    --lambda_graph 0.05

# 强制 CPU（无 GPU 时）
python train.py --patient_dir /path/to/BraTS2021_00060 --cpu
```

### 4. 输出

全部保存到 `--out_dir`（默认 `./output/`）：

| 文件 | 内容 |
|------|------|
| `brats_slices.png` | 四模态（T1/T2/FLAIR/T1ce）三视图切片 + 分割叠加 |
| `brats_results.png` | 训练损失曲线 / 预测概率 / GAT注意力热图 / 扩散轨迹 / PCA投影 |
| `diffusion_process.png` | 前向加噪轨迹 / 特征分布演变 / 逆向去噪轨迹 |
| `graph_topology.png` | 模态有向图 / 四头注意力热图 / GNN前后PCA |

## 损失函数

```
L_total = L_task + λ₁·L_diffusion + λ₂·L_graph

L_task       : 交叉熵（背景/水肿/增强肿瘤三分类）
L_diffusion  : 条件 DDPM 去噪 MSE（模态互补生成）
L_graph      : 图对比学习（模态拓扑对齐）
```

## 环境要求

- Python ≥ 3.9
- PyTorch ≥ 2.0（CPU 版即可运行）
- 内存 ≥ 8 GB（加载 NIfTI 体数据）
- 无需 GPU，CPU 上 60 epoch 约 3–5 分钟
