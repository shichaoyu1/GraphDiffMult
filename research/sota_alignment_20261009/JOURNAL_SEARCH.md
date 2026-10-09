# 近三年细粒度医学多模态对齐：期刊与实现核验

检索日期：2026-10-09；窗口：2024–2026 年正式发表成果。本记录是来源核验与适配设计，不是已完成的真实队列实验。已读取 PROJECT_STATE.md。

## 候选与真实发表状态

| 方法 | 已核实发表信息 | 核心 | 官方代码 | 本项目适配判断 |
|---|---|---|---|---|
| CARZero | CVPR 2024 | 双向跨注意力形成相似度表征，再线性映射为配对得分 | [laihaoran/CARZero](https://github.com/laihaoran/CARZero) | 首选；保留配对条件化跨注意力与可学习相似度投影，不能换成普通余弦后仍称 CARZero |
| RadZero | NeurIPS 2025 主会 | 文本条件化局部余弦注意力；局部相似度图；每个正例单独计算 MP-NCE | [deepnoid-ai/RadZero](https://github.com/deepnoid-ai/RadZero) | 首选；适合多个患者字段的监督，但 MRI 区域 token 并非原文 CXR patch，须称 metadata-adapted |
| AFLoc | Nature Biomedical Engineering，2026-01-06 正式上线；卷 10，1595–1609 | 多层语义结构对比，对齐词、句、报告与不同层图像表征 | [YH0517/AFLoc](https://github.com/YH0517/AFLoc)，默认分支 master | 优质相关工作；当前没有报告层次、完整局部影像特征，直接压缩到患者原型会丢失核心，不宜宣称完整复现 |
| MaCo | Nature Communications 15，7620；2024-09-02 | MAE 掩码高分辨率重建与掩码位置相关性加权对比 | [SZUHvern/MaCo](https://github.com/SZUHvern/MaCo) | 作为第二阶段候选；仅给三个区域特征随机 dropout 不等于其掩码 patch 重建与位置权重机制 |
| ALTA | IEEE TMI 2025，DOI 10.1109/TMI.2025.3575853 | 冻结 masked-model 预训练编码器并插入 adapter；全局/局部对齐，维持 MLM/MIM；时序多视图输入 | [DopamineLcy/ALTA](https://github.com/DopamineLcy/ALTA) | 当前无 MRM 预训练基础、时序报告和重建任务，不是本轮优先适配对象 |

正式状态来自出版社或主会论文页和作者仓库交叉核验。AFLoc 的 DOI 字符串含 025，预印本最初在 2024 年，但正式发表年是 2026；不能按 2024 顶刊引用。REFERS（ICCV 2021）超出本次窗口；MGCA（NeurIPS 2022）也不计入近三年方法。

## 两个首选方法的核心要求

CARZero 应保留“图像全局 query 对文本局部 tokens”和“文本全局 query 对图像局部 tokens”的双向分支，通过 trainable similarity representation 输出配对得分。为免跨注意力退化，文本至少使用真实模板 tokenizer 的 token 序列；单一患者 label 的 learned prototype 只提供一个 token，不能冒充完整文本局部对齐。MRI 编码器和预训练信息变动需明示。

RadZero 作者方法 §3.2.2 定义：图像 tokens 包含 global CLS 和 local patches；对视觉和文本向量分别 L2 归一化，得到 s_k = cosine(v_k,t) × exp(τ)。softmax(s_k) 在视觉 token 维上求权重，使用原始视觉 value 加权求和，再归一化 pooled value 与归一化文本计算最终得分。解释图来自直接余弦得分及 sigmoid；不能把 softmax 注意力图当作该相似度图。MP-NCE 的一个正例分母由该正例及真实负例组成，其他正例不作为竞争项。

官方实现入口：`exp/cxr_pt/model/align_transformers.py`、`losses.py`、`modeling.py`。已打开官方仓库与 LICENSE；RadZero 许可证为 CC-BY-NC 4.0，CARZero 为 Apache 2.0。本轮 `git ls-remote HEAD` 核实 RadZero 为 `656ae5f1af3f106e96c95542ce3ee5c0ee8777fc`，CARZero 为 `fa4e09cdfe7801e36e83d96815e3fcf66ffdd13d`。GitHub API 遭 rate limit，提交记录使用 Git remote 实际输出核实。

官方 RadZero `exp/cxr_pt/configs/radzero.yaml` 实际设 `sim_op: cos`、`attn_temperature: null`、`loss_temperature: 0.07`、`mpnce_row_sum: False`、`mpnce_col_sum: False`。本轮实际读取了该配置和 `losses.py` 的原始源码；采用官方源码时 query 和 local value 都先归一化，attention pooling 也使用归一化后的 local value。最终 pooled cosine 经 loss_temperature 缩放用于 NCE；attention_temperature 空时与 loss 共用该可学习参数。两方向损失取平均。作者 HTML 的未归一化 value 描述与公开代码存在细节差异，适配应选定并披露所遵从的源码版，不能混用。

## 本轮公平适配边界

本项目输入是 MRI 和患者结构化元数据，没有完整影像报告或区域组织学。所有比较须清楚标注 **metadata-adapted CARZero / metadata-adapted RadZero**。将各 field–value 转为固定模板，使用同一个真实冻结文本编码器及同一模板缓存；缓存对象是候选 field–value，不包含测试患者标签或报告。未知值保留掩码，不编造阳性、病灶位置或组织学描述。

同编码器比较应包含同一 MRI backbone、同一冻结文本表的普通余弦对比、CARZero 适配和 RadZero 适配。原 PASA learned-prototype 结果另列，不能把文本预训练效应解释成对齐算法优势；如评估 PASA 与新方法总体能力，同时报告训练参数量、预训练来源与预算，并明确其表征条件差异。

至少固定患者集合、split manifest、候选词表、目标定义、已知/未知掩码、ROI 输入、种子、训练轮数、checkpoint-selection 与计算预算。患者分子字段用于共同评价；区域定位只能是探索输出，缺乏区域真值时不能据其证明区域病理对齐。

两个适配对比能补充近期方法实验，但不自动满足对原版 CXR/报告 SOTA 的完整复现要求，也不能直接把文献公布的 CXR 指标与 MRI 检索数字排成同任务榜单。

## 已实际打开的主要来源

- CARZero [CVPR 2024 主会论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Lai_CARZero_Cross-Attention_Alignment_for_Radiology_Zero-Shot_Classification_CVPR_2024_paper.pdf)、[官方仓库](https://github.com/laihaoran/CARZero)。
- RadZero [NeurIPS 2025 主会记录](https://papers.neurips.cc/paper_files/paper/2025/hash/510e5c2fd5ad326f75594563f4ad5e0d-Abstract-Conference.html)、[作者方法 HTML](https://arxiv.org/html/2504.07416v2)、[官方仓库](https://github.com/deepnoid-ai/RadZero)、[LICENSE](https://github.com/deepnoid-ai/RadZero/blob/main/LICENSE)。正式 NeurIPS PDF 超过浏览工具大小限制；公式核验使用作者 HTML，发布版代码仍须交叉核对。
- AFLoc [出版社正式页面](https://www.nature.com/articles/s41551-025-01574-7)、[官方仓库](https://github.com/YH0517/AFLoc)。出版社正文订阅限制，已核日期、摘要、出处和代码；完整层级公式仍需官方源码或作者公开稿核验。
- MaCo [出版社全文](https://www.nature.com/articles/s41467-024-51749-0)、[官方仓库](https://github.com/SZUHvern/MaCo)。已读取 Methods 中 masked reconstruction 与 correlation-weighting 定义。
- ALTA [作者全文](https://arxiv.org/html/2506.08990v1)、[官方仓库](https://github.com/DopamineLcy/ALTA)、[IEEE DOI](https://doi.org/10.1109/TMI.2025.3575853)。IEEE/PubMed 网页受浏览提取限制；正式 TMI 2025 归属由公开 DOI、作者仓库与出版索引交叉核验。

本记录不宣称检索穷尽全部期刊。Nature Methods 与 MedIA 本轮未找到比上述方法更符合当前配对输入、代码公开和细粒度对齐核心的优先候选。
