# 2024–2026 细粒度多模态对齐方法核查

检索日期：2026-10-09。窗口按正式发表年 2024、2025、2026 计；预印本上传年、DOI 中年份、作者 README 更新日期均不直接替代正式发表年。本报告是有目的的候选筛选，不是完整系统综述，也不含尚未运行的性能结果。

## 已核查的候选

| 方法 | 已核查正式来源 | 细粒度机制及原输入 | 官方实现与本轮适配判断 |
|---|---|---|---|
| CARZero | [CVPR 2024 主会，11137–11146](https://openaccess.thecvf.com/content/CVPR2024/html/Lai_CARZero_Cross-Attention_Alignment_for_Radiology_Zero-Shot_Classification_CVPR_2024_paper.html) | 全局文本查询局部图像、全局图像查询局部词特征，cross-attention/FFN 得相似度表征，再线性打分；原输入是胸片和自由文本报告。 | [作者仓库](https://github.com/laihaoran/CARZero)，Apache-2.0。本轮优先适配其真实双向注意力评分模块，保留医学文本编码器及 token 特征；MRI 编码器、结构化候选文本、监督损失的改变须显式记录。 |
| RadZero | [NeurIPS 2025 主会](https://proceedings.neurips.cc/paper_files/paper/2025/hash/510e5c2fd5ad326f75594563f4ad5e0d-Abstract-Conference.html) | VL-CABS 用局部图像与文本的相似度建立跨注意力；LLM 提取 finding sentences，多正样本对比训练；胸片局部特征与句子输入。 | [作者仓库](https://github.com/deepnoid-ai/RadZero)。本轮优先适配真实 VL-CABS；用患者元数据候选句替代报告抽取句须披露，不能据此声称复现胸片论文的定位或零样本结论。 |
| MaCo | [Nature Communications 15,7620，2024-09-02](https://www.nature.com/articles/s41467-024-51749-0) | masked autoencoding 与图文对比结合，按保留图像 patches 对报告的重要性调整对比权重；胸片和报告。 | [作者仓库](https://github.com/SZUHvern/MaCo)。有真实代码；需要重建掩码、重构、correlation weighting 的训练管线，不能只以加一个 InfoNCE 当 MaCo。 |
| Med-ST | [ICML 2024，PMLR235:56382–56396](https://proceedings.mlr.press/v235/yang24v.html) | 多视角专家融合、modality-weighted token/region 对齐，以及时间循环一致性。 | [作者仓库](https://github.com/SVT-Yang/MedST)。UTSW 当前数据是 MRI 序列和患者元数据，未建立时间配对与完整报告；完整方法的输入条件缺失，本轮不作为完整复现。 |
| MLIP | [CVPR 2024 主会，11704–11714](https://openaccess.thecvf.com/content/CVPR2024/html/Li_MLIP_Enhancing_Medical_Visual_Representation_with_Divergence_Encoder_and_Knowledge-guided_CVPR_2024_paper.html) | divergence encoder 的全局对齐、token–knowledge–patch 局部对齐、知识辅助 category 对齐；需要医学知识库及报告。 | [论文指向的作者仓库](https://github.com/gentlefress/MLIP) 检索时只有 README/“Comming Soon”，不能称完整官方代码已发布。另有同名 ISBI2024 MLIP，不能混淆。 |
| FG-CLIP | [ICML 2025，PMLR267:68777–68793](https://proceedings.mlr.press/v267/xie25k.html) | 自然图像 long captions、region–text 配对和细粒度 hard negatives；含局部框监督。 | [作者仓库](https://github.com/360CVGroup/FG-CLIP)，旧版代码在 v1.0。适合补充通用 fine-grained 基线，但自然图像、框描述和大规模预训练与 MRI 不同；移除文本和区域训练后不能再命名为原方法复现。 |
| FG-CLIP 2 | [ICML 2026 官方下载列表](https://icml.cc/Downloads/2026)，[作者机构接收公告，2026-05-06](https://research.360.cn/en/blog/3557a26c-4bf4-44d6-b07e-9ab3e03197b6) | 双语 region–text 和 long captions 对齐，加入 textual intra-modal contrastive 目标区分相近文本。 | [作者仓库主分支](https://github.com/360CVGroup/FG-CLIP)，[作者预印本](https://arxiv.org/abs/2510.10921)。已核查 ICML2026 接收与官方列表；英文 README 把接收新闻写成 2025/05/02，与中文和机构公告不一致。 |
| AFLoc | [Nature Biomedical Engineering 10,1595–1609，2026-01-06](https://www.nature.com/articles/s41551-025-01574-7) | 多层语义结构对比，把报告中不同粒度的医学概念与图像特征对齐；胸片报告预训练，并评估其他模态。 | [作者仓库](https://github.com/YH0517/AFLoc)，Apache-2.0。原训练保留报告层次与多个视觉层次，当前短元数据句不能直接构成完整语义层次；不以单一局部对比替代完整 AFLoc。 |

## 本轮比较的可解释范围

CARZero 和 RadZero 的原版模型以胸片/报告为中心，直接运行胸片权重在脑 MRI 上，混入了模态差异、外部预训练和输入条件。第一轮建议用共同 MRI 输入、共同病例划分、相同候选描述与医学文本特征，比较真实对齐模块，并将结果明确标记为 **CARZero MRI adaptation**、**RadZero MRI adaptation**。这可以回答新对齐机制在当前患者元数据检索任务上的效果；不等于原论文在其原始任务/训练规模上的复现，也不直接证明区域组织学定位。

候选文本只能描述字段和值，固定词表由训练数据建立；不得包含病例身份、当前测试患者的其他信息或所谓专家确认的区域病理事实。缺失字段不能自动作为负例。所有方法评价同一候选库、同一正/负/未知掩码与同一冻结病例划分。

若最终要在正文标注“与最新完整 SOTA 方法比较”，需进一步补完整方法输入、预训练预算、文本编码器、目标损失、代码差异说明和复现验证；本轮模块适配结果应单独命名，以便评审判断公平性。

## CARZero 官方源码精读记录

源码 HEAD：`fa4e09cdfe7801e36e83d96815e3fcf66ffdd13d`，通过 `git ls-remote` 核对。源码浏览与论文方法均已读取。

- [dqn_wo_self_atten.py / TQN_Model](https://github.com/laihaoran/CARZero/blob/fa4e09cdfe7801e36e83d96815e3fcf66ffdd13d/CARZero/models/dqn_wo_self_atten.py)：输入 `image_features[B,L,D]` 与 `text_features[T,D]`；变为 `[L,B,D]` 和 `[T,B,D]`，共同 LayerNorm，四头无 query self-attention 的 transformer decoder，FFN 宽度 1024、ReLU、dropout 0.1；最后 LayerNorm、dropout 与 `Linear(D,1)`，输出 `[B,T,1]`。
- [transformer_decoder.py / TransformerDecoderWoSelfAttenLayer](https://github.com/laihaoran/CARZero/blob/fa4e09cdfe7801e36e83d96815e3fcf66ffdd13d/CARZero/models/transformer_decoder.py)：pre-norm 查询做 multihead cross-attention，加残差后做 pre-norm FFN 再加残差。类虽定义 `self_attn` 参数，实际无 self-attention 分支调用，不应据名称猜机制。
- [CARZero_model_dqn_wo_self_atten.py / CARZeroDQNWOSA](https://github.com/laihaoran/CARZero/blob/fa4e09cdfe7801e36e83d96815e3fcf66ffdd13d/CARZero/models/CARZero_model_dqn_wo_self_atten.py)：同一个 fusion module 复用于两个方向，第一遍局部 image + 全局 text，第二遍局部 text + 全局 image。官方 `i2t_cls/t2i_cls` 名称与论文按 query 命名的方向容易相反，张量流才是依据；两个分数矩阵分别做 loss。
- [dqn_cos_loss.py / DQNCOSLoss](https://github.com/laihaoran/CARZero/blob/fa4e09cdfe7801e36e83d96815e3fcf66ffdd13d/CARZero/loss/dqn_cos_loss.py)：实际为 `0.5*(CE(S,diag)+CE(S.T,diag))`；文件开头的 BCE 注释与可执行实现不符。原样 diagonal 单正样本假设不适合重复候选患者标签，适配后需标明多正/未知掩码的改变。
- [LICENSE](https://github.com/laihaoran/CARZero/blob/fa4e09cdfe7801e36e83d96815e3fcf66ffdd13d/LICENSE)：Apache-2.0。
- 官方 TQN 内部 `memory_key_padding_mask=None`，当前适配若使用 padding 的局部文本，应加入真实 padding mask 并记录修改。
- 原仓库 README 提示 batch64→128→256 和 80GB GPU；当前 RTX4080 的小规模训练是不同预算，不能声称按原版预算复现。

以上公式来自[官方 CVPR 论文的 3.2 节](https://openaccess.thecvf.com/content/CVPR2024/papers/Lai_CARZero_Cross-Attention_Alignment_for_Radiology_Zero-Shot_Classification_CVPR_2024_paper.pdf)与真实源码。模型在当前任务中的实际效果必须等待服务器实验。

## 不纳入窗口或不当作主会 SOTA 的条目

- MGCA：NeurIPS2022；LoVT：ECCV2022；GLoRIA：ICCV2021；MedKLIP：ICCV2023。可保留经典对照，但不是本次 2024–2026 新方法窗口。
- [RadZero3D](https://openaccess.thecvf.com/content/ICCV2025W/VLM3D/html/Park_RadZero3D_Bridging_Self-Supervised_Video_Models_and_Medical_Vision-Language_Alignment_for_ICCVW_2025_paper.html)：ICCV2025 VLM3D workshop，不能写成 ICCV 主会。
- [MVCM](https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/html/Zou_MVCM_Enhancing_Multi-View_and_Cross-Modality_Alignment_for_Medical_Visual_Question_CVPRW_2025_paper.html)：CVPR2025 workshop。
- [MCR](https://arxiv.org/abs/2312.15840)：本次仅核到 arXiv，未完成正式顶会/顶刊出处核查，不据聚合站猜会议信息。
- 2026 搜索亦检出 segmentation/VQA 方法；具有“alignment”一词不等于 MRI–候选语义检索任务匹配，不能仅按标题纳入。
