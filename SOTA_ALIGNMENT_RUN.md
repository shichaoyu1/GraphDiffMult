# 近年细粒度对齐：PASA 任务适配实验

日期：2026-10-09。检索按正式发表年2024–2026筛选，候选与出处见 `research/sota_alignment_20261009/CONFERENCE_SEARCH.md` 和 `JOURNAL_SEARCH.md`。本轮代码已实现，正式数值由 AutoDL 运行产生。

## 1. 选择依据与实施边界

优先接入 [CARZero（CVPR2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Lai_CARZero_Cross-Attention_Alignment_for_Radiology_Zero-Shot_Classification_CVPR_2024_paper.html) 与 [RadZero（NeurIPS2025主会）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/510e5c2fd5ad326f75594563f4ad5e0d-Abstract-Conference.html)。两者都有可核查的局部视觉—语言对齐机制与官方代码，本轮直接保留相应官方模块。AFLoc、MaCo、ALTA、Med-ST、FG-CLIP/2等也已列入候选；报告结构、重建、时间配对或预训练条件不完整时，不把简化模块当整篇复现。

**本轮是 MRI/结构化患者元数据的 method-adapted 比较。** 原论文使用胸片、报告、不同预训练编码器和更大训练预算；本数据没有完整报告或区域组织学真值。它可以回答“这些对齐机制在相同患者属性任务上是否优于PASA/余弦对照”，不能称已复现原胸片榜单、零样本性能或达到SOTA，也不能验证区域病理定位。

## 2. 统一实验矩阵

固定一次患者划分，默认seeds42/43/44、30轮、batch2、ROI96、7切片、AdamW(lr3e-4/weight_decay1e-4)、cosine scheduler、梯度裁剪1；用验证集患者字段宏平均mAP选checkpoint，不用测试集选模型。6方案×3种子=18个正式运行；不同头的参数量和时间另报，不能称参数量完全一致。

| 运行名 | 实际实现 |
|---|---|
| pasa_full | PASA图/私有/扩散+可学习原型，多正集合检索与中心约束及原辅助损失 |
| pasa_minimal | 相同ROI编码器+原型检索，无图/私有/扩散及中心约束 |
| pasa_text_control | PASA全框架，使用与其他文本方法相同的冻结文本表及训练投影，控制文本预训练影响 |
| text_cosine | 相同MRI编码器+相同冻结文本表，患者均值表征与文本余弦对比 |
| carzero_metadata | 官方四层无query-self-attention的双向交叉注意力、FFN、线性相似度头；同模块作用于视觉局部/文本全局与文本词局部/视觉全局 |
| radzero_metadata | 官方VL-CABS(cos)、共用可学习attention/loss温度、两层官方DINOv2训练块、逐正例MP-NCE |

另输出只用训练标签频率、Laplace+1平滑的 `train_frequency` 控制。它不读取测试患者属性作为预测输入、不额外训练网络，给出类别不平衡下的基准；表中注明确定性控制，不伪装为三个独立训练种子。

全局视觉项由三个ROI均值构成；局部项为三个ROI，不是原论文的高分辨率patch网格。CARZero共享宽度由768适配为128，保留四头、FFN1024、4层与原linear scoring；按词mask裁剪padding再用官方decoder，避免padding参与对齐。RadZero视觉层宽度/heads适配为128/4，保留两个DINOv2块与cos scoring。固定官方源码提交及许可证在 `third_party/ALIGNMENT_SOURCES.json`，保留Apache2.0/CC-BY-NC4.0及HuggingFace Apache2.0归属。

文本侧统一为冻结 `sentence-transformers/all-mpnet-base-v2`，仅编码训练词表中field–value固定候选句，缓存token序列和attention-mask均值，不生成报告、病灶位置或额外标签。所有文本方法共用同一缓存、可学习128维投影；实际模型revision和缓存SHA256保存。与原版可训练报告编码器有区别，必须披露。

训练将已知患者病理/分子属性作为多正例，字段内明确互斥取值作为负例；缺失字段未知掩码。CARZero双向对角InfoNCE扩展为类别共享标签的双向multi-positive mass NCE；raw MLP logits不额外除.07。RadZero用稳定log-sum-exp计算每个正例对“该正例+已知负例”的分母，全部正例配对平均，并扩展共享标签的双向关系；不是原版患者报告一对一group_map。对标签缺失、同类患者和双向损失的这些适配均明确记录。

## 3. 评价与统计

主要终点为 **patient molecular macro mAP**。先形成患者×候选的实际模型分数，再分IDH/MGMT/1p19Q字段排名，字段内等权，患者等权。PASA本轮先平均区域相似度再排名；这与旧版“区域先排名再平均”不同，所有18个运行统一重跑，禁止引用旧分数填本表。候选来自训练词表，未知字段不评分且记录缺失；词表外真值留分母、计未命中。

MRR/Hit@1/Recall@1/字段内PairAUC作为次要指标。每字段常为单正例、小候选，Hit@5/10可能饱和，mAP与MRR在这些字段上相同；不靠这些高分推断区域真实性。cross-attention raw logits不称余弦相似度或anchor consistency。

另报告真正跨患者的biomarker_macro_auc：字段内候选softmax后，按每个分子类别在已知、词表内病例上做one-vs-rest AUROC，再类别/字段宏平均；记录类别可计算性与病例覆盖。单类别测试集AUROC不可估计。它与候选边排序PairAUC不同，词表外病例不进入这项条件AUROC，但仍在主要mAP中计未命中。只作为该队列的预测分析，不能自动称临床效用验证。

汇总对相同患者、相同训练seed作mAP差值；患者成对bootstrap保留全部已训练seed，主要参照pasa_full，并补充pasa_text_control，输出逐seed差值和95%区间。区间仅覆盖固定训练模型的患者抽样，不覆盖完整训练程序不确定性；多方案探索比较不作未经校正的确定显著性结论。没有独立外部队列或临床效用证据。

## 4. AutoDL 一键启动

默认MRI和TSV路径已使用你提供的 `/root/autodl-tmp/dataset/UTSW-Glioma` 和 `/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv`。新建独立campaign，保留上一轮。

更新代码后，进入项目根目录并安装文本缓存依赖：

也可以使用 `artifacts/PASA_ALIGNMENT_SOTA.zip` 在新代码目录解压启动；包中包含官方模块、固定版本/许可证、研究记录和测试。当前未产生正式性能数值。

```bash
python -m pip install -r requirements_alignment.txt
```

第一次需要从HuggingFace下载MPNet到服务器缓存；可以用 `--text_model /已下载的本地模型目录` 离线运行。预训练文本只下载一次，先在CPU生成小的固定候选文本缓存，GPU训练阶段不运行完整文本编码器。普通运行还需要已有的CUDA PyTorch、numpy、nibabel、matplotlib。

先检查六个方案各1轮（使用完整冻结划分，不加--epochs1）：

```bash
python -u -B tools/run_recent_alignment.py --stage smoke --output_root /root/autodl-tmp/pasa_runs/sota_alignment_20261009
```

正式一轮18个运行，一句启动：

```bash
nohup python -u -B tools/run_recent_alignment.py --stage sota --output_root /root/autodl-tmp/pasa_runs/sota_alignment_20261009 > /root/autodl-tmp/sota_alignment_20261009.log 2>&1 &
```

```bash
tail -f /root/autodl-tmp/sota_alignment_20261009.log
```

如果上一轮的 `/root/autodl-tmp/pasa_runs/pasa_v2_20261009/splits.json` 存在，自动复用并严格检查病例集合；可用 `--reference_splits 路径` 指定。否则首次按split_seed42冻结。重复同一命令跳过已完成任务，失败保留旧attempt后重跑；不是续训到中断epoch。改变源码、metadata、文本模型、输入条件或预算需新建campaign。

原Python/PyPI网络失败或模型下载失败时，停止在准备阶段，不降级为随机文本向量。本地验证使用明确的合成文本缓存，不把它当真实预训练结果。若服务器暂时不能访问HuggingFace，应先下载模型到本地目录再指定路径。

## 5. 交付结果

`summary.csv`逐运行结果；`sota_mean.csv`跨seed均值/SD；`paired_comparisons.json`主要终点成对差；每个成功运行有config/参数量、source与数据定义、text cache hash、history、best.pt、patient_score_records.json和test_metrics.json。正式阶段结束后自动汇总；缺失运行或协议不一致会停止汇总。

正式训练由用户在AutoDL执行。本地只验证实现、张量梯度、损失公式、保存/重评分和合成流程。取得完整结果后再判断PASA对不同机制是否有优势，并按结果修订主张，不预设PASA必须胜出。
