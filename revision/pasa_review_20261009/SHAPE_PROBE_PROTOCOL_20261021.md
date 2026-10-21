# 分割统计探针：2026-10-21 冻结协议

协议版本：`pasa_shape_probe_v1`。本文件在真实探针拟合和查看其验证指标之前写定。它是观察既有 CNN 结果后增加的探索性检查，不是预注册确认实验，也不是新的独立验证队列。

## 问题与解释边界

检查仅凭本轮 CNN 输入已经包含的分割统计，能否预测患者分子标签。整块和分区 CNN 都使用分割掩码，频率基线不能排除形态组成的解释。

本探针是三个独立分子分类任务；CNN 训练包含分子、分级、肿瘤类型五个任务。二者架构、优化与监督结构不同，因此任何胜败均不能直接量化 MRI 强度的增量，也不能作为同架构受控消融。分割数字 3/5 仍无已核实的组织学定义。本探针不提供区域病理对应或因果证据。

原始数字标签的组成和 presence 也可能代理分割来源、标号习惯或病例构成。未显式输入来源字段不等于已排除这些关联；此类相关性本身不能当作组织学机制或数据泄漏的证据。

## 固定数据和标签来源

- 影像输入清单：已完成 seed42 修复 pilot 的 `inputs.json`，配套 `protocol.json` 和 `anchor_vocab.json`。
- 原始 438 名训练患者共同排除既有三例全空分割，保持本轮 435 train / 63 val，不重新划分、不挑子群。
- 只取 `audit.slab_voxel_counts_after_resize`，对应与 CNN 相同的 7×96×96 分割采样；不读取 MRI，也不重新采样。
- 配套本次已完成 repeats 的 `repeat_config.json`，逐项验证其保存的 reference 五个文件哈希：protocol、inputs、vocab、complete、summary。防止同名 reference 后来被改写。
- **训练标签唯一来源是原始 metadata TSV**，由必填 `--metadata` 指定，SHA256 必须等于冻结 pilot 的 `metadata_sha256`。不使用历史评分文件恢复训练标签。
- 只调用既有 `dataset.load_utsw_metadata` 读取 TSV，按固定 train435 ID 取行，复用 `train_semantic_alignment.semantic_anchors` 的 strip、缺失值处理、字段规范化和值小写规则。不会调用 `discover_semantic_cases`、扫描或加载 MRI。
- 按 train435 的原始元数据重建完整词表，与冻结 pilot 的26个 anchors逐项一致；标签构造源码 `dataset.py` 和 `train_semantic_alignment.py` 也须与该 pilot 的冻结哈希一致。不使用会额外接受 mutant/wt 等同义词的另一分类标签解析器。
- 验证标签从新 pilot `whole_roi_field/validation_best_auc.json` 的 targets 恢复。其 CNN 分数不进入探针。训练 scaler 和模型拟合、保存完成后才解码验证 targets，并与原始 TSV 的同63人的 canonical keys 核对身份。
- 原始 TSV 包含全队列，整体哈希和读取会接触整份源文件；本探针只为冻结 train435、val63 提取标签，不读取历史 test 产物、不构造或评分 test。训练拟合只使用 train435。旧 test 已参与过开发，不能称为未触及的确认集。
- 标签来源、词表或 canonical 类别无法核实时终止；已观察但不在冻结候选中的分子值报错，不将其或缺失当阴性，不猜标签。

### 开发准备阶段发现与处理

最初拟从名为 `train_prior_score_records.json` 的历史文件恢复训练标签。运行前独立审查核对实际患者身份与调用代码，发现其中 `train` 指频率先验来自训练数据，评分对象实际为旧 test124，并非 train438。因此该恢复路线被放弃、相关读取逻辑被移除。没有基于这些记录拟合或评分任何真实探针。

本地目前没有找到可核验的原始 metadata TSV，故本地只执行合成测试；真实 CPU 探针需在含原始 TSV 的服务器运行。该检查说明文件名不是划分身份证据，应在使用标签前核对来源、患者集合和实际信息流。

## 预先固定的 11 个特征

设 `n1…n5` 是 resize 后五种原始数字标签的体素计数，`N=sum(n1…n5)`：

1. `log1p(N)`；
2. 五个前景组成比例 `nj/N`；
3. 五个 presence 指示 `1[nj>0]`。

前景比例的分母明确为同一采样中五类前景总数，不是原始三维体积。要求 `0<N<=7×96×96`，计数为有限非负整数，presence 与计数一致。

不加入 MRI 强度、患者 ID、路径、mtime、分割来源、原始三维体积、其他元数据或额外形态特征。患者 ID 仅用于对齐标签和病例记录，不是模型特征。

## 拟合与评分

- `StandardScaler` 仅在435名训练患者的11维特征上拟合；验证只 transform。
- 每个分子字段独立拟合一个 L2 LogisticRegression：`C=1, solver=lbfgs, max_iter=2000, tol=1e-8, class_weight=None, fit_intercept=True`。
- 对该字段已知标签的训练患者才参与分类器拟合；缺失字段不当负类。每个训练字段必须有两个类别，否则终止。
- 使用 sklearn 默认二分类决策：margin>0 为阳性，精确 margin=0 时为阴性。不拟合阈值、不做校准。
- 不调参、不选择训练 seed、不做特征筛选、不根据验证结果重跑多个版本。不选择 epoch。
- 主汇总：IDH、MGMT、1p19Q 各自 AUROC，以及三字段等权宏 AUROC；完整列出覆盖、类别计数、缺失数、阳性和阴性召回、balanced accuracy。三个任务均报告，不挑有利字段。
- 三字段阳性分别为 IDH mutated、MGMT methylated、1p19Q co-deleted。
- 不做显著性检验；这是重复使用同63名开发验证患者的描述性探针。

## 保存与复核

运行时先在新输出目录写入冻结 `protocol.json`，记录上述设定、依赖版本、源码和来源产物 SHA256。先保存 `train_fit.json` 的 scaler、系数、训练覆盖与拟合患者清单，再恢复验证 targets 和评分。

默认原始输出 `output/pasa_shape_probe_20261021/`，已有目录禁止覆盖；启动脚本可指定带日期的新目录。逐患者标签和分数仅保存在被 Git 忽略的 output 下，默认 summary 也只写到同目录。`--summary-out` 可选地另存聚合汇总，汇总仅含统计和哈希，无患者清单；服务器默认不改 tracked revision 文件。

源码：`tools/run_pasa_shape_probe.py`。合成验证检查特征边界、不读取路径或强度、canonical 类别恢复、缺失掩码、train-only 标准化和默认边界，不把这些检查当作真实实验结果。
