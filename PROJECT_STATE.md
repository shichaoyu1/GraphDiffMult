# PASA 项目交接

## 2026-10-20 修复实现与新探索协议

已准备补seed43/44的独立调度器：`tools/repeat_pasa_repair_pilot.py`，后台入口 `bash tools/launch_pasa_repair_repeats.sh`。复用已完成seed42 pilot的protocol/inputs/vocab及11项科学源码哈希，仅改变训练seed，共8个新10轮运行，原训练runner保持不变，不重复smoke。支持校验完成seed后跳过、失败新attempt重试，不从last伪装续训。尚未在服务器启动这8个运行。

最新返回 `output/20261019_AAAI_review/pasa_repair_pilot_20261020_095259_432520`：四组各10轮seed42完成，435train/63val，保留病例全有有效区域；40份逐轮指标重算通过，11源码hash与86a8e6f一致，12检查点选模正确、验证重放一致。整块mass/field宏验证AUROC=.7507/.7937，分区mass/field=.7173/.7734。单seed支持检查逐字段目标，不支持分区优势；四组1p19Q首位仍全为多数类。见 `revision/pasa_review_20261009/REPAIR_PILOT_RESULTS_20261020.md`。下一步优先同协议补seed43/44，暂不加图扩散或延长训练；尚无独立test/外部确认。

已核查FeTS标准1=NCR、2=ED、4=ET；UTSW人工分割3/5未找到官方数值字典，用户确认无额外说明。新pilot明确将原始1/2/3/4/5作为独立匿名空间分箱，不合并或命名组织类别。核查记录在 `revision/pasa_review_20261009/LABEL_DEFINITIONS_20261020.md`。

新实现 `pasa_region_inputs.py` + `pasa_repair_pilot.py`：最大病灶面积切片采样、resize后presence、空区域不进CNN/BN/损失汇总，可选图屏蔽缺失行列；全分割空者所有方案共同冻结排除。旧训练入口保持原证据定义。

新入口 `tools/run_pasa_repair_pilot.py`、后台入口 `tools/launch_pasa_repair.sh`，说明见 `PASA_REPAIR_PILOT_RUN.md`。协议pasa_repair_pilot_v1：全局ROI/原始分区汇总×集合/逐字段监督4组，同CNN参数、患者、词表、单向患者级监督，无图扩散中心损失，固定且隔离验证随机状态，AUROC选模并保存best-map和last。smoke四组各1轮、最多train/val各32人；pilot四组各10轮、seed42，要求已完成匹配smoke。均不评价test，不下载HF。

本地58项测试通过，包含真实临时NIfTI四组单轮CPU流程；这些是合成验证，不是UTSW训练结果。尚未在AutoDL执行新修复实验。后续按smoke→核对coverage/验证重放→pilot推进，真实成绩不可预断。

## 2026-10-19 最新阶段状态（以下历史记录保留）

最新分割审计返回：`output/20261019_AAAI_review/pasa_region_audit_20261019_173123`，501例记录与前次一致。34例全空输入分解为：25例采样仅有未处理label 3，6例有全体积病灶但中间切片无阳性，3例当前分割全体积为空。95例含label 3，其中70例同时含3和4，不能直接3→4；另3例含1–2个label 5体素。形状/affine匹配均通过。见 `revision/pasa_review_20261009/REGION_AUDIT_FINDINGS_20261019.md`。下一步输入规则修订前需核实manual/FeTS标签字典；尚未改训练代码或新训练。

最新返回：`output/20261019_AAAI_review/pasa_diagnostics_20261019_165637_406586` 已完整跑完。12份验证评分已重算，3个checkpoint哈希一致；随机推理宏AUROC极差约0.0110–0.0224，固定seed精确重放。关键新增问题：训练31/438、验证3/63病例三个区域输入全空。真实梯度探测仅同一组6人，不可当18个独立患者。决策见 `revision/pasa_review_20261009/RETURNED_DIAGNOSTICS_20261019.md`；训练前先用 `tools/audit_pasa_region_inputs.py` 核查完整分割标签与采样切片，暂不臆测编码映射。

用户返回两轮服务器实验，45 个正式运行已完成本地审计。第一阶段证据快照在 `revision/pasa_review_20261009/stage1_snapshot_20261019/`，包含文本证据归档及 802 个原文件哈希；大文件留在原位置，非异机备份。原始结果位于 `output/20261019_AAAI_review/sota_baline_output/`。

第 2、3 步诊断已完成本地可执行部分，见 `revision/pasa_review_20261009/STEPS23_DIAGNOSIS_20261019.md`。关键发现：PASA 首位预测等于频率基线，但后期验证 AUROC 改善而 mAP 选模停留第 1–2 轮；第二轮完整模型选中的图构建器仍在冻结期；第一轮区域向量高度趋同；合成梯度验证集合正样本损失可被容易正标签主导；eval 扩散路径存在随机采样。不能把这些发现表述为已证明修复会提升测试性能。

没有新患者训练、没有修改训练源码或原实验。下一步需先冻结新评价/选模协议，再在服务器做验证重复推理、空区域统计和少量真实 batch 梯度诊断。现有测试集已用于开发决策；后期 checkpoint 本批未保留，不能从 history 恢复测试成绩。

AutoDL 诊断入口已实现：`tools/diagnose_pasa_server.py`；后台一键入口 `bash tools/launch_pasa_diagnostics.sh`，用法见 `PASA_DIAGNOSTICS_RUN.md`。读取旧 campaign 的三个 full 检查点，扫描训练/验证分割，重复验证评分，并对少量训练批次做梯度探测。验证历史科学函数、模型源码与数据身份；独立新目录保存结果，未启动实际远程运行。

更新：2026-10-09。当前源码基线、证据文件 SHA256 见 `migration/gpt6_20261009/baseline.json`。Git 分支 main，提交 f94cf1ec1041c9f15f78c84fb42bb384103d003f。迁移开始时研究流程、模板、迁移计划和 tools/ 为未跟踪文件；保留原有内容。

## 接手顺序

1. 阅读 `AGENTS.md` 与 `RESEARCH_WORKFLOW_PASA.md`。
2. 查看 `PASA_EVALUATION_PROTOCOL.md`，区分现状与建议；新研究使用 `RESEARCH_PROTOCOL_TEMPLATE.md`。
3. 查 `migration/gpt6_20261009/RESULTS.md` 了解本次模型试用和配置状态。

## 当前研究与证据范围

项目用患者元数据构造语义锚点，训练 MRI 区域表征进行检索；分级是下游任务。主入口 `train_semantic_alignment.py`，下游入口 `train_utsw.py`。

迁移时核实的旧实现（2026-10-09 本轮评审修订已更新，见下文）：

- 锚点字段为 Grade、Tumor Type、IDH、MGMT、1p19Q；临床字段可选。锚点向量是可学习参数，不能默认称为文本编码器输出。
- `region_rules` 按区域名称选择患者字段。它提供患者级元数据派生监督，尚不等于区域组织学真值。
- 训练与评价共享标签定义本身正常；实际泄漏需查患者重叠、测试数据使用、预处理和选模记录。
- `recall@k` 当前是任一正样本 Hit@k；`anchor_consistency` 是正负相似度均值差。
- `no_anchor` 排除病理字段，改变训练与评价目标及候选库。原始分差不能直接解释为同任务模块贡献。
- 训练集建词表、验证集 mAP 选 checkpoint 是已存在的正确做法。空验证/测试集回退有重叠风险，是否发生需核对实际 splits。
- seed 同时影响病例采样、数据划分和训练；历史开发轮次不能默认视为同一冻结协议的独立重复。

## 稿件与实验版本

本地材料位于 `output/goal-https-openreview-net-pdf-https/`：

- 方法：`outputs/Chapter3_Method_GDI_20260609.tex`。
- 结果：`outputs/Experiments_Results_Findings_20260617.tex`。
- 溯源：`outputs/Experiment_Evidence_Source_Notes_20260617.md`。
- 表格：`work/evidence/sota_comparison_table.csv`。

这些是中间材料，尚未确认等于正式投稿快照。CSV 的 stabilized 3 seeds 中 no_graph Hit@1=.7800，full=.7222；正式稿中的映射必须依赖投稿 PDF、生成表脚本和运行配置确认，不能事后互换名称。BraTS 代理标签结果也不能替代真实病理/分子外部验证。

## 下一轮研究的关口

代码修订现状：`pasa_patient_metadata_v2` 已接入缺失掩码、正确指标、共同分子评价、严格划分、分离种子、对照入口和版本输出。14项合成测试通过，包含8种配置的一轮假模型流程及保存后重评分；不是8个真实实验。文章保持原样；修改计划与代码说明在 `revision/pasa_review_20261009/`。下一步为真实数据审计、冻结实验及重跑。

1. 找到正式投稿 PDF、提交日期和对应源码/运行目录；未找到前保留版本不确定性。
2. 冻结新评价协议、词表、病例划分、已知/未知定义及比较预算。
3. 实现并验证评价协议后再重跑；本轮迁移没有改变训练语义，也没有重训练。
4. 用简单基线和反例确定方法的必要性，再据真实结果确定图模块及区域主张的范围。

## 模型与使用方式

近年方法对比（2026-10-09）：文献核查在 `research/sota_alignment_20261009/`，选择CARZero(CVPR2024)/RadZero(NeurIPS2025)真实官方对齐模块的MRI元数据适配。6方案×3seeds共18个正式运行，另有训练频率控制；运行说明为 SOTA_ALIGNMENT_RUN.md，入口 tools/run_recent_alignment.py。新协议 pasa_recent_alignment_v1 先汇总患者分数再字段排名，与旧区域评分不同，必须全部重跑。用户在AutoDL启动，尚无正式数值；合成缓存不充当预训练结果。

服务器准备：AutoDL路径与RTX4080配置已写入 SERVER_RUN.md；新调度器支持冻结协议、分阶段运行和保留失败attempt。19项本地测试通过，上传包为 artifacts/PASA_SERVER_V2.zip。尚未连接服务器、读取服务器真实队列或启动远程实验。

迁移前用户默认配置已为 GPT-6.1 Sol / low；这不证明当前聊天正在用相同设置。日常候选保持该组合，关键研究审计采用 Astra 复核。试用记录负责说明实测结果与限制，模型一致意见不作为科学真值。模型自动路由不是 AGENTS.md 能保证的功能。
