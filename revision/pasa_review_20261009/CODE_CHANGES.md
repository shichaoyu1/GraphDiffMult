# 本轮代码修改与验证

2026-10-09。用户要求：只制定文章修改计划，代码直接修改。没有修改论文正文或历史实验分数。

## 已接入的科学协议

入口仍为 `train_semantic_alignment.py`，新评分模块为 `semantic_evaluation.py`，结果协议为 `pasa_patient_metadata_v2`。

- 训练默认 `all_patient_anchors`。已知患者属性是监督来源；`region_rules` 保留为辅助设置，取消全缺失后的字段兜底。
- 已知互斥分类字段的其他取值作为负样本；缺失/未选字段被屏蔽。年龄不推断为互斥分类负例。这里仍依赖数据字典核查原始字段是否单值、取值拼写是否规范；代码不会识别所有矛盾临床记录。
- 正、负、未知掩码进入对比损失，空正样本行不再进入 log-sum-exp 均值。v2 MedCLIP 路径不沿用会屏蔽真实同字段负例的旧 heuristic，因此该路径的检索损失与 masked CLIP 一致，不能当独立已发表系统比较。
- 评价默认 `--evaluation_fields molecular`，按字段分别检索并固定于所有对照。`no_anchor` 表示去病理辅助监督；病理随机原型不进入主评价。`--evaluation_fields all` 只用于都有相应监督的另一个明确任务。
- 真正 `recall@k` 与 `hit@k` 分开；MRR 和 AP 对无法检索的已知真值计未命中；词表外真值保留在分母，无真值的字段报告缺失并不评分。AUC 对 ties 平分，只有可确认正负时可算。
- 区域先平均，再在患者内平均字段，最后患者等权。相似度间隔改名 `positive_negative_similarity_gap`；不再将它称为稳定性。
- 测试主分数使用完整测试集；旧 `align_max_cases` 不再截断主指标。患者级文件保存字段、已知候选、完整真值数、未见键与监督来源。
- 空划分、重复患者ID、跨集合重叠立即报错。`--splits_file` 可加载冻结划分；`--sample_seed`、`--split_seed` 与训练 `--seed` 分开。患者别名去重仍需要真实数据映射核验。
- 保存 `protocol.json`、核心源码 hash、词表 hash、规范化标签及划分 hash、缺失/OOV覆盖；MRI 原文件未逐个 hash。分割派生代理标签单列来源，不能当独立病理真值。
- 禁止覆盖已有运行目录。新默认输出为 `output/semantic_alignment_experiment_v2`。

## 对照入口的含义

| 设置 | 实际干预 | 使用边界 |
|---|---|---|
| full / no_anchor | 有/无病理辅助监督；共同分子评价 | 新任务，不能沿用旧0.131差值 |
| full / no_anchor_loss | 标签相同，只关闭中心约束 | 测损失项，区别于病理监督贡献 |
| clip / global_clip | 相同编码器，区域检索 vs 区域特征均值全局检索 | global是区域池化，不是整幅MRI编码基线 |
| multilabel | 相同原型打分的 masked BCE 内部对照 | 不等于独立多任务临床分类系统；中心项默认仍存在，可用lambda_anchor=0关闭 |
| --alignment_objective single_positive | 检索损失只抽一个正例，其他已知正例屏蔽 | 中心约束仍可使用完整正集，故只称检索目标对照；可关闭中心项另做实验 |
| full / no_graph；full / graph_only | 图或扩散开关 | 必须相同任务/划分/预算实际重跑才有结论 |

`paper_config` 不再把指定消融静默重置为 full。所有基线是可运行入口，没有新科研成绩；真实临床预测、区域专家验证和已发表系统直接比较仍在文章修订计划中。

## 统计工具

`utils/bootstrap_semantic_5seed.py` 根据记录版本选择旧/新指标，禁止混合两协议；v2禁止混合不同病例、划分、词表、评价字段及源码版本。新协议对固定已训练 seeds 使用同一患者抽样，报告患者抽样不确定性。重复抽中的患者使用独立抽样实例ID，保留其重采样权重。该工具仍主要输出各方案的区间，匹配方案的差值检验需按冻结实验设计实施，不能从两区间的重叠判断显著性。

## 验证与剩余关口

测试入口：`python -m unittest discover -s tests -p test_semantic_review_revision.py -v`。包含掩码/指标/未见标签、梯度、空目标、分组聚合、划分、保存后重评分及一轮合成模型流程。Windows沙箱读取Matplotlib配置被拒，获准后在沙箱外运行。合成模型只验证流程，不证明真实MRI网络或临床效果。

最终结果：14项测试通过（退出码0），包含full、no_anchor、no_anchor_loss、global_clip、multilabel、medclip_style、clip、no_graph八种一轮合成流程；所有保存结果按v2记录重评分一致。编译检查及`git diff --check`通过。single_positive的实现已接入，但没有单独完成真实模型训练或性能验收。

正式运行前必须核验原始字段字典、患者唯一标识、中心×等级覆盖和冻结协议；新协议至少重跑基础对照后再改主张。构建正样本工时、细粒度区域参考、外部重复数字的独立性尚无证据，不能靠代码修补宣称已解决。
