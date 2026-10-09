# GPT-6 项目试用：统一任务

只读审查现有研究源码与证据；不读取本目录其他模型的回答、RESEARCH_WORKFLOW_PASA.md 或迁移总结。用中文作答，主要判断附路径和行号。不能把本地中间稿默认当正式投稿版。

材料：train_semantic_alignment.py、experiment_dataset.py；output/goal-https-openreview-net-pdf-https/outputs/Chapter3_Method_GDI_20260609.tex、Experiments_Results_Findings_20260617.tex、Experiment_Evidence_Source_Notes_20260617.md，以及该 output 项目 work/evidence/sota_comparison_table.csv。

任务：
1. 用合成患者 Grade=4、Type=Glioblastoma、IDH=wildtype、MGMT=unknown、1p19Q CODEL=non-codeleted，重建 enhancing、edema、necrotic 区域的正样本。说明监督粒度与缺失处理。
2. 判断训练/评价的标签定义是否造成泄漏，区分已证实事实、条件风险与尚无依据的主张。
3. 审核 full/no_anchor 比较能否支持模块贡献，提出可比较的对照。
4. 解释代码的 Recall@k、anchor_consistency 和多正样本损失实际衡量什么。
5. 核对 full/no_graph 数据能支持什么，如何确认正式稿版本。
6. 写一段不超过150字的当前可支持核心主张。
7. 新代码任务：在指定独有输出目录实现纯标准库的 masked_retrieval_metrics(ranked_keys, positive_keys, negative_keys, k)；未知候选先从排名排除；返回 hit_at_k、recall_at_k、reciprocal_rank。正负集合重叠、排名重复、k不是正整数应报错；无正标签应返回三个None；缺失于排名的正样本保留在召回分母。提供可直接运行的合成测试，不修改训练源码。
8. 新设计任务：若按患者分组之后，某中心只有高等级患者，能否用现有队列区分中心捷径与分级泛化？提出可执行检查、不可识别边界和需要新增的证据。

输出：指定独有目录中的 audit.md、metrics.py、test_metrics.py。禁止重训练、读取真实患者数据、修改研究源码或配置。不使用额外代理。最后报告已运行验证与剩余限制。
