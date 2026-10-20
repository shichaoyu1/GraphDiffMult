# UTSW/FeTS 标签定义核查及处理边界

日期：2026-10-20。

## 已核实

[FeTS官方数据说明](https://fets-ai.github.io/Challenge/data/)规定1=NCR坏死核心、2=ED瘤周水肿/浸润组织、4=ET增强肿瘤。[官方分割源码](https://github.com/FeTS-AI/Front-End/blob/master/src/applications/FeTS_CLI_Segment.cxx#L425)的融合类别为0/1/2/4。

[UTSW数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13287574/)及[TCIA集合页面](https://www.cancerimagingarchive.net/collection/utsw-glioma/)描述三类区域与人工修订，但核查未找到人工修订NIfTI标签3、5的数值字典。用户确认本地无该字典。因此这两类的组织学含义仍未解决，不能假装完成数值语义确认。

## 新探索性pilot的明确选择

- raw_label_1到raw_label_5分别保留，不合并3与4，不推断5是噪声或某组织类型。
- 背景仅0；集合外值、非整数、非有限值一律报错。
- 这些编号是空间分箱，不作为独立区域病理真值。
- 原始5在3例中仅1–2个体素，但不据此删标签。最近邻resize丢失的区域被记录为缺失；幸存的微小区域在等权池化中可能权重偏大，这是本pilot限制。
- 对所有方法共同冻结无有效分割的病例排除，并保留ID、原因和分割SHA256。根据现有审计预期训练435、验证63，实际以新准备清单为准。
- 后续取得标签字典再创建新输入版本；不能覆盖此次原始数字分箱结果。

这使空间表征和患者级监督的探索可以推进，但不构成完成区域组织学语义核实，也不能挽救旧稿的区域病理对应主张。
