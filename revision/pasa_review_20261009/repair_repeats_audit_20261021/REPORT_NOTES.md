# 报告来源与交付说明

- 报告任务：判断新增 seed 43、44 是否重复了 seed 42 的监督/输入比较，并据此限定论文主张和下一步实验。
- 主读者为实验研究者；使用 build-report 的 technical 结构，因为主要价值是指标、比较设计、模型解释和证据审计。
- 交付选用本地可携带 HTML；不向 Sites 发布，报告只嵌入聚合结果，不含患者标识、原始 MRI、元数据或本机绝对路径。
- 唯一数值来源为同目录 `summary.json`，由父任务的 `audit_repair_repeats_20261021.py` 从返回分数重算。训练源码语义来源为 `pasa_repair_pilot.py`，冻结基线 86a8e6f。
- 当前共享校验器强制图表来源包含实际 SQL，不能直接接受标记为 Python 的文件读取代码。因此构建脚本使用内存 SQLite JSON1 实际读取本地摘要 JSON，canonical 来源保存这条真实查询、参数对应文件和后续 Python 转换脚本；这不新增数据、不改变科学数值、不把重新排版当作新科学来源。
- 技术结构映射：标题 → 结论摘要 → 比较范围/分母（提前以避免图表误读）→ 宏 AUROC/配对差值/分字段证据 → 记录复核与模型定义 → 限制 → 下一步建议 → 待回答问题。范围与定义提前是有意的窄调整。
- 数值仅为同一开发验证集上的已选成绩；训练 SD 使用 ddof=1。没有患者泛化区间，没有显著性检验，没有独立测试确认。

## 图表契约

- 分析问题：监督目标改善与分区输入增益是否跨训练 seed 重复？
- 图族与形式：比较，原生 grouped bar；四方案 × 三 seed，12 个真实训练结果。
- 种子不是时间，故不使用折线趋势；不将三个 seed 视为三个独立患者队列。
- 轴：方案类别 / 分子宏 AUROC；同尺度，默认标准柱图从零起。0.5 作为常数分数参考。
- 颜色：使用共享读取器 categorical palette 统一的主题 token；柱在组内的固定顺序与可见图例同时标识 seed，不依赖颜色单独识别。运行时禁止 legacy `series` 颜色映射字段，因此不额外覆盖共享调色板。
- 数据保留各 seed AUROC、平均值、SD、轮次、分字段均值、共同人数、参数与优化步，可追溯并供读取器查看明细。
- 报告中其它数值使用精确表格，便于核查配对方向、选择轮次和少数类召回。
- 容器：全宽原生图，报告按单列阅读；唯一 HTML 由插件 canonical reader 打包器生成，无独立 CSS/JavaScript 图实现。

## 构建与验证

1. 运行本目录 `build_report.py` 生成 canonical `artifact.json`。
2. 从 Data Analytics 插件根目录运行 `npm run report:deliver -- --input <artifact.json> --output <report.html>`。
3. 将打包器结果保存为 `delivery_receipt.json`。`passed` 代表完整共享读取器浏览器验证通过；`structural_only` 则仅代表结构、payload 一致及语义 fallback 验证通过，必须向用户说明缺少浏览器检查。

不在本报告中宣称已完成后续真实实验。分割统计探针已实现并经合成测试/代码复审，但尚无真实拟合结果。

本次打包结果：canonical validation、package 与结构验证通过（`structural_only`）；未找到已安装 Chromium headless-shell，因此没有执行增强读取器的桌面/窄屏/来源弹窗浏览器核验，也没有静态图 SVG 提取。语义 fallback 保留全部数据表与文字，未安装浏览器。该限制仅涉及报告展示 QA，不改变原始实验重算结果。

## 下一步状态的窄修订

- 仅修改 `next` 段落的实现/运行状态；图表、表格、原数据、来源、其它段落和读序保持一致。重新生成时只另有正常生成时间变化。
- 删除 `next` 段及两个生成时间字段后，更新前后 artifact 的规范化 SHA256 均为 `a4af1ebd42df24853dc77e891a2994138797483e7207347f50347eca1cc75e1b`，确认其它内容未变。更新后再次打包通过，仍为 `structural_only`。
- 父任务已实现 `tools/run_pasa_shape_probe.py` 和 CPU 一键入口 `tools/launch_pasa_shape_probe.sh`；7 项合成测试、独立代码复审通过。协议见 `SHAPE_PROBE_PROTOCOL_20261021.md`；真实实验尚未运行，本地没有原始 metadata TSV。
- 固定 11 特征为 resize 后前景体素总量的 log1p、五个原始标签的前景比例、五个 presence；训练集内标准化和三个分子字段的 L2 逻辑回归。入口核验元数据及参考/源码/词表身份。这里没有报告或暗示真实 AUROC。
- 实现准备时曾拟从旧 `train_prior` 文件恢复训练标签；独立审查发现其记录实际对应旧 test 124 人。在任何真实拟合前已放弃该来源并删除读取逻辑，改为原始元数据 TSV；没有据该误判产生本报告中的任何实验成绩。
- 探针为患者级分割统计的替代解释检查。它不构成同 CNN 五任务、同参数/优化条件的受控消融；不能把分差量化为 MRI 强度贡献。
