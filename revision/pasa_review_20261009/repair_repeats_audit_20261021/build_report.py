"""Package the audited aggregate data into the canonical portable report input."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
SUMMARY_TEXT = (OUT / 'summary.json').read_text(encoding='utf-8')
SOURCE_SQL = "SELECT json_extract(:summary_json, '$') AS audited_summary;"
with sqlite3.connect(':memory:') as connection:
    DATA = json.loads(connection.execute(SOURCE_SQL, {'summary_json': SUMMARY_TEXT}).fetchone()[0])
NAMES = {
    'whole_roi_mass': '整块 ROI＋集合监督',
    'whole_roi_field': '整块 ROI＋逐字段监督',
    'region_pool_mass': '分区汇总＋集合监督',
    'region_pool_field': '分区汇总＋逐字段监督',
}
COMPARISONS = {
    'field_minus_mass_whole_roi': '整块：逐字段 − 集合',
    'field_minus_mass_region_pool': '分区：逐字段 − 集合',
    'region_minus_whole_field': '逐字段：分区 − 整块',
    'region_minus_whole_mass': '集合：分区 − 整块',
}


def f4(value):
    return f'{value:.4f}'


def signed(value):
    return f'{value:+.4f}'


def markdown(identifier, body, source='audit'):
    return {'id': identifier, 'type': 'markdown', 'body': body,
            **({'sourceId': source} if source else {})}


methods = []
for index, (method, label) in enumerate(NAMES.items()):
    stats = DATA['method_summary'][method]
    runs = sorted((r for r in DATA['runs'] if r['method'] == method), key=lambda r: r['seed'])
    methods.append({
        'order': index, 'method': label,
        **{f'seed{r["seed"]}': r['auc'] for r in runs},
        **{f'seed{r["seed"]}_display': f4(r['auc']) for r in runs},
        'mean_sd': f'{f4(stats["auc"]["mean"])} ± {f4(stats["auc"]["training_seed_sd"])}',
        'mean': stats['auc']['mean'], 'sd': stats['auc']['training_seed_sd'],
        'selection_epochs': ' / '.join(str(r['selected_epoch']) for r in runs),
        'train_patients': DATA['split_sizes']['train'],
        'validation_patients': DATA['split_sizes']['val'],
        'epochs_per_run': len(runs[0]['auc_trajectory']),
        'parameter_count': runs[0]['parameters'],
        'optimization_steps': runs[0]['training_steps'],
        'IDH': stats['fields']['IDH']['auc']['mean'],
        'MGMT': stats['fields']['MGMT']['auc']['mean'],
        '1p19Q': stats['fields']['1p19Q CODEL']['auc']['mean'],
    })

pairs = []
for index, (key, label) in enumerate(COMPARISONS.items()):
    comparison = DATA['paired_comparisons'][key]
    pairs.append({'order': index, 'comparison': label,
                  **{f'seed{s}': signed(comparison['by_seed'][str(s)]) for s in DATA['seeds']},
                  'mean': signed(comparison['mean']),
                  'positive': f'{comparison["positive_directions"]} / {len(DATA["seeds"])}'})

field_rows = []
for index, (method, label) in enumerate(NAMES.items()):
    stats = DATA['method_summary'][method]
    field_rows.append({'order': index, 'method': label,
                       **{field: f'{f4(stats["fields"][field]["auc"]["mean"])} ± '
                                 f'{f4(stats["fields"][field]["auc"]["training_seed_sd"])}'
                          for field in ('IDH', 'MGMT', '1p19Q CODEL')}})

codel_rows = []
for r in DATA['runs']:
    field = r['fields']['1p19Q CODEL']
    codel_rows.append({'order': r['seed'] * 10 + list(NAMES).index(r['method']),
                       'seed': r['seed'], 'method': NAMES[r['method']],
                       'auc': f4(field['auc']), 'predicted_positive': field['predicted_a'],
                       'positive_recall': f'{field["true_positive_a"]}/{field["class_a_count"]}',
                       'patients': field['patients']})

all_majority = sum(r['fields']['1p19Q CODEL']['predicted_a'] == 0 for r in DATA['runs'])
field_gain = DATA['paired_comparisons']['field_minus_mass_whole_roi']['mean']
region_gain = DATA['paired_comparisons']['field_minus_mass_region_pool']['mean']
region_field = DATA['paired_comparisons']['region_minus_whole_field']['by_seed']
checks = DATA['checks']
now = datetime.now(timezone.utc).isoformat()
title = 'PASA Repair Repeat Results'
sources = [
    {'id': 'audit', 'label': '三 seed 修复实验聚合审计',
     'path': 'revision/pasa_review_20261009/repair_repeats_audit_20261021/summary.json',
     'query': {'engine': 'SQLite JSON1 + Python', 'language': 'sql',
               'description': 'SQLite JSON1 实际读取已重算的审计摘要；Python 依 build_report.py 提取、命名和格式化聚合行。',
               'sql': SOURCE_SQL,
               'bindings': {'summary_json': 'UTF-8 text of revision/pasa_review_20261009/repair_repeats_audit_20261021/summary.json'},
               'transformation': 'build_report.py；不重新训练或更改原审计数值。',
               'tables_used': ['repair_repeats_audit_20261021/summary.json'],
               'filters': ['训练 seed 42、43、44', '共同训练 435 人、开发验证 63 人',
                           '各模型按验证宏 AUROC 首次严格最大值选模', '本轮无 test 评分'],
               'metric_definitions': {
                   'auc': '每字段用两个候选分数之差计算二分类 AUROC，再等权平均三个分子字段。',
                   'seed_sd': '同一验证集上，三个训练 seed 的样本标准差（ddof=1）；不是患者置信区间。',
                   'paired_difference': '同一训练 seed、同一验证患者和标签的两个已选模型宏 AUROC 之差。',
                   'positive_recall': '默认两个候选中分数更大者为预测类别；共缺失（co-deleted）真阳性数 / 9。',
               }}},
    {'id': 'model', 'label': '修复 pilot 模型与监督定义', 'path': 'pasa_repair_pilot.py',
     'query': {'language': 'python', 'description': 'RepairAlignmentModel、fieldwise_loss；原型归一化余弦头与逐字段交叉熵。',
               'tables_used': ['pasa_repair_pilot.py'],
               'filters': ['冻结科学源码版本 86a8e6f', '原始分区编号 1/2/3/4/5 不赋组织学含义']}},
]

blocks = [
    markdown('title', '# ' + title, None),
    markdown('summary', '## 可以推进监督设计，区域优势仍未成立\n\n'
             f'**已核实：**新增两个 seed 的运行完整，合并后共 {len(DATA["runs"])} 个模型。'
             f'逐字段监督在整块与分区的六组配对中均提高宏 AUROC，平均差值分别为 '
             f'**{signed(field_gain)}** 和 **{signed(region_gain)}**。\n\n'
             '**研究判断：**保留逐字段监督继续开发；目前不宜用这批结果恢复“分区更优”或区域病理对齐的主张。'
             '这些是同一开发验证集上的模型选择结果，尚非独立确认。'),
    markdown('scope', '## 比较对象与分母保持一致\n\n'
             f'共同训练 **{DATA["split_sizes"]["train"]} 人**、开发验证 **{DATA["split_sizes"]["val"]} 人**；'
             '三个 seed 仅改变训练随机性，没有形成三批独立患者。每组 10 轮、2,180 次优化、409,056 个参数，'
             '候选词表和缺失标签掩码一致。分区方案编码更多视图，参数和步数相同不等于计算量相同。\n\n'
             '**主要终点：**IDH、MGMT、1p19Q 三个字段 AUROC 的等权平均；已知标签分别为 62、27、31 人。'
             '各组按验证宏 AUROC 首次达到严格最大值的轮次选模。表中的“±”只表示三个训练 seed 的样本标准差。'),
    markdown('macro_finding', '## 逐字段监督的改善可以跨训练 seed 重复\n\n'
             '下图逐组展示三个 seed 的已选验证 AUROC，便于同时看效果和训练波动；数值越高，患者间排序区分能力越好。'
             '整块逐字段的均值最高，但这不能证明它对新患者更优。分区集合的波动最大，说明仅看一次运行容易误判。'),
    {'id': 'macro_chart', 'type': 'chart', 'chartId': 'macro', 'layout': 'full'},
    {'id': 'macro_table', 'type': 'table', 'tableId': 'macro_detail', 'layout': 'full'},
    markdown('paired_finding', '## 分区相对整块的方向不稳定\n\n'
             f'在逐字段监督下，分区相对整块的差值依次为 '
             f'**{signed(region_field["42"])}、{signed(region_field["43"])}、{signed(region_field["44"])}**；'
             '两个 seed 下降、一个上升。当前证据支持“尚未观察到稳定增益”，不足以判定分区设计在所有条件下无效。'
             '差值均来自各自按同一规则选中的模型，仍受小验证集及选模波动影响。'),
    {'id': 'paired_table', 'type': 'table', 'tableId': 'paired_detail', 'layout': 'full'},
    markdown('field_finding', '## 总体改善不等于每个分子字段全面改善\n\n'
             '逐字段监督对 MGMT 的改善在两种输入和三个 seed 中方向一致；其他字段仍有差异。'
             '分区逐字段的 1p19Q 平均 AUROC 低于整块逐字段，不能事后只挑有利字段来解释分区价值。'
             'MGMT 只有 27 名已知标签患者，字段差异应作为开发观察，不能外推成确定的生物学机制。'),
    {'id': 'fields_table', 'type': 'table', 'tableId': 'field_detail', 'layout': 'full'},
    markdown('threshold_finding', '## 1p19Q 的默认预测仍偏向多数类\n\n'
             f'**{all_majority}/{len(DATA["runs"])} 个已选模型**把 31 名已知患者全部判为 non co-deleted。'
             '仅 seed 43 的两个逐字段模型各预测出 1 名共缺失患者，少数类召回均为 **1/9**。'
             '较高 AUROC 表示分数排序有信息，并不表示默认决策边界或少数类识别已解决。'
             '阈值或校准如需调整，应另定开发协议，不能用旧 test 选阈值。'),
    {'id': 'codel_table', 'type': 'table', 'tableId': 'codel_detail', 'layout': 'full'},
    markdown('integrity', '## 指标和选模记录已交叉核对\n\n'
             f'从原始患者分数重算了 **{checks["rescored_epoch_files"]} 份逐轮记录**和 '
             f'**{checks["rescored_selected_files"]} 份已选模型记录**；检索指标、分字段和宏 AUROC 与保存值一致。'
             '每轮 189 条患者—字段记录中，120 条有已知标签、69 条因缺失跳过；缺失没有当作阴性。'
             '标签与掩码签名一致，原始分数有限，AUROC 另以正负患者逐对比较交叉核算；首次最大值选模通过。'
             '保留的 498 名患者没有全空区域输入。'),
    markdown('model_scope', '## 本轮逐字段模型已经属于多任务分类\n\n'
             '影像表征和可学习原型归一化后计算余弦分数；单正样本字段的损失就是字段内交叉熵，再对患者已知字段取平均。'
             '没有独立文本编码器、语义描述或区域病理真值输入。因此不需要再训练一个数学等价的“普通分类器”来制造对照。'
             '本轮可以支持患者级监督下的分子预测与排序，仍不能证明原型名称赋予了额外语义信息。', 'model'),
    markdown('limits', '## 这批实验仍属于开发证据\n\n'
             '- 三个训练 seed 共用同一小验证集；这里没有做显著性检验，标准差也不是泛化误差区间。\n'
             '- 输入、骨干、损失和选模均较历史版本改变，不能将全部变化归因于单个修复，也不能与旧 test 指标直接作提升比较。\n'
             '- 旧 test 已参与历史开发决策；本轮没有 test 评分，最终确认需另有未参与这些决策的数据。\n'
             '- 人工分割属于模型可用输入。原始编号 3/5 的组织学含义未知，患者元数据也不是区域组织病理参考标准。\n'
             '- 本地审计读取返回产物，没有重新读取服务器全部 MRI 体素；MRI 原始体素未作完整内容哈希。', None),
    markdown('next', '## 下一步先排除分割形态这一替代解释\n\n'
             '**已准备，真实实验尚未运行：**分割统计探针已实现，7 项合成测试和独立代码复审通过；'
             '尚无真实患者探针成绩。本地缺少原始元数据 TSV，下一步在 AutoDL 用 CPU 执行 '
             '`bash tools/launch_pasa_shape_probe.sh`。\n\n'
             '探针固定使用 11 项输入：resize 后前景体素总量、五个原始标签的前景比例和五个 presence 指示，'
             '不读取 MRI 强度；仅在训练集拟合标准化和三个分子字段的 L2 逻辑回归。'
             '入口会核验原始元数据 TSV 哈希、五份参考产物哈希、标签源码和词表身份。'
             '它用于检查患者级分割统计是否已能预测分子状态；与 CNN 的监督任务及模型条件不同，'
             '不能将两者分差直接解释为 MRI 强度的增量，也不能当作同条件消融。\n\n'
             '当前优先保留整块逐字段作为简单主参考，同时保存分区未获得稳定增益的结果。'
             '在替代解释和独立确认路径明确之前，不恢复图、扩散或中心损失。', None),
    markdown('questions', '## 后续证据需要回答的两个问题\n\n'
             '1. 相同标签和输入条件下，MRI 强度相对分割形态是否提供额外可重复的信息？\n'
             '2. 冻结方案在未参与开发的新患者或独立队列上是否仍成立？\n\n'
             '若保留区域病理对应的论文主张，还需要粒度匹配、来源独立的空间病理参考证据；现有患者元数据不能补足这一点。', None),
]

columns = lambda items: [{'field': key, 'label': label, 'type': 'text'} for key, label in items]
tables = [
    {'id': 'macro_detail', 'title': '各方案验证宏 AUROC', 'dataset': 'methods', 'sourceId': 'audit',
     'defaultSort': {'field': 'method', 'direction': 'asc'},
     'columns': columns([('method', '方案'), ('seed42_display', 'seed 42'), ('seed43_display', 'seed 43'),
                         ('seed44_display', 'seed 44'), ('mean_sd', '均值 ± 训练 SD'), ('selection_epochs', '选中轮次 42/43/44')])},
    {'id': 'paired_detail', 'title': '同 seed 配对宏 AUROC 差值', 'dataset': 'pairs', 'sourceId': 'audit',
     'defaultSort': {'field': 'comparison', 'direction': 'asc'},
     'columns': columns([('comparison', '比较方向'), ('seed42', 'seed 42'), ('seed43', 'seed 43'),
                         ('seed44', 'seed 44'), ('mean', '平均差值'), ('positive', '提高的 seed 数')])},
    {'id': 'field_detail', 'title': '各字段验证 AUROC：均值 ± 训练 SD', 'dataset': 'fields', 'sourceId': 'audit',
     'defaultSort': {'field': 'method', 'direction': 'asc'},
     'columns': columns([('method', '方案'), ('IDH', 'IDH（62 人）'), ('MGMT', 'MGMT（27 人）'),
                         ('1p19Q CODEL', '1p19Q（31 人）')])},
    {'id': 'codel_detail', 'title': '1p19Q 默认决策与共缺失召回', 'dataset': 'codel', 'sourceId': 'audit',
     'defaultSort': {'field': 'seed', 'direction': 'asc'},
     'columns': columns([('seed', '训练 seed'), ('method', '方案'), ('auc', 'AUROC'),
                         ('predicted_positive', '预测共缺失人数'), ('positive_recall', '共缺失召回')])},
]

chart = {
    'id': 'macro', 'title': '四种方案在三个训练 seed 下的验证宏 AUROC',
    'subtitle': '同一开发验证集 63 人；每根柱为各自按预设规则选中的模型。',
    'showDescription': True, 'type': 'bar', 'intent': 'comparison',
    'question': '逐字段监督的改善和分区输入的增益能否跨训练 seed 重复？',
    'rationale': '训练 seed 是离散重复，采用分组柱，不连成时间趋势。',
    'comparisonContext': {'grain': '每方案每训练 seed', 'denominator': '同一验证集 63 人，三个已知分子字段',
                          'unit': 'AUROC（0–1）', 'baseline': '各方案共同数据、标签、轮数和选模规则'},
    'dataset': 'methods', 'sourceId': 'audit', 'layout': 'full', 'valueFormat': 'number',
    'encodings': {'x': {'field': 'method', 'type': 'nominal', 'label': '输入与监督方案'},
                  'y': {'fields': ['seed42', 'seed43', 'seed44'], 'type': 'quantitative', 'label': '验证宏 AUROC'}},
    'palette': {'kind': 'categorical'}, 'legend': {'position': 'bottom', 'sort': 'spec', 'title': '训练随机性'},
    'labels': {'values': 'all'}, 'settings': {'groupMode': 'grouped', 'categoryLabelPolicy': 'wrap', 'sort': 'none'},
    'referenceLines': [{'axis': 'y', 'value': 0.5, 'label': '常数分数 AUROC 0.5', 'color': 'neutral', 'lineStyle': 'dashed'}],
}

artifact = {
    'surface': 'report',
    'manifest': {'version': 1, 'surface': 'report', 'title': title,
                 'description': 'PASA 修复实验三个训练 seed 的验证结果、完整性和证据边界。',
                 'generatedAt': now, 'sources': sources, 'charts': [chart], 'tables': tables,
                 'cards': [], 'blocks': blocks},
    'snapshot': {'version': 1, 'generatedAt': now, 'status': 'ready',
                 'datasets': {'methods': methods, 'pairs': pairs, 'fields': field_rows, 'codel': codel_rows}},
    'sources': sources,
}
(OUT / 'artifact.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print('Prepared canonical artifact.json from audited summary.json')
