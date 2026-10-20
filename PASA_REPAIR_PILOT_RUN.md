# 输入修复后的最小对照实验

版本：`pasa_repair_pilot_v1`，2026-10-20。阶段为探索；只使用原train/val，不建立test loader。

## 先核实到的边界

标签3和5没有可确认的人工修订数值字典，用户也无额外说明。具体证据见 `revision/pasa_review_20261009/LABEL_DEFINITIONS_20261020.md`。本实验分别保留1/2/3/4/5原始编号，不命名为五类组织。

## 四组与共同条件

| 方法 | 输入/汇总 | 监督目标 |
|---|---|---|
| whole_roi_mass | 相同病灶联合ROI | 正标签集合总概率 |
| whole_roi_field | 相同病灶联合ROI | 每患者已知字段内分别监督 |
| region_pool_mass | 同一ROI按原始编号分区，汇总有效区域 | 正标签集合总概率 |
| region_pool_field | 同一ROI按原始编号分区，汇总有效区域 | 每患者已知字段内分别监督 |

全部共用相同CNN定义、头、可学习锚点容量、初始化种子、训练患者、词表、切片和预算。区域方案共享CNN，不因区域数增加参数量，但前向视图更多，计算量并不完全相同。所有方案先形成患者分数，再使用单向image→anchor监督；不加图/扩散/中心损失/类别权重。目标是识别输入分区与损失定义的作用，不是完整PASA验证。

原始metadata已知字段提供训练监督，缺失字段掩码；词表从实际保留的训练集建立。逐字段目标先平均每患者已知字段，再平均患者；评价主终点是验证集IDH/MGMT/1p19Q可估字段的宏AUROC，同时报告每字段样本量、mAP和频率基线。smoke可能仅部分字段可估，不能据其宏值宣称三个字段全部验证。

## 输入处理

使用旧审计的分割文件与SHA256；保持原train/val成员身份。对1–5联合前景选择面积最大的切片中心，同面积取较小z；最近邻缩放后重新判断区域存在性。空区域不进入CNN/BatchNorm，不进入患者汇总。可选图实现亦屏蔽缺失行列，单节点图无边。

全分割空或resize后完全空的病例，所有方法共同显式排除，冻结原因与SHA256，不挑选替代患者或静默更换分割。各方案的保留患者一致。准备阶段检查四模态形状/affine并冻结路径、大小和mtime；原始MRI体素未计算哈希。每次读取核对分割字节哈希与图像stat。全部原始训练结果保留原样。

## 验证与选模

验证是无增强、有固定患者顺序的推理。固定且隔离Python/NumPy/torch/CUDA随机状态，恢复验证loader generator，启用确定性算法；完成前重放并要求分数一致。pilot没有扩散随机采样。训练随机状态不受验证消耗影响。

主检查点按验证宏AUROC严格提高保存，同分保留首次；额外保留best-map和last，每轮患者验证分数均写出。AUROC不可估时报错，不回退到mAP。该规则对四组相同。

## AutoDL执行

沿用当前实验环境与数据，无Hugging Face下载。

```bash
cd /root/autodl-tmp/repo/GraphDiffMult
git pull --ff-only
bash tools/launch_pasa_repair.sh smoke
```

默认smoke：四组各1轮，原训练/验证中保留病例的前32人，seed42。扫描完整train/val后再取smoke子集；候选词表仅从smoke训练患者建立。日志和结果路径由启动器打印，可关闭终端。日志出现COMPLETE且目录中有complete.json才算完成。

核对smoke的排除人数、字段覆盖、重复验证及四组完整性后，启动10轮pilot：

```bash
bash tools/launch_pasa_repair.sh pilot --smoke-reference output/实际的pasa_repair_smoke目录
```

pilot使用全部保留train/val，四组×10轮×seed42。必须提供相同代码、ROI、种子与方法配置的已完成smoke。新实验实际性能尚未测量，耗时不作未经实测的承诺。

前台或查看计划：

```bash
python -u -B tools/run_pasa_repair_pilot.py --stage smoke
python -B tools/run_pasa_repair_pilot.py --stage pilot --dry-run
```

## 输出与下一关

新目录保存protocol、输入manifest、共同排除、词表、各方法history、每轮验证分数、best_auc.pt/best_map.pt/last.pt、频率基线、summary和complete。已有目录拒绝覆盖，失败留下完整attempt证据；需新目录重试，不静默续训。

本轮不使用旧test重新选方案。若区域输入或字段监督在验证上无稳定增量，应保留负结果并简化路线；单seed开发观察不构成最终确认。最终确认还需预先锁定的独立数据/外部评估。

## 补充训练种子43、44

seed42已完成四组10轮。新入口只复用该次protocol、inputs和词表，不重新prepare或重复smoke，不修改任何科学训练源码。新调度器记录参考材料SHA256、当前git提交和自身源码SHA256；每个新seed仅改变训练随机性。

默认参考服务器目录为 `output/pasa_repair_pilot_20261020_095259_432520`。同一环境执行：

```bash
cd /root/autodl-tmp/repo/GraphDiffMult
git pull --ff-only
bash tools/launch_pasa_repair_repeats.sh
```

依次执行seed43的四组、seed44的四组，共8个新运行；各10轮、batch2、435train/63val、原AUROC选模等参数全部继承参考协议。无测试集评分或HF下载。不会重跑seed42。日志和新结果目录由启动器打印，可以关闭终端。

原四组GPU训练累计约6.47小时，因此两个新seed可参考约13小时，另加输入预检和磁盘读取；这是按上次实测估计，不是完成时间承诺。

实际路径不同可追加 `--reference /实际路径/seed42目录`。查看计划而不加载影像或训练：

```bash
python -B tools/repeat_pasa_repair_pilot.py --reference output/pasa_repair_pilot_20261020_095259_432520 --dry-run
```

完成标志为新目录根部complete.json和日志COMPLETE。输出包含seed_43/attempt_001、seed_44/attempt_001与汇总，汇总纳入原seed42共12个运行。

### 失败恢复

失败会保留attempt及failed.json，停止队列。当前模型文件没有完整优化器/随机状态，故不从last.pt继续。恢复时已完成并核验通过的seed跳过；未完成seed在新attempt从四组开始，完整保留旧attempt。

```bash
bash tools/launch_pasa_repair_repeats.sh --resume --out output/实际的pasa_repair_repeats目录
```

恢复时校验科学源码、参考材料、元数据、分割、图像stat、已完成attempt的输入及词表，发生变化则停止；不会修改旧哈希来绕过冻结。并发恢复由.run.lock阻止，普通失败自动释放锁。若进程被强制杀死遗留锁，先核实锁内PID确已停止，再处理该锁文件，不在活动进程上重复启动。

三个seed共用开发验证集，仅用于检验训练随机性与配对方向，不表示三个独立验证队列。结果返回前不增加模块或延长预算。
