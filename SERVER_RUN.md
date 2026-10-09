# AutoDL：PASA v2 实验运行

日期：2026-10-09。已按你的数据路径和 RTX 4080 配置。项目建议解压到 `/root/autodl-tmp/pasa_v2_code_20261009/brats_fusion`，默认 batch_size=2、ROI=96、7切片；单卡顺序运行。本地19项测试通过，含实际网络在随机张量上的前向/反向；服务器实测前不能据此推断真实显存或训练时间。

## 1. 上传与环境

将 `artifacts/PASA_SERVER_V2.zip` 上传到 AutoDL 的 `/root/autodl-tmp/`，然后：

```bash
mkdir -p /root/autodl-tmp/pasa_v2_code_20261009
cd /root/autodl-tmp/pasa_v2_code_20261009
unzip /root/autodl-tmp/PASA_SERVER_V2.zip
cd brats_fusion
sha256sum -c SHA256SUMS.txt
export MPLBACKEND=Agg
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

使用 AutoDL 中已安装且 CUDA 可用的 PyTorch 环境。若其他依赖未安装，补充本项目的读取/绘图依赖：

```bash
python -m pip install nibabel numpy matplotlib scipy
python -B -m unittest discover -s tests -v
```

若检查显示 CUDA unavailable，先修复服务器 PyTorch/CUDA 环境；不要用 CPU 结果冒充单卡训练。实验入口不依赖 torchvision。运行目录会保存实际环境版本，当前没有规定必须升级已有 PyTorch。

## 2. 数据预检与冻结划分

```bash
python -B tools/run_pasa_server.py \
  --stage prepare \
  --output_root /root/autodl-tmp/pasa_runs/pasa_v2_20261009
```

已内置的两个默认路径：

- MRI：`/root/autodl-tmp/dataset/UTSW-Glioma`
- 元数据：`/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv`

预检会检查 CUDA、metadata、患者划分、分子标签覆盖，保存 `prepared.json`、`splits.json`、`environment.json` 和环境版本列表。空验证/测试集会报错，需要确定有效划分后换新目录预检，不能复用训练/测试患者。请查看 missing_fields 和 oov_positives；这一步检查数据可运行性，不替代患者别名、中心分布和临床标签字典核验。

## 3. 一轮真实数据检查

```bash
python -B tools/run_pasa_server.py \
  --stage smoke \
  --output_root /root/autodl-tmp/pasa_runs/pasa_v2_20261009
```

smoke 会用完整冻结划分和实际ROI，只跑 minimal-unit 与 full 各1轮、seed42。不要额外加 `--epochs 1`：smoke 内部自动采用1轮，正式实验计划仍保持默认30轮，以便使用相同 campaign。这个阶段用于检查数据读取、显存、前向/反向、保存和绘图，不用于论文选优。

如显存不足，换新 output_root，从 prepare 开始统一设置较小的 batch_size；同一比较组保持设置一致，不单独让某个模型改变输入尺寸。

## 4. 第一组核心实验：12个运行

```bash
mkdir -p logs
nohup env MPLBACKEND=Agg python -u -B tools/run_pasa_server.py \
  --stage core \
  --output_root /root/autodl-tmp/pasa_runs/pasa_v2_20261009 \
  > logs/pasa_v2_core.log 2>&1 &
echo $! > logs/pasa_v2_core.pid
tail -f logs/pasa_v2_core.log
```

4个方案×seeds42/43/44，默认30轮：

| 运行名称 | 问题与实际实现 |
|---|---|
| unit_retrieval | 最小区域表征+多正样本检索，无图/扩散/中心约束 |
| pooled_global | 同编码器区域均值全局查询；不是整幅MRI编码器 |
| unit_multilabel | 同原型输出的masked BCE；不是独立已发表系统 |
| unit_single_positive | 检索目标抽一个正例，其他真阳性屏蔽；中心约束关闭 |

所有方案用患者属性监督、共同分子评价、同样冻结划分/词表/预算。不要将核心实验中的内部基线称为已发表 CLIP/MedCLIP/PMC-CLIP 的直接复现。

## 5. 第二组组件实验：15个运行

核心结果审完后再运行：

```bash
nohup env MPLBACKEND=Agg python -u -B tools/run_pasa_server.py \
  --stage ablation \
  --output_root /root/autodl-tmp/pasa_runs/pasa_v2_20261009 \
  > logs/pasa_v2_ablation.log 2>&1 &
```

full、去病理辅助监督、去中心约束、去图、去扩散，分别跑3个种子。无论图/扩散结果如何，保留全部运行。`--stage all` 可顺序运行核心与组件共27个正式运行，不包含smoke。先做smoke和核心组再决定后续预算。

## 6. 中断、日志和汇总

重复同一命令时，已完成且命令相同的运行自动跳过。失败运行保留在 `attempt_001` 等目录，下一次创建新attempt；不是从中断epoch续训。更改源码、metadata、输入尺寸或正式轮数后，应使用新的 output_root，脚本会阻止在原campaign混跑。

```text
pasa_v2_20261009/
  prepared.json / splits.json / environment.json / pip_freeze.txt
  runs/smoke/.../attempt_001/
  runs/core/.../attempt_001/
  runs/ablation/.../attempt_001/
  summary.csv
```

每次成功保存完整日志、checkpoint、配置、词表、protocol.json、逐患者记录与指标。`complete.json` 标记服务器任务成功，summary.csv仅收录已完成运行。Hit@k和Recall@k已分开，字段/患者等权；无真值和词表外真值有单独计数。旧实验分数不可与这些新分数混表。

全部核心运行完成后，可以计算固定已训练种子的患者bootstrap区间：

```bash
python -B utils/bootstrap_semantic_5seed.py \
  --output_root /root/autodl-tmp/pasa_runs/pasa_v2_20261009/runs/core \
  --n_bootstrap 2000 \
  --out_csv /root/autodl-tmp/pasa_runs/pasa_v2_20261009/core_bootstrap.csv \
  --out_json /root/autodl-tmp/pasa_runs/pasa_v2_20261009/core_bootstrap.json
```

工具名称保留历史5seed，但实际接受任意种子数。核心与组件分开汇总；该区间不覆盖训练程序全部随机性，也不直接给出配对方案差的显著性。给我带回 summary.csv、prepared.json、各成功运行的protocol.json和逐患者记录，即可继续核对、做配对差分析和决定叙事。患者记录包含本地ID，按你的数据共享规则传递。

旧 `run_all_pro.sh` 及其历史说明不是此次v2实验调度入口；使用新的Python调度器，避免旧重试、指标名和参数习惯混入。
