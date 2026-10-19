# AutoDL 第 2、3 步真实数据诊断

入口 `tools/diagnose_pasa_server.py`；后台启动器 `tools/launch_pasa_diagnostics.sh`。

默认使用原服务器项目中的 `output/sota_alignmentv2_20261009`，三个完整 PASA 检查点（42/43/44），UTSW 影像 `/root/autodl-tmp/dataset/UTSW-Glioma`，元数据 `/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv`。不下载 Hugging Face 模型，不需要重新创建文本缓存。沿用当前已安装的实验环境。

原正式队列结束后，在相同环境执行：

```bash
cd /root/autodl-tmp/repo/GraphDiffMult
git pull --ff-only
bash tools/launch_pasa_diagnostics.sh
```

启动器输出 PID、结果目录和日志路径，可关闭终端。启动成功只表示进程已创建，以日志末尾 `COMPLETE:` 和结果目录内 `complete.json` 为真正完成标志。失败保留日志和部分输出，不会写 complete.json。

如路径不同：

```bash
bash tools/launch_pasa_diagnostics.sh --campaign /实际路径/sota_alignmentv2_20261009
```

也可前台执行：

```bash
python -u -B tools/diagnose_pasa_server.py --campaign output/sota_alignmentv2_20261009
```

## 执行范围

1. 验证冻结元数据、病例划分、词表、关键科学源码与历史辅助函数。JSON 输出补丁可以存在，科学定义发生变化则停止。
2. 对训练/验证患者扫描分割，统计原裁剪切片和网络输入尺寸下 label 1/2/4 的空区域数。与当前数据集相同的 ROI、切片、resize 规则。只加载分割与首个影像的形状，不对每个患者加载四种 MRI。
3. 对 3 个现有完整 PASA 检查点，各运行 3 次验证评分，再用相同随机种子复现第 1 次，保存每次患者分数与指标。
4. 每个检查点取 3 个固定顺序、无增强的训练批次，以 train 模式做梯度诊断；按该检查点的历史选模轮次应用图冻结与权重。记录加权损失、共享表示头参数的梯度范数、每个正锚点的分数梯度。

梯度检查使用历史权重与真实训练输入，但不是恢复历史训练轨迹：未恢复优化器、训练时的数据顺序与随机状态。结果是已保存检查点的局部敏感性。执行后恢复 BN/图 EMA 等缓冲，无 optimizer step。无测试集 loader。

## 输出

独立新目录 `output/pasa_diagnostics_<时间>_<启动PID>/`：

- `diagnostic_config.json`：参数、冻结协议、当前源码 SHA256、git HEAD。
- `region_records.json` / `region_summary.json`：训练/验证区域缺失情况。
- `pasa_full_s*/repeat_*.json`、`fixed_seed_replay.json`：验证重复评分。
- `pasa_full_s*/gradient_probe.json`：损失分项与梯度。
- `summary.json`、`complete.json`：汇总与完成标志。

新目录不得位于旧 campaign 内，已存在输出目录会拒绝覆盖。原检查点和结果保持原位。

完成后把这个新结果目录带回本地。它用于决定下一轮小实验；不是重新进行 45 个正式训练，也不会生成“后期模型”的测试成绩。耗时取决于磁盘和分割读取，尚未在真实 AutoDL 上测量。

## 本地审查与验证

真实患者数据由用户在 AutoDL 执行。本地覆盖空区域统计、诊断后模型参数/BN/EMA 不变、固定种子复现，以及完整命令的合成流程；后者还验证不读取测试病例、无参数更新、原检查点字节不变。合成流程不代表真实患者诊断已经运行。

阶段性汇总报告与诊断脚本随代码保留；患者级原始证据 ZIP 和返回结果留在本地。
