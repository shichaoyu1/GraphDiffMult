# 分割统计探针：AutoDL CPU 运行

这一步检查只有分割前景总量、原始编号组成和presence时能预测到什么程度。无需GPU，不加载MRI，不下载Hugging Face资源。没有真实结果前不预断它与CNN谁更好。

输入为已经完成的seed42修复pilot及seed43/44重复运行的绑定记录：

- `output/pasa_repair_pilot_20261020_095259_432520/`
- `output/pasa_repair_repeats_20261020_171817_445198/repeat_config.json`
- `/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv`

脚本核对参考5文件和元数据SHA256，按固定435名训练患者重建原词表/分子标签，仅训练患者拟合标准化和分类器。验证用同63名患者，缺失不当阴性。旧`train_prior_score_records.json`是测试病例上的训练频率基线，不能作为训练标签；本入口不读取它。

## 一键运行

更新代码后，在AutoDL终端执行：

```bash
cd /root/autodl-tmp/repo/GraphDiffMult
git pull --ff-only
bash tools/launch_pasa_shape_probe.sh
```

启动脚本使用CPU并输出后台日志和新结果目录，可关闭终端。此次计算本身不需要学术加速；拉取GitHub代码仍取决于服务器网络。

若原pilot或repeat目录改了位置：

```bash
bash tools/launch_pasa_shape_probe.sh \
  --reference /实际路径/pasa_repair_pilot_20261020_095259_432520 \
  --binding /实际路径/pasa_repair_repeats_20261020_171817_445198/repeat_config.json
```

元数据路径也可追加`--metadata /实际路径/UTSW_Glioma_Metadata-2-1.tsv`。已有结果不覆盖；不要修改旧hash或用不匹配文件绕过检查。

## 完成与回传

成功目录含`protocol.json`、`train_fit.json`、`validation_patient_scores.json`、`summary.json`及`complete.json`。先确认日志正常结束且`complete.json`存在；若仅有启动PID，尚不能认定成功。

下载整个新目录及同名`.log`到本地`output/20261019_AAAI_review/`，再核查结果。患者标签、系数拟合清单与逐患者分数仅在被Git忽略的output内，不能直接提交公开仓库。

## 科学范围

11特征和三个L2逻辑回归预先固定，不调C、不选阈值、不按验证结果挑特征。完整协议见`revision/pasa_review_20261009/SHAPE_PROBE_PROTOCOL_20261021.md`。

它与CNN的五任务共享训练、骨干和优化不同；任何差值都不能直接解释为MRI强度的增量。它也不提供区域组织学参考，原始label3/5的含义仍未知。若形态统计很强，后续需要同架构、同监督的掩膜/强度对照；若较弱，也只排除了这组有限统计和线性模型。
