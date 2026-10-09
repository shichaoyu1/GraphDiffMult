# 实现验证与结果状态

2026-10-09。本轮纳入CARZero(CVPR2024)/RadZero(NeurIPS2025)官方模块的MRI元数据适配，执行条件/差异见SOTA_ALIGNMENT_RUN.md。用户明确由自己在AutoDL启动；这里没有正式队列成绩。

验证命令：`python -B -m unittest discover -s tests -v`。退出码0，29项测试通过，其中10项为新比较任务。覆盖全部6方案的真实网络随机张量前向/反向、每方案一轮合成缓存/假数据加载训练、验证选模、测试记录、配对汇总；未知标签梯度、CARZero padding、RadZero singleton、MP-NCE在原group_map特殊情况下与官方公式数值一致；固定文本缓存只使用训练词表、mask均值和checksum；真正跨患者AUROC的单类不可计算边界。

初轮验证发现并修正RadZero逐正例损失把标量传入cat的问题；最终合并全部正例项再平均，保留原MP-NCE正例配对加权方式，后续完整回归通过。官方模块修改和来源见third_party/ALIGNMENT_SOURCES.json，各许可完整保留。

本地PyPI和wheel HTTPS下载失败，未把依赖安装失败解释为方法不支持。通过Git读取固定HuggingFace4.49.0源码，抽取eager/GELU前向块作为RadZero两层adapter，实际训练阶段只依赖PyTorch。文本缓存构建仍使用Transformers，并已测试接口与缓存完整性；本地没有成功下载真实MPNet权重，没有用随机缓存冒充预训练结果。服务器首次需要下载真实模型或指定本地模型路径。

临床/科研验收仍待：服务器真实数据读入和显存检查、字段字典/缺失与类别核验、18个完整matched运行、结果审计。所有论文主张必须根据这些真实结果决定；高Hit@5或patient属性检索不能证明区域病理真实性。
