# 实验目录

本目录集中保存项目的模型构建、训练实验、模拟工件与评估报告：

- `construct_model/`：DS-CNN、CNN、BC-ResNet 和 MobileNetV2 模型结构；
- `Tranin.py`：八类单标签训练入口；
- `Train_birdset.py`：BirdSet 多标签候选基准诊断训练入口；
- `fine_tune_db3v.py`：从 Xeno-canto DS-CNN 权重开始的 DB3V
  5/10/20-shot 地区适配与无泄漏选型入口；同时保留历史策略和严格
  Head-Only、BN+Head、BN+Head+Replay、Full Fine-Tuning；
- `run_db3v_ablation_multiseed.py`：执行三种特征、三种shot、四种严格策略和
  三个随机种子的108组实验，并生成逐seed和均值/样本标准差汇总；
- `fine_tune_birdset.py`：从 Xeno-canto DS-CNN 权重开始的 BirdSet SSW
  grouped 5/10/20-shot 多标签适配入口；同时支持历史三策略和严格
  Head-Only、BN+Head、BN+Head+Replay、Full Fine-Tuning；
- `run_birdset_ablation_multiseed.py`：执行三种特征、三种shot、四种严格策略和
  三个随机种子的108组BirdSet实验，并完成共同BirdSet held-out、Xeno遗忘和
  完整DB3V跨域复测；
- `summarize_birdset_fewshot.py`：从原始 JSON 重建 BirdSet 选型与跨域对比 CSV；
- `int8_inference.py`：MFCC、LogMel、PCEN 特征感知的严格 INT8 转换、接口验证
  和批量 TFLite 推理；
- `evaluate_int8_experiments.py`：复测全部 FP32 已选中的零样本与小样本链路，
  并生成逐链路量化报告；
- `evaluate_db3v.py`、`evaluate_birdset_ssw.py`：外部数据集评估入口；
- `convert.py`：模型转换与部署工件生成入口；
- `TinyML_model_8class/`：MFCC 兼容基准模型及报告；
- `Feature_comparison_8class/`：Xeno-canto 三特征正式实验；
- `BirdSet_baseline_8class/`、`BirdSet_feature_comparison_8class/`：BirdSet 候选基准诊断；
- `DB3V_baseline_diagnostic_8class/`：DB3V 候选基准诊断；
- `DB3V_fewshot_8class/`：MFCC、LogMel、PCEN 三条 DS-CNN 链路的 DB3V 5-shot
  策略搜索、选中模型及 DB3V/BirdSet 复测报告；
- `DB3V_fewshot_10shot_8class/`、`DB3V_fewshot_20shot_8class/`：目标
  10/20-shot 的全部候选、模型和复测报告；
- `DB3V_fewshot_comparison_8class/`：5/10/20-shot 的统一选型记录与共同留出集
  对比汇总；
- `DB3V_fewshot_ablation_multiseed_8class/`：严格四策略的108组模型、Xeno遗忘
  测试、共同DB3V held-out报告以及36组聚合统计；
- `BirdSet_fewshot_8class/`：三种特征、三种微调策略的 BirdSet 小样本模型，
  以及 BirdSet held-out、Xeno-canto、DB3V 复测和统一汇总；
- `BirdSet_fewshot_ablation_multiseed_8class/`：BirdSet严格四策略的108组模型、
  三域复测、逐seed记录以及36组均值/样本标准差；
- `INT8_quantization_8class/`：15条选中链路的严格INT8模型、输入接口、逐链路
  混淆矩阵/粒度指标和FP32→INT8统一汇总；
- `BirdSet_strict_INT8_quantization_8class/`：BirdSet严格实验中按三seed平均适配
  分数选中策略后，对全部选中seed模型执行的严格INT8量化与聚合。

正式基准模型仍只由 Xeno-canto 训练和内部验证决定；BirdSet 和 DB3V 的候选基准
实验只用于方法诊断，不改变两者作为外部测试集与小样本增强集的正式角色。小样本
策略使用各自 support 与 Xeno-canto 保留率选型，但不读取 held-out。完整协议、
指标解释和数据来源见项目根目录的 `README.md`。
