# DB3V严格微调与多随机种子实验

本目录保存 MFCC、LogMel、PCEN 三条 Xeno-canto DS-CNN 基准链路在 DB3V
5/10/20-shot support 上的严格微调消融。四种策略为：

- `head_only`：仅训练最终softmax层；
- `bn_head`：训练全部BatchNorm与最终softmax层；
- `bn_head_replay`：BN+Head，并按1:1加入类别均衡的Xeno训练切片；
- `full`：训练全部层，包括BatchNorm。

固定support实际包含120/239/461条录音，最终统一在20-shot support之外的10,197条
共同DB3V held-out录音上评估。随机种子为42、123、2026；每个结果均同步保存
Xeno-canto验证Macro-F1及其相对基模型的变化。共完成
`3特征 × 3 shot × 4策略 × 3 seed = 108` 组训练和评估。

按平均适配分数选中的链路另在统一BirdSet公共test上复测：197条长录音、
18,265个五秒片段、91,325个一秒窗口。规格与样本身份哈希见
`../BirdSet_common_test_8class/test_protocol.json`。

主要文件：

- `experiment_protocol.json`：数据隔离、随机性、策略和统计协议；
- `runs.csv`：108个逐seed结果；
- `aggregate.csv`：36个“特征×shot×策略”的均值和样本标准差；
- `<feature>/<shot>shot/<policy>/seed_<seed>/`：模型、训练历史、微调报告和
  DB3V逐地区/汇总评估；选中策略目录还包含统一BirdSet公共test跨域报告。

策略选择只读取support内部验证与Xeno保留率，不读取DB3V held-out。按三个seed的
平均适配分数，PCEN 20-shot `full`取得最高严格策略结果：DB3V共同held-out
Macro-F1为70.50%±0.40%，Xeno Macro-F1为63.66%±0.44%，相对基模型提高
10.98±0.44个百分点。按平均适配分数选中的9组策略现已对全部三个seed完成统一
BirdSet公共test跨域复测和严格INT8量化；量化结果位于相邻目录
`../DB3V_strict_INT8_quantization_8class/`。PCEN 20-shot `full`量化后DB3V
Macro-F1降至52.52%±0.79%，当前严格INT8推荐为LogMel 10-shot `head_only`，
DB3V Macro-F1为68.14%±0.17%。

复现入口为上级目录中的 `run_db3v_ablation_multiseed.py`；单组训练实现位于
`fine_tune_db3v.py`。DB3V原始数据来源为
[Zenodo 11544734](https://doi.org/10.5281/zenodo.11544734)。
