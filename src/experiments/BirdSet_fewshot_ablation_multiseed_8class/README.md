# BirdSet严格小样本、多随机种子与跨域复测

本目录记录从Xeno-canto八分类DS-CNN权重开始的BirdSet SSW grouped
5/10/20-shot严格适配实验。实验覆盖MFCC、LogMel、PCEN三种输入特征，
`Head-Only`、`BN+Head`、`BN+Head+Replay`、`Full Fine-Tuning`四种策略，
以及42、123、2026三个随机种子，共108个FP32模型。

## 数据隔离与shot定义

shot表示“每类期望包含的五秒正片段数”，但划分单位是完整原始长录音。
为避免同一声景泄漏到support和test，实际support片段数可以显著超过shot目标。
三个特征使用完全相同的录音和片段身份，support按5 ⊂ 10 ⊂ 20嵌套。

| 目标规模 | support长录音 | support五秒片段 | 对应独立test长录音 | test五秒片段 | 类别5缺口 |
|---:|---:|---:|---:|---:|---:|
| 5-shot | 10 | 684 | 201 | 18,994 | 3 |
| 10-shot | 11 | 699 | 200 | 18,979 | 8 |
| 20-shot | 14 | 1,413 | 197 | 18,265 | 18 |

类别5 `Setophaga_ruticilla` 在完整目标子集中只有同一长录音内的2个正片段，
因此不能严格达到5/10/20个正片段。本实验明确记录缺口，不复制稀有样本凑数。
为了让三种shot结果可以直接比较，所有模型最终均在20-shot划分中完全隔离的
197条长录音、18,265个五秒片段和91,325个一秒窗口上复测。该集合也是零样本与
DB3V适配链路当前统一使用的BirdSet公共test；可复现规格和身份哈希见
`../BirdSet_common_test_8class/test_protocol.json`。

## 严格策略与选型

- `head_only`：只训练最后的八类sigmoid Dense层。
- `bn_head`：训练全部BatchNorm层和最后分类层。
- `bn_head_replay`：在`BN+Head`基础上，以1:1比例加入类别平衡的
  Xeno-canto训练集回放样本。
- `full`：训练全部层，包括BatchNorm层。

每个seed使用一个不同的高质量support内部录音级验证变体，并改变回放抽样、
batch顺序和TensorFlow随机操作。选型分数为“support验证片段Top-1任意目标命中率
× 截断到1的Xeno-canto Macro-F1保留率”。BirdSet公共held-out和完整DB3V
均在训练及策略/轮次选择完成后才读取，不参与选择。

## 三seed均值优选结果

下表每个数值均为三个seed的均值±样本标准差（`ddof=1`）。BirdSet Top-1表示
五秒片段的最高概率类别命中任一multi-hot目标；BirdSet F1只在全局纯单物种片段
的held-out支持类别上计算。Xeno使用来源录音级八类Macro-F1；本表的DB3V列保留
原实验生成时的八秒块级次要指标，只用于追溯，不作为当前DB3V主指标。

| 特征 | shot | 均值优选策略 | 选型分数 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | Xeno Macro-F1 | DB3V八秒块Macro-F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| MFCC | 5 | BN+Head+Replay | 14.92% ± 6.63% | 20.06% ± 0.57% | 13.32% ± 0.56% | 48.41% ± 0.44% | 48.45% ± 0.47% |
| MFCC | 10 | BN+Head+Replay | 15.30% ± 6.43% | 19.88% ± 0.29% | 13.44% ± 0.25% | 47.89% ± 0.61% | 48.45% ± 0.23% |
| MFCC | 20 | BN+Head+Replay | 41.84% ± 1.72% | 23.89% ± 0.44% | 17.57% ± 0.49% | 48.53% ± 0.40% | 49.23% ± 0.48% |
| LogMel | 5 | BN+Head+Replay | 18.20% ± 2.03% | 20.31% ± 2.17% | 13.32% ± 2.35% | 56.10% ± 0.63% | 59.47% ± 1.52% |
| LogMel | 10 | BN+Head+Replay | 18.62% ± 1.85% | 20.39% ± 2.02% | 13.66% ± 2.57% | 56.11% ± 1.31% | 59.33% ± 1.67% |
| LogMel | 20 | Head-Only | 40.50% ± 1.05% | 18.66% ± 0.91% | 10.62% ± 0.75% | 57.81% ± 1.24% | 63.73% ± 1.05% |
| PCEN | 5 | BN+Head | 26.76% ± 14.98% | 36.11% ± 1.11% | 27.11% ± 0.40% | 55.27% ± 1.34% | 61.41% ± 1.73% |
| PCEN | 10 | BN+Head | 27.03% ± 16.35% | **37.56% ± 2.08%** | **28.06% ± 1.60%** | 54.76% ± 1.29% | 60.56% ± 1.75% |
| PCEN | 20 | BN+Head | **48.66% ± 1.96%** | 29.68% ± 0.41% | 23.42% ± 0.18% | **57.85% ± 0.06%** | **63.67% ± 0.13%** |

FP32下，PCEN 10-shot `BN+Head`取得最高BirdSet实际held-out Top-1和纯单物种
Macro-F1；PCEN 20-shot `BN+Head`则有最高选型分数，并在Xeno保留和DB3V跨域
结果上更均衡。5/10/20-shot不是简单独立样本计数，新增完整长录音会一次加入大量
相关片段并改变support内部验证构成，因此实际held-out表现不保证随shot单调上升。
这些FP32排序不能直接用于INT8部署；对应量化结果及选中27个模型在完整DB3V
1,363条原始来源录音上的当前主指标，见相邻
`BirdSet_strict_INT8_quantization_8class/`目录。

## 可追溯来源与复现

- BirdSet数据集：[Hugging Face官方页](https://huggingface.co/datasets/DBD-research-group/BirdSet)
- BirdSet代码：[官方GitHub仓库](https://github.com/DBD-research-group/BirdSet)
- BirdSet论文：[ICLR 2025论文页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/484d254ff80e99d543159440a06db0de-Abstract-Conference.html)
- SSW底层声景：[Zenodo 7079380](https://zenodo.org/records/7079380)
- DB3V跨域测试：[Zenodo 11544734](https://doi.org/10.5281/zenodo.11544734)
- 固化协议与来源：`experiment_protocol.json`
- 108个逐seed结果：`runs.csv`
- 36个策略聚合结果：`aggregate.csv`
- 每组原始模型与报告：
  `{MFCC,LogMel,PCEN}/{5shot,10shot,20shot}/{head_only,bn_head,bn_head_replay,full}/seed_<seed>/`
- 本地划分清单（数据目录不进入Git）：
  `src/dataset_processing/output/<feature>_BirdSet_external_split*_8class/split_manifest.json`

从仓库根目录复现全部实验或只重建汇总：

```powershell
& .\.venv\Scripts\python.exe src\experiments\run_birdset_ablation_multiseed.py
& .\.venv\Scripts\python.exe src\experiments\run_birdset_ablation_multiseed.py `
  --summarize-only
```
