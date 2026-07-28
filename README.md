# 鸟类鸣声分类项目

本项目仅使用 Xeno-canto 鸟鸣录音训练基准分类器。BirdSet SSW 与 DB3V
分别作为跨数据集声景泛化和特定地区泛化的独立测试集，并各自提供与测试录音
隔离的小样本 support 集。当前推荐模型是八类 Xeno-canto Log-Mel DS-CNN；
两个外部 held-out test 均不参与训练、早停或模型选择。BirdSet grouped
5/10/20-shot 多标签适配和 DB3V 5/10/20-shot 单标签适配均已完成。

## 当前八类配置

| ID | 类别 |
|---:|---|
| 0 | `Agelaius_phoeniceus` |
| 1 | `Cardinalis_cardinalis` |
| 2 | `Certhia_americana` |
| 3 | `Corvus_brachyrhynchos` |
| 4 | `Setophaga_aestiva` |
| 5 | `Setophaga_ruticilla` |
| 6 | `Spinus_tristis` |
| 7 | `Turdus_migratorius` |

当前原始集位于 `src/row_dataset/row_bird_dataset_A_next/`，共 2,400 条录音。每个文件均为 16 kHz、单声道、16-bit PCM WAV；全局 ID 与音频哈希已去重，并与 DB3V 来源 ID 隔离。

## 项目结构

```text
.
├── README.md
└── src/
    ├── README.md                           # src目录职责与Git边界
    ├── row_dataset/
    │   ├── data_crawler_xeno_canto.py       # Xeno-canto 爬虫；支持八类、A–E、DB3V 来源排除
    │   ├── data_crawler_DB3V.py             # DB3V 原始数据下载/整理
    │   ├── DB3V/
    │   │   └── extracted/data_wav_8s_2/     # 原始 DB3V WAV；仅用于独立评估/后续小样本实验
    │   ├── BirdSet_SSW/                     # BirdSet SSW 元数据缓存；原始分片不落盘
    │   ├── row_bird_dataset_A_next/         # 当前八类 Xeno-canto 原始训练数据
    │   └── xeno_removed_db3v_overlap.json   # 历史 DB3V/Xeno-canto 来源重叠记录
    ├── dataset_processing/
    │   ├── data_sugment_MFCC.py             # 八类共享预处理核心：MFCC / Log-Mel / PCEN
    │   ├── data_sugment_MFCC_DB3V.py        # DB3V 区域数据适配器，支持上述三种特征
    │   ├── data_prepare_birdset_ssw.py       # BirdSet SSW 流式筛选及多标签特征预处理
    │   ├── data_prepare_birdset_training.py  # BirdSet候选基准诊断切分（多标签）
    │   ├── data_prepare_db3v_training.py     # DB3V候选基准诊断切分（同地区留出）
    │   ├── prepare_external_fewshot.py       # BirdSet/DB3V support与held-out隔离划分
    │   ├── label_map_8class.json            # 当前八类连续标签映射
    │   └── output/                          # 生成特征；不进入Git
    └── experiments/
        ├── README.md                        # 实验目录与数据角色说明
        ├── Tranin.py                        # 单标签训练入口
        ├── Train_birdset.py                 # BirdSet多标签诊断训练
        ├── fine_tune_birdset.py             # BirdSet grouped 5/10/20-shot多标签适配
        ├── fine_tune_db3v.py                # DB3V 5/10/20-shot地区适配
        ├── run_birdset_ablation_multiseed.py # BirdSet严格策略多随机种子实验与汇总
        ├── run_db3v_ablation_multiseed.py   # DB3V严格策略多随机种子实验与汇总
        ├── int8_inference.py                # 特征感知严格INT8接口与批量推理
        ├── evaluate_int8_experiments.py     # 零样本/小样本INT8统一复测
        ├── evaluate_db3v.py                 # DB3V区域评估
        ├── evaluate_birdset_ssw.py          # BirdSet多标签评估
        ├── convert.py                       # 模型转换入口
        ├── construct_model/                 # 模型结构
        ├── TinyML_model_8class/             # MFCC兼容基准模型与报告
        ├── Feature_comparison_8class/       # Xeno三特征正式实验
        ├── BirdSet_fewshot_ablation_multiseed_8class/ # BirdSet严格108组实验
        ├── BirdSet_strict_INT8_quantization_8class/ # BirdSet严格链路INT8复测
        ├── DB3V_fewshot_ablation_multiseed_8class/ # 108组严格策略实验与统计
        ├── DB3V_strict_INT8_quantization_8class/ # DB3V严格链路INT8与BirdSet复测
        ├── ZeroShot_strict_INT8_quantization_8class/ # 当前零样本严格INT8
        ├── legacy_single_seed_reference.md  # 已淘汰单种子链路必要结论
        ├── BirdSet_baseline_8class/         # BirdSet MFCC诊断结果
        ├── BirdSet_feature_comparison_8class/
        └── DB3V_baseline_diagnostic_8class/
```

目录树只列出当前八类流程的规范路径。并行抓取暂存目录、旧实验工件、`__pycache__/`、`.venv/` 和日志文件都不是当前训练输入或源码结构。

## 推荐工作流

使用项目虚拟环境执行以下命令。重新抓取时不要使用 `--allow-partial`，以确保每个类别满足 300 条有效录音的下限。

```powershell
# 1. 获取八类 Xeno-canto 原始数据（默认接受 A,B,C,D,E 评分）
& .\.venv\Scripts\python.exe src\row_dataset\data_crawler_xeno_canto.py `
  --eligible-eight `
  --target-per-species 300 `
  --min-per-species 300 `
  --output-dir src\row_dataset\row_bird_dataset_A_next

# 2. 生成 Xeno-canto 训练/验证 MFCC 特征
& .\.venv\Scripts\python.exe src\dataset_processing\data_sugment_MFCC.py `
  --input-dir src\row_dataset\row_bird_dataset_A_next `
  --output-dir src\dataset_processing\output\MFCC_dataset_A_8class

# 3. 生成同标签顺序的 DB3V 独立测试特征
& .\.venv\Scripts\python.exe src\dataset_processing\data_sugment_MFCC_DB3V.py `
  --output-dir src\dataset_processing\output\MFCC_dataset_DB3V_8class `
  --xeno-label-map src\dataset_processing\output\MFCC_dataset_A_8class\label_map.json

# 4. 训练 DS-CNN；EarlyStopping 监控录音级 Macro-F1
$env:TF_CPP_MIN_LOG_LEVEL = '2'
& .\.venv\Scripts\python.exe "src\experiments\Tranin.py" `
  --model DS_CNN_Model `
  --dataset-dir src\dataset_processing\output\MFCC_dataset_A_8class `
  --model-dir "src\experiments\TinyML_model_8class" `
  --epochs 30 --patience 8

# 5. 在完全独立的 DB3V 上测试
& .\.venv\Scripts\python.exe "src\experiments\evaluate_db3v.py" `
  --models DS_CNN_Model `
  --dataset-dir src\dataset_processing\output\MFCC_dataset_DB3V_8class `
  --model-dir "src\experiments\TinyML_model_8class"
```

## 数据与评估约定

- Xeno-canto 按完整录音的 `session_key` 切分：当前为每类 240 条训练录音、60 条验证录音，训练/验证之间没有共享录音或会话。
- 每条录音最多贡献 8 个均匀分布的 1 秒片段；训练权重使每条录音总权重相同，避免长录音主导优化。
- 当前 MFCC 特征形状为 `(32, 13)`：Xeno-canto 训练/验证数组分别为 `(14545, 32, 13)` 和 `(3652, 32, 13)`；DB3V 数组为 `(85264, 32, 13)`，对应 10,658 条录音。
- Log-Mel 和 PCEN 的形状均为 `(32, 40)`，录音划分、切片索引及样本数与 MFCC 完全一致，因此特征对比不受数据划分差异影响。
- DB3V 保持为独立地区测试集。评估时会同时报告 1 秒切片指标，以及将一条 DB3V 录音的 8 个 softmax 向量平均后的录音级指标。
- 模型、`*.labels.json` 标签侧车、训练历史、验证摘要与 DB3V JSON/CSV 报告会保存在同一个模型目录中。转换为 TFLite 或 C 头文件时，单独运行 `convert.py`。
- Xeno-canto API 密钥不写入仓库；运行爬取脚本前必须设置
  `XENO_CANTO_API_KEY` 环境变量。音频标准化依赖 FFmpeg。

## 数据来源、许可与实验角色

| 数据 | 上游来源 | 本实验中的角色 | 许可/引用说明 |
|---|---|---|---|
| Xeno-canto | [Xeno-canto](https://xeno-canto.org/) 及其 [API](https://xeno-canto.org/explore/api) | 2,400 条焦点录音用于训练和内部验证 | 每条录音的 Creative Commons 许可保存在对应 `metadata.jsonl`；再分发或商用时必须逐条遵守 |
| DB3V | [Zenodo 11544734](https://doi.org/10.5281/zenodo.11544734) | held-out测试三个地区泛化；隔离support已用于5/10/20-shot地区适配 | Zenodo记录标注为 CC BY 4.0；使用时应引用数据集记录 |
| BirdSet | [官方数据集](https://huggingface.co/datasets/DBD-research-group/BirdSet)、[官方代码](https://github.com/DBD-research-group/BirdSet)、[ICLR 2025 论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/484d254ff80e99d543159440a06db0de-Abstract-Conference.html) | SSW held-out测试跨数据集多标签声景泛化；隔离support用于后续小样本增强 | BirdSet代码为 BSD-3-Clause；底层音频仍遵循各来源许可 |
| SSW底层声景 | [Zenodo 7079380](https://zenodo.org/records/7079380) | BirdSet SSW评估音频的原始来源 | 当前保留片段的清单均标记为 CC BY 4.0，逐片段来源和许可保存在 `manifest.json` |

BirdSet论文引用：

```bibtex
@inproceedings{rauch2025birdset,
  title     = {BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics},
  author    = {Lukas Rauch and Raphael Schwinger and Moritz Wirth and
               Ren{\'e} Heinrich and Denis Huseljic and Marek Herde and
               Jonas Lange and Stefan Kahl and Bernhard Sick and
               Sven Tomforde and Christoph Scholz},
  booktitle = {International Conference on Learning Representations},
  year      = {2025}
}
```

## 当前实验进度

截至 **2026-07-28**，已经完成：

1. 确定八个目标物种和固定标签顺序。
2. 获取并清洗每类 300 条、总计 2,400 条 Xeno-canto 录音；执行来源 ID、音频哈希和 DB3V 来源隔离。
3. 按完整录音会话切分为每类 240 条训练录音和 60 条验证录音；训练与验证不共享录音或会话。
4. 完成统一的 16 kHz、单声道、一秒窗口处理，并在相同录音划分和切片上生成 MFCC、Log-Mel、PCEN。
5. 训练八分类 DS-CNN，以录音级 Macro-F1 监控 EarlyStopping 并保存最佳权重。
6. 完成三种特征在 DB3V 三个地区的 1 秒切片级及 8 秒录音级独立评估。
7. 流式读取 BirdSet SSW 全部四个 `test_5s` 分片（约 7.85 GB），未在本地保存完整原始分片。
8. 从 205,200 个 SSW 片段中筛出 19,678 个命中目标物种的五秒多标签片段，分别生成 98,390 个 MFCC、Log-Mel、PCEN 一秒窗口。
9. 将 BirdSet SSW 和 DB3V 分别划分为来源隔离的 few-shot support 与 held-out test，完成最佳 Xeno-canto Log-Mel 模型的两类零样本外部评估。
10. 补做 BirdSet SSW 与 DB3V 分别作为训练来源的反事实候选基准实验；结果仅用于检验其基准适用性，不改变正式数据角色。
11. 建立DB3V嵌套support：120条 ⊂ 239条 ⊂ 461条；针对稀疏分层明确记录
    10-shot缺1条、20-shot缺19条，不使用重复录音补齐。
12. 完成严格 `Head-Only`、`BN+Head`、`BN+Head+Replay` 和包含BatchNorm的
    `Full Fine-Tuning`；在三种特征、5/10/20-shot和随机种子42/123/2026上完成
    108组训练、Xeno-canto遗忘测试和共同DB3V held-out评估，并报告均值与样本标准差。
13. 将同一严格四策略、多随机种子协议扩展到BirdSet grouped 5/10/20-shot；
    每组均复测共同BirdSet 20-shot held-out、Xeno-canto和完整DB3V，并对按三seed
    平均适配分数选中的9组策略、27个seed模型执行严格INT8量化与三域复测；
    27个模型均为int8输入/输出、0个浮点张量。
14. 对DB3V严格实验按三seed平均适配分数选中的9组策略、27个seed模型补齐
    BirdSet SSW FP32跨域复测和严格INT8量化；全部模型重新复测Xeno-canto、
    BirdSet及10,197条共同DB3V held-out，均为int8输入/输出、0个浮点张量。
15. 将当前零样本、DB3V适配和BirdSet适配结果统一到同一公共测试集：排除
    嵌套20-shot support后的197条长录音、18,265个
    五秒片段和91,325个一秒窗口。MFCC、LogMel、PCEN样本身份SHA-256一致，
    support/test录音重叠为0。
16. 将三个仍有效的零样本INT8模型迁入独立当前目录；删除旧单种子模型、旧策略
    入口和失效复现命令，只保留必要结论与原汇总SHA-256供方法追溯。

尚未完成的实验包括：背景/未知类别、多鸟混合增强、PCEN量化感知训练、其他模型在
当前八类规范数据上的公平复测，以及改变外部support抽样的重复实验和置信区间估计。

## 实验链路总览：从Xeno-canto基准到严格INT8

当前所有正式可比实验均从同一套八分类Xeno-canto权重开始，模型统一为DS-CNN。
BirdSet与DB3V从零训练实验仅用于数据角色诊断，不进入本节主链路排名。

```text
Xeno-canto训练
  ├── MFCC / LogMel / PCEN FP32基准
  ├── BirdSet、DB3V零样本泛化
  ├── DB3V 5/10/20-shot单标签地区适配
  ├── BirdSet 5/10/20-shot多标签声景适配
  └── 选中FP32模型 → PTQ严格INT8 → Xeno/BirdSet/DB3V复测
```

### 统一结果口径

| 数据集 | 主要指标 | 测试粒度与范围 |
|---|---|---|
| Xeno-canto | Macro-F1 | 480条内部验证录音；同一录音的一秒窗口先聚合 |
| DB3V零样本/BirdSet适配复测 | Macro-F1 | 完整10,658条八秒单标签录音 |
| DB3V小样本适配 | Macro-F1 | 20-shot support之外固定的10,197条共同held-out录音 |
| BirdSet（所有当前链路） | Top-1命中任一真实标签、纯单物种支持类Macro-F1 | 固定公共test：197条长录音、18,265个五秒片段、91,325个一秒窗口 |

BirdSet公共test由种子42的grouped 20-shot划分确定；因5/10-shot support嵌套于
20-shot support，该集合同时排除全部5/10/20-shot训练录音。完整211条、5-shot
held-out 201条和10-shot held-out 200条结果仅作为历史诊断，不进入当前横向排名。
协议、样本身份哈希和三种特征核验记录见
`src/experiments/BirdSet_common_test_8class/test_protocol.json`。

以下数值均为百分比。`FP32→INT8`只在同一模型和同一评估范围内比较；历史单种子
结果与严格三种子均值分别列示，不混合排名。

### Xeno-canto FP32基准与零样本选择

| 特征 | 输入 | Xeno Macro-F1 | DB3V Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 |
|---|---:|---:|---:|---:|---:|
| MFCC | 32×13 | 49.62% | 50.47% | 16.99% | 9.74% |
| **LogMel** | 32×40 | **63.56%** | **67.62%** | 17.78% | **9.88%** |
| PCEN | 32×40 | 52.69% | 59.93% | **19.52%** | 8.04% |

正式零样本基准保持为
`Xeno-canto → LogMel → DS-CNN`。LogMel同时取得最高Xeno-canto、DB3V和
BirdSet纯单物种Macro-F1；PCEN仅在BirdSet多标签Top-1上更高。

### 零样本严格INT8

本表使用统一的BirdSet公共test和完整10,658条DB3V录音。

| 特征 | Xeno Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---:|---:|---:|---:|
| MFCC | 49.62%→48.77% | 16.99%→15.36% | 9.74%→9.40% | 50.47%→51.25% |
| **LogMel** | **63.56%→59.01%** | 17.78%→18.42% | **9.88%→10.22%** | **67.62%→65.94%** |
| PCEN | 52.69%→35.07% | **19.52%→21.40%** | 8.04%→9.20% | 59.93%→50.24% |

MFCC量化损失最小但绝对精度偏低；LogMel量化后仍是最佳通用零样本模型；PCEN的
Xeno Macro-F1下降17.62个百分点，不能直接沿用其FP32结论。

### DB3V严格三种子FP32与INT8

严格实验使用`head_only`、`bn_head`、`bn_head_replay`和`full`四种策略，在
三种特征、三个shot和随机种子42/123/2026上共完成108次训练。下表为按预设平均
适配分数选中的9条链路及其27个严格INT8模型。BirdSet使用统一公共test，
DB3V使用10,197条共同held-out；数值均为三种子均值±样本标准差。

| 特征 | shot/策略 | Xeno Macro-F1（FP32→INT8） | BirdSet Top-1（FP32→INT8） | BirdSet纯单物种Macro-F1（FP32→INT8） | DB3V Macro-F1（FP32→INT8） |
|---|---|---:|---:|---:|---:|
| MFCC | 5/head_only | 47.79±0.60→46.40±0.89 | 12.87±2.22→14.55±2.43 | 8.37±1.04→8.99±0.97 | 53.94±0.46→53.74±0.47 |
| MFCC | 10/head_only | 48.43±1.35→47.53±1.83 | 14.18±2.94→13.51±2.89 | 8.82±1.12→8.57±1.18 | 53.74±2.07→53.09±1.54 |
| MFCC | 20/bn_head_replay | 49.64±0.35→47.83±1.88 | 16.50±0.81→16.35±1.47 | 10.42±0.24→9.66±0.41 | 56.64±1.87→56.25±1.98 |
| LogMel | 5/head_only | 62.33±0.76→52.21±0.80 | 19.12±1.12→23.37±6.94 | 10.34±0.30→14.80±1.74 | 67.72±0.12→64.05±1.79 |
| **LogMel** | **10/head_only** | **61.20±0.81→57.54±0.52** | **20.72±1.26→27.46±3.71** | **10.72±0.25→14.87±0.84** | **67.84±0.05→68.14±0.17** |
| LogMel | 20/bn_head_replay | 60.95±0.49→**58.76±0.75** | 20.29±0.07→26.09±1.46 | 10.27±0.07→14.22±0.91 | 66.97±0.25→66.79±0.23 |
| PCEN | 5/bn_head_replay | 62.58±3.37→31.96±1.37 | 16.29±2.02→19.11±4.35 | 12.32±2.29→8.22±0.35 | 67.64±2.64→46.53±0.63 |
| PCEN | 10/head_only | 58.01±2.65→37.09±0.64 | 12.46±5.61→21.96±9.50 | 8.41±0.16→13.05±3.34 | 65.27±3.39→52.05±1.57 |
| PCEN | 20/full | **63.66±0.44**→37.69±1.61 | 11.04±0.31→**34.05±2.44** | 9.99±0.19→**15.80±1.17** | **70.50±0.40**→52.52±0.79 |

当前DB3V严格INT8部署推荐为
`Xeno-canto → LogMel → DS-CNN → DB3V 10-shot head_only → INT8`：
DB3V Macro-F1为68.14%±0.17%，相对FP32提高0.30±0.18个百分点；Xeno
Macro-F1为57.54%±0.52%。LogMel 20-shot `bn_head_replay`的Xeno保留更高，
但DB3V绝对精度较低。PCEN 20-shot `full`仍是FP32最优，但量化后严重退化。

### BirdSet严格三种子FP32与INT8

BirdSet严格实验同样完成108次FP32训练，并量化9条选中策略的全部三个种子，
共27个严格INT8模型。以下为三种子均值±样本标准差。

| 特征 | shot/策略 | Xeno Macro-F1（FP32→INT8） | BirdSet Top-1（FP32→INT8） | BirdSet纯单物种Macro-F1（FP32→INT8） | DB3V Macro-F1（FP32→INT8） |
|---|---|---:|---:|---:|---:|
| MFCC | 5/bn_head_replay | 48.41±0.44→48.39±1.59 | 20.06±0.57→20.15±0.82 | 13.32±0.56→14.45±1.02 | 48.45±0.47→48.26±0.32 |
| MFCC | 10/bn_head_replay | 47.89±0.61→47.81±0.71 | 19.88±0.29→20.69±0.57 | 13.44±0.25→15.20±0.43 | 48.45±0.23→48.07±0.37 |
| **MFCC** | **20/bn_head_replay** | 48.53±0.40→47.72±0.95 | 23.89±0.44→**24.12±1.43** | 17.57±0.49→**18.40±1.35** | 49.23±0.48→48.70±0.45 |
| LogMel | 5/bn_head_replay | 56.10±0.63→53.14±1.57 | 20.31±2.17→20.22±5.03 | 13.32±2.35→13.20±1.49 | 59.47±1.52→57.85±1.07 |
| LogMel | 10/bn_head_replay | 56.11±1.31→54.40±1.88 | 20.39±2.02→23.09±4.19 | 13.66±2.57→14.19±0.66 | 59.33±1.67→57.84±2.06 |
| **LogMel** | **20/head_only** | **57.81±1.24→55.62±2.18** | 18.66±0.91→21.18±1.76 | 10.62±0.75→12.21±1.24 | **63.73±1.05→63.32±1.65** |
| PCEN | 5/bn_head | 55.27±1.34→17.97±2.16 | 36.11±1.11→19.78±0.09 | 27.11±0.40→4.25±0.08 | 61.41±1.73→27.80±4.79 |
| PCEN | 10/bn_head | 54.76±1.29→19.03±1.08 | **37.56±2.08**→19.84±0.04 | **28.06±1.60**→4.34±0.11 | 60.56±1.75→27.62±3.52 |
| PCEN | 20/bn_head | 57.85±0.06→16.85±1.33 | 29.68±0.41→19.28±0.18 | 23.42±0.18→4.18±0.05 | 63.67±0.13→28.53±0.60 |

FP32 BirdSet目标域最优为PCEN 10-shot `bn_head`；严格INT8后，BirdSet专项指标
最高且量化稳定的是MFCC 20-shot `bn_head_replay`，同时兼顾Xeno、BirdSet和DB3V
绝对精度的链路为LogMel 20-shot `head_only`。三个PCEN严格链路量化后均不可用，
后续必须改用量化感知训练并重新执行全部held-out评估。

### 当前推荐与未闭环项

| 使用场景 | 当前推荐链路 | 关键结果 |
|---|---|---|
| 通用FP32零样本 | Xeno→LogMel→DS-CNN | Xeno 63.56%，DB3V 67.62% |
| 通用INT8零样本 | Xeno→LogMel→DS-CNN→INT8 | Xeno 59.01%，DB3V 65.94% |
| DB3V严格FP32 | PCEN 20-shot/full | DB3V 70.50%±0.40%，Xeno 63.66%±0.44% |
| DB3V严格INT8 | LogMel 10-shot/head_only | DB3V 68.14%±0.17%，Xeno 57.54%±0.52% |
| BirdSet严格FP32 | PCEN 10-shot/bn_head | Top-1 37.56%±2.08%，单物种F1 28.06%±1.60% |
| BirdSet专项稳定INT8 | MFCC 20-shot/bn_head_replay | Top-1 24.12%，单物种F1 18.40% |
| BirdSet跨域均衡INT8 | LogMel 20-shot/head_only | Xeno 55.62%，BirdSet Top-1 21.18%，DB3V 63.32% |

当前标准量化工件包括3个零样本、27个DB3V严格多种子和27个BirdSet严格多种子
模型，合计57个TFLite模型。所有模型均为int8输入/输出、0个浮点张量、无
`QUANTIZE`/`DEQUANTIZE`算子，大小约49,968–49,984 bytes。

本节汇总来源：

- Xeno基准与DB3V零样本：`src/experiments/Feature_comparison_8class/comparison_summary.csv`
- BirdSet统一零样本：`src/experiments/Feature_comparison_8class/birdset_common_20shot_heldout_summary.csv`
- BirdSet统一测试协议：`src/experiments/BirdSet_common_test_8class/test_protocol.json`
- DB3V严格三种子：`src/experiments/DB3V_fewshot_ablation_multiseed_8class/aggregate.csv`
- DB3V严格多种子INT8：`src/experiments/DB3V_strict_INT8_quantization_8class/aggregate.csv`
- BirdSet严格三种子FP32：`src/experiments/BirdSet_fewshot_ablation_multiseed_8class/aggregate.csv`
- BirdSet严格多种子INT8：`src/experiments/BirdSet_strict_INT8_quantization_8class/aggregate.csv`
- 零样本严格INT8：`src/experiments/ZeroShot_strict_INT8_quantization_8class/summary.csv`
- 已淘汰链路的必要结论与来源指纹：`src/experiments/legacy_single_seed_reference.md`

后续章节保留完整实验协议、候选策略、数据划分、逐粒度指标、复现命令及原始报告路径。

## 当前实验角色与完整链路

Xeno-canto 是唯一基准模型训练集。BirdSet SSW 和 DB3V 不再用于从零训练基准
模型，而是并行承担“独立测试”和“小样本增强”两种外部角色：

- **Xeno-canto**：唯一训练/内部验证来源，用于选择特征、模型和早停轮次。
- **BirdSet SSW held-out test**：测试跨数据集、真实多鸟声景和背景噪声泛化。
- **BirdSet SSW support**：已完成 grouped 5/10/20-shot 多标签声景适配，不参与零样本测试。
- **DB3V held-out test**：测试三个特定地区的单标签泛化。
- **DB3V support**：按“地区×类别”抽取；已完成5/10/20-shot地区小样本适配。

support 与 held-out test 在原始录音级完全不重叠。任何小样本微调都必须先保存
零样本报告，并且只能使用support。DB3V使用support内部录音级Macro-F1和
Xeno-canto保留率联合选型；BirdSet使用support内部片段Top-1任意目标命中率和
Xeno-canto保留率联合选型。两个held-out均不用于选择策略、轮次或学习率。

### 为什么BirdSet和DB3V不作为基准训练集

这里的“不能”是实验方法上的角色约束，并不是说两个数据集在技术上无法训练。
如果用它们从头训练模型，它们就会失去当前实验中最重要的独立外部参照作用，
并且无法回答预先设定的跨数据集与跨地区泛化问题。

#### BirdSet SSW

BirdSet SSW 不适合作为本项目八类单标签基准训练集，原因如下：

1. **数据角色不同。** 当前使用的是官方 `test_5s` 多标签声景划分，其设计目标
   是评估真实声景分类，而不是作为本项目的焦点录音训练集。用它训练后就不能再
   将同一批声景作为独立跨数据集测试。
2. **任务定义不同。** 一个五秒片段可以同时包含多个鸟种，需要multi-hot标签和
   sigmoid输出；本项目基准任务是一条焦点录音对应一个主类别的八类softmax。
   从BirdSet训练会同时改变数据集和任务，无法与现有基准作单变量比较。
3. **八类覆盖严重失衡。** 19,678个目标片段来自211条长录音，各类支持差异很大；
   `Setophaga_ruticilla`只有两个片段且来自同一条长录音，无法同时构造无泄漏的
   八类训练集和八类验证集。
4. **一秒窗口存在弱标签噪声。** SSW标签属于完整五秒片段。将标签复制到五个
   一秒窗口时，部分窗口可能没有对应鸟鸣，不适合作为当前一秒单标签基准的直接
   监督信号。
5. **独立来源数量不足。** 98,390个一秒窗口并不等于98,390个独立样本；它们来自
   19,678个相邻片段和211条长录音，空间、设备和背景相关性较强。

因此BirdSet保留为跨数据集、多鸟声景和背景噪声泛化测试；隔离出的support只用于
从Xeno-canto权重开始的小样本适配。下文从BirdSet零初始化的模型是一次性反事实
诊断模型，不纳入正式基准或后续模型选择。

#### DB3V

DB3V不作为基准训练集主要是实验目标和泄漏控制问题：

1. **它是地区泛化的目标域。** 项目需要用DB3V三个地区回答模型在未参与训练的
   地区能否泛化。如果从DB3V训练，测试就退化为同域识别，不能再称为地区泛化。
2. **没有可同时保留的官方基准训练/验证/地区外测协议。** 当前数据按三个地区
   组织；任意从中划分训练集都会让模型接触目标地区的设备、背景和类别先验。
3. **目标域泄漏会高估结果。** 即使训练和测试文件不重复，来自同一地区的录音仍
   共享声学环境。使用DB3V选择特征、阈值或早停轮次，会使区域测试结果带有调参
   偏差。
4. **地区样本量不均衡。** 三个地区分别有5,251、752和4,655条录音，直接混合作为
   基准训练集容易被地区1和地区3主导，并掩盖地区2的不确定性。

因此DB3V held-out只用于特定地区泛化。项目先保留120条support/10,538条测试录音
的原始5-shot报告，再补充嵌套的120/239/461条support实验；跨shot比较统一使用
20-shot support之外的10,197条共同held-out录音，并与相同测试集上的零样本结果
比较。

#### Xeno-canto为何满足基准条件

相比之下，当前Xeno-canto子集提供每类300条、总计2,400条焦点录音，八类均衡；
可按完整录音会话稳定划分为每类240条训练和60条验证；标签任务与八类softmax
一致；并且来源ID、音频哈希和DB3V来源已经隔离。它因此能够同时满足类别覆盖、
任务一致、无泄漏模型选择和保留两个外部测试域四项基准要求。

### 三条基准模型链路

| 排名 | 唯一训练集 | 特征 | 输入 | Xeno验证 Macro-F1 | DB3V完整集 Accuracy | DB3V完整集 Macro-F1 | DB3V Top-3 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Xeno-canto | Log-Mel | 32×40 | **63.56%** | **70.71%** | **67.62%** | **91.13%** |
| 2 | Xeno-canto | PCEN | 32×40 | 52.69% | 63.57% | 59.93% | 85.97% |
| 3 | Xeno-canto | MFCC | 32×13 | 49.62% | 56.97% | 50.47% | 84.79% |

Log-Mel 是当前正式推荐链路；PCEN 为噪声归一化研究链路；MFCC 输入最小，保留
为资源受限 TinyML 兼容链路。三者都只能由 Xeno-canto 训练。

### 基准数据集结果与模型选择依据

Xeno-canto 既是唯一训练数据来源，也是唯一允许用于模型选择的数据域。2,400条
录音按完整录音会话固定划分为1,920条训练录音和480条内部验证录音，每类分别为
240/60条；三种特征使用相同录音、相同切片位置、相同DS-CNN、随机种子42、
30轮上限和8轮早停。

| Xeno-canto基准链路 | 验证录音 | Accuracy | Balanced Accuracy | Macro-F1 | 选择结论 |
|---|---:|---:|---:|---:|---|
| Log-Mel + DS-CNN | 480 | **62.29%** | **62.29%** | **63.56%** | 正式推荐模型 |
| PCEN + DS-CNN | 480 | 52.92% | 52.92% | 52.69% | 次选研究模型 |
| MFCC + DS-CNN | 480 | 50.00% | 50.00% | 49.62% | TinyML兼容模型 |

模型选择的唯一主要指标是 **Xeno-canto内部验证录音级 Macro-F1**。每轮先平均
同一验证录音所有一秒窗口的概率，再计算Macro-F1；EarlyStopping和最佳权重恢复
也只监控该指标。因此Log-Mel在查看外部held-out结果之前就已经被选为正式推荐
模型，选择差距如下：

- Log-Mel相对PCEN提高10.87个Macro-F1百分点；
- Log-Mel相对MFCC提高13.94个Macro-F1百分点；
- Accuracy和Balanced Accuracy也由Log-Mel同时取得最高值，没有指标冲突。

BirdSet和DB3V结果不得用于选择特征、早停轮次、阈值或超参数。它们只回答模型
选定后的两个外部问题：是否能跨到真实多标签声景，以及是否能跨到特定地区。
后续few-shot微调也必须保留本表作为原始基准，并报告Xeno验证变化以衡量灾难性
遗忘。

基准结果来源：

- `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.validation.json`
- `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.history.json`
- `src/experiments/Feature_comparison_8class/comparison_summary.csv`

### 外部零样本测试与few-shot划分

当前以最佳 `Xeno-canto + Log-Mel + DS-CNN` 模型作为外部测试基准：

| 外部数据 | 测试目的 | support | held-out test | held-out主要结果 |
|---|---|---:|---:|---|
| BirdSet SSW | 跨数据集、多标签真实声景泛化 | 最大support为14条长录音、1,413个片段 | 统一公共test：197条长录音、18,265个片段 | 片段Top-1任意目标17.78%，Top-3 40.90%；纯单物种Macro-F1 9.88% |
| DB3V | 三个特定地区的单标签泛化 | 120条录音 | 10,538条录音 | Accuracy 70.71%，Balanced Accuracy 70.45%，Macro-F1 67.53%，Top-3 91.09% |

BirdSet support按5/10/20-shot嵌套构建且必须保持长录音完整，实际分别包含
10/11/14条长录音；统一测试排除最大的20-shot support。稀有的
`Setophaga_ruticilla` 只有同一长录音中的两个片段，无法达到任一shot目标，且统一
test中没有该类正片段，所以纯单物种Macro-F1只覆盖有支持的7类。DB3V嵌套support
实际分别为120、239和461条录音；横向比较使用共同的10,197条held-out录音。

这两个零样本结果是小样本增强的固定前测。DB3V与BirdSet support适配均已完成；
每次微调后均按以下三个方向复测：

1. 对应 held-out test，衡量目标数据域提升；
2. Xeno-canto 内部验证，检查是否发生灾难性遗忘；
3. 另一个外部 held-out test，检查增强是否损害跨域泛化。

结果与划分来源：

- `src/dataset_processing/output/LogMel_BirdSet_external_split_8class/split_manifest.json`
- `src/dataset_processing/output/LogMel_DB3V_external_split_8class/split_manifest.json`
- `src/experiments/Feature_comparison_8class/LogMel/BirdSet_SSW_heldout_evaluation.json`
- `src/experiments/Feature_comparison_8class/LogMel/DB3V_heldout_evaluation.json`
- `src/experiments/Feature_comparison_8class/LogMel/DB3V_heldout_evaluation_summary.csv`

### 候选基准数据集反事实诊断（不参与正式选型）

为验证前述数据角色判断，项目另外执行了两组从零训练实验：分别把BirdSet SSW和
DB3V当作候选基准训练来源，再到另一个完全独立的数据集测试。该实验只回答“如果
强行将其作为基准会发生什么”，其内部验证结果不参与正式模型选择，也不会改变
Xeno-canto的唯一正式基准地位。

#### BirdSet作为候选基准

BirdSet按原始长录音固定拆分为169条训练录音（16,102个五秒片段）和42条验证录音
（3,576个片段），使用multi-hot、sigmoid和二元交叉熵。由于
`Setophaga_ruticilla`只有同一长录音中的两个片段，该录音只能保留在训练集，
所以内部Macro-F1只覆盖验证集中有正样本的7类。外部测试使用未参与训练和选型的
完整DB3V 10,658条录音。

| 特征 | BirdSet验证Top-1任意目标 | BirdSet验证7类Macro-F1 | DB3V录音Accuracy | DB3V录音Balanced Accuracy | DB3V录音Macro-F1 | 同特征Xeno→DB3V Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| MFCC | 31.46% | **14.86%** | **17.72%** | 13.97% | 9.57% | 50.47% |
| Log-Mel | **46.59%** | 14.77% | 15.73% | **17.44%** | **9.97%** | **67.62%** |
| PCEN | 17.81% | 0.00% | 11.04% | 9.64% | 4.52% | 59.93% |

PCEN的0.00%表示在固定0.5 sigmoid阈值下没有形成有效的验证集正类预测，不表示
argmax完全随机；其Top-1任意目标仍为17.81%。更关键的是，三种BirdSet训练模型
跨到DB3V后的Macro-F1仅4.52%–9.97%，比相同特征的Xeno-canto模型低
40.90–57.65个百分点。结果同时暴露了七类验证、任务定义不一致、长录音来源少和
跨域退化，因此BirdSet不能承担当前八类单标签正式基准。

#### DB3V作为候选基准

DB3V按“地区×类别”在完整八秒录音级拆分：8,526条训练、2,132条验证；三个地区
都同时出现在两部分中。内部指标因此只是**同地区留出识别**，不是未见地区泛化。
跨数据集测试使用完整BirdSet SSW目标子集，共211条长录音、19,678个五秒片段；
“纯单物种Macro-F1”在10,230个全局单物种片段上计算。

| 特征 | DB3V验证Accuracy | DB3V验证Balanced Accuracy | DB3V验证Macro-F1 | BirdSet Top-1任意目标 | BirdSet Top-3任意目标 | BirdSet纯单物种Macro-F1 | 同特征Xeno→BirdSet纯单物种Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFCC | 88.65% | 90.16% | 88.44% | 15.19% | 51.98% | 11.32% | 9.77% |
| Log-Mel | **91.56%** | **91.69%** | **90.33%** | **19.79%** | **53.45%** | **15.81%** | **10.12%** |
| PCEN | 84.85% | 82.80% | 82.17% | 16.43% | 36.96% | 14.00% | 8.08% |

DB3V Log-Mel在同地区验证上达到90.33% Macro-F1，并在同口径BirdSet全集上略高于
同特征Xeno模型。这说明DB3V可提供有价值的目标域监督，适合后续地区support微调；
但高内部得分主要回答“见过三个目标地区后能否识别同地区新文件”。一旦用DB3V
训练，就失去DB3V作为独立地区泛化测试的能力，也无法再回答模型是否泛化到未参与
训练的地区。因此该结果支持“DB3V适合目标域适配、不适合作为正式基准”的定位。

综合两组诊断：BirdSet训练后跨DB3V严重退化；DB3V训练虽能改善部分BirdSet指标，
却消耗了关键地区测试域。只有Xeno-canto同时保留八类单标签一致性、无泄漏内部
选型和两个独立外部数据域，所以正式基准及推荐链路保持不变。

诊断实验来源：

- BirdSet固定切分与协议：`src/dataset_processing/birdset_baseline_validation_recordings.json`、`src/experiments/BirdSet_feature_comparison_8class/experiment_protocol.json`
- BirdSet训练结果：`src/experiments/BirdSet_feature_comparison_8class/comparison_summary.csv`、`src/experiments/BirdSet_baseline_8class/`
- DB3V固定协议与结果：`src/experiments/DB3V_baseline_diagnostic_8class/experiment_protocol.json`、`src/experiments/DB3V_baseline_diagnostic_8class/comparison_summary.csv`
- DB3V各特征原始报告：`src/experiments/DB3V_baseline_diagnostic_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.validation.json` 与 `BirdSet_SSW_evaluation.json`
- 同口径Xeno参考：`src/experiments/Feature_comparison_8class/birdset_full_summary.csv` 与 `{MFCC,LogMel,PCEN}/BirdSet_SSW_full_evaluation.json`
- 数据集原始出处见上文“数据来源、引用与许可”；诊断协议文件也保存了BirdSet、SSW和DB3V的官方链接。

## 当前 DS-CNN 基准模型

当前兼容基准模型为 MFCC `DS_CNN_Model`；最高精度推荐模型是
`Feature_comparison_8class/LogMel/DS_CNN_Model.h5`：

- 推荐模型输入：一秒音频对应的 `(32, 40, 1)` Log-Mel；
- 兼容模型输入：一秒音频对应的 `(32, 13, 1)` MFCC；
- 输出：八类 softmax；
- 训练数据：Xeno-canto；
- 独立测试：BirdSet SSW 与 DB3V；两者另有完全隔离的小样本 support；
- 推荐模型文件：`src/experiments/Feature_comparison_8class/LogMel/DS_CNN_Model.h5`；
- 兼容模型文件：`src/experiments/TinyML_model_8class/DS_CNN_Model.h5`。

内部留出验证以完整录音为评估单位：

| 数据与粒度 | 数量 | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Xeno-canto内部验证，录音级 | 480条录音 | 50.00% | 50.00% | **49.62%** |

结果来源：`src/experiments/TinyML_model_8class/DS_CNN_Model.validation.json`。

## 八分类特征处理方法对比

`src/dataset_processing` 中与本实验有关的四个入口已整理为两类职责：

1. `data_sugment_MFCC.py` 是共享实现，可生成 MFCC、Log-Mel 或 PCEN；
2. `data_sugment_logMel.py` 和 `data_sugment_PCEN.py` 是调用共享实现的便捷入口；
3. `data_sugment_MFCC_DB3V.py` 是 DB3V 区域数据适配器，不是第四种声学特征，可通过 `--feature` 对三种特征执行一致的外部测试预处理。

公平复测固定使用相同的八类标签顺序、2,400 条 Xeno-canto 原始录音、1,920/480 条录音级训练/验证划分、每条录音最多 8 个相同位置的一秒切片、DS-CNN、随机种子 42、最多 30 轮和 8 轮早停。DB3V 仅用于最终测试，不参与模型选择。

| 特征 | 输入形状 | Xeno-canto 验证 Accuracy | 验证 Macro-F1 | DB3V 切片 Accuracy | DB3V 切片 Macro-F1 | DB3V 录音 Accuracy | DB3V 录音 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFCC | 32×13 | 50.00% | 49.62% | 39.80% | 35.48% | 56.97% | 50.47% |
| Log-Mel | 32×40 | **62.29%** | **63.56%** | **44.67%** | **42.84%** | **70.71%** | **67.62%** |
| PCEN | 32×40 | 52.92% | 52.69% | 40.03% | 37.71% | 63.57% | 59.93% |

本轮中 Log-Mel 最优：相对 MFCC，内部验证 Macro-F1 提高 13.94 个百分点，DB3V 录音级 Macro-F1 提高 17.15 个百分点；PCEN 也分别提高 3.07 和 9.46 个百分点。该结论只适用于本次固定 DS-CNN 和当前数据协议，还不能直接外推到其他模型或噪声条件。

复现实验时分别运行三个 Xeno-canto 入口，并为 DB3V 入口传入匹配的 `--feature`。训练和测试命令与“推荐工作流”相同，只需对应修改数据集和模型目录。

结果来源：

- `src/experiments/Feature_comparison_8class/comparison_summary.csv`
- `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DS_CNN_Model.validation.json`
- `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DB3V_evaluation.json`
- `src/experiments/Feature_comparison_8class/{MFCC,LogMel,PCEN}/DB3V_evaluation_summary.csv`

## DB3V 独立区域测试结果

DB3V 每条八秒录音产生八个一秒 MFCC 窗口。“切片级”独立评价每个窗口；“录音级”先平均同一录音的八个 softmax 向量再分类。

### DB3V汇总

| 粒度 | 数量 | Accuracy | Balanced Accuracy | Macro-F1 | Top-3 |
|---|---:|---:|---:|---:|---:|
| 一秒切片级 | 85,264个切片 | 39.80% | 37.34% | 35.48% | 67.69% |
| 八秒录音级 | 10,658条录音 | **56.97%** | **52.14%** | **50.47%** | **84.79%** |

### DB3V分地区录音级结果

| 地区 | 录音数 | Accuracy | Balanced Accuracy | Macro-F1 | Top-3 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5,251 | 56.54% | 50.09% | 48.19% | 85.36% |
| 2 | 752 | **71.28%** | **64.05%** | **57.38%** | **89.23%** |
| 3 | 4,655 | 55.15% | 50.53% | 47.99% | 83.44% |

地区2表现最高但样本量最少。时间聚合相对切片级使汇总 Accuracy 提高约 17.2 个百分点、Macro-F1 提高约 15.0 个百分点。

结果来源：

- `src/experiments/TinyML_model_8class/DB3V_evaluation.json`
- `src/experiments/TinyML_model_8class/DB3V_evaluation_summary.csv`

## BirdSet与DB3V外部测试及小样本协议

BirdSet 和 DB3V 均不用于训练基准模型。推荐使用与最佳 Xeno-canto 模型匹配的
Log-Mel 特征，先建立 support/test 隔离划分，再保存零样本报告。

```powershell
# BirdSet：按原始长录音拆分 support 与 held-out test
& .\.venv\Scripts\python.exe src\dataset_processing\prepare_external_fewshot.py `
  birdset `
  --source-dir src\dataset_processing\output\LogMel_dataset_BirdSet_SSW_8class `
  --output-dir src\dataset_processing\output\LogMel_BirdSet_external_split_8class `
  --shots 5

# DB3V：按地区和类别拆分 support 与 held-out test
& .\.venv\Scripts\python.exe src\dataset_processing\prepare_external_fewshot.py `
  db3v `
  --source-dir src\dataset_processing\output\LogMel_dataset_DB3V_8class `
  --output-dir src\dataset_processing\output\LogMel_DB3V_external_split_8class `
  --shots 5

# BirdSet held-out：跨数据集、多标签声景泛化
& .\.venv\Scripts\python.exe "src\experiments\evaluate_birdset_ssw.py" `
  --models DS_CNN_Model `
  --dataset-dir src\dataset_processing\output\LogMel_BirdSet_external_split_8class\test `
  --model-dir "src\experiments\Feature_comparison_8class\LogMel" `
  --output "src\experiments\Feature_comparison_8class\LogMel\BirdSet_SSW_heldout_evaluation.json"

# DB3V held-out：三个特定地区泛化
& .\.venv\Scripts\python.exe "src\experiments\evaluate_db3v.py" `
  --models DS_CNN_Model `
  --dataset-dir src\dataset_processing\output\LogMel_DB3V_external_split_8class `
  --model-dir "src\experiments\Feature_comparison_8class\LogMel" `
  --output "src\experiments\Feature_comparison_8class\LogMel\DB3V_heldout_evaluation.json"
```

### 小样本增强实验约束

- support 只能用于从 Xeno-canto 基准权重开始的微调，不能从头训练新基准。
- BirdSet support 保留 multi-hot 标签；DB3V support 使用单标签。
- held-out test 永远不参与梯度更新、阈值选择、早停或超参数选择。
- 必须报告 zero-shot 与 few-shot 的差值，而不是只报告微调后最高值。
- BirdSet 和 DB3V 应分别微调，才能区分“跨数据集适配”与“特定地区适配”。
- 每次微调后同时复测 Xeno-canto、BirdSet held-out 和 DB3V held-out。

### 已淘汰单种子链路说明

原DB3V单种子5/10/20-shot、BirdSet单种子grouped 5-shot以及对应12条小样本INT8
链路已由当前严格多随机种子协议替代。旧模型、逐轮历史、旧策略入口和失效复现命令
已从当前树移除；对现有方法分析仍有价值的关键结论、局限和原汇总SHA-256统一保存
在`src/experiments/legacy_single_seed_reference.md`。这些历史数值不参与当前排名、模型
选择或部署推荐。
### DB3V严格微调策略与多随机种子补充实验

为补齐旧实验中“`head`实际训练两个Dense层”“`all`仍冻结BatchNorm”以及只有单一
随机种子的问题，新增四个定义无歧义的严格策略。模型总参数为30,056：

| 策略 | 可训练范围 | 可训练参数 | BatchNorm可训练 | Xeno replay |
|---|---|---:|---|---|
| `head_only` | 仅最终8类softmax Dense层 | 1,032 | 否 | 否 |
| `bn_head` | 全部BatchNorm与最终softmax层 | 1,480 | 是 | 否 |
| `bn_head_replay` | 与`bn_head`相同 | 1,480 | 是 | 是，Xeno训练切片与DB3V support切片1:1 |
| `full` | 全部层，包括BatchNorm | 29,608 | 是 | 否 |

实验固定使用原来的嵌套DB3V support（5/10/20-shot实际为120/239/461条录音）和
10,197条共同held-out。随机种子42、123、2026分别改变support内部训练/验证划分、
Replay抽样、批次顺序和TensorFlow随机操作；外部support身份和最终held-out身份保持
固定，因此本表衡量训练与内部选型随机性，不包含重新抽取外部support带来的方差。
Replay只从Xeno-canto基准训练划分按类别均衡抽样，不读取Xeno验证、DB3V held-out或
BirdSet。每个结果均为3个随机种子的均值±样本标准差（`ddof=1`）。

每个“特征×shot”先按三个seed的平均适配分数选策略；适配分数仍为support内部验证
录音级Macro-F1乘以截断到1的Xeno Macro-F1保留率，DB3V held-out不参与选型：

| 特征 | shot | 选中严格策略 | Xeno Macro-F1 | Xeno变化 | DB3V共同held-out Macro-F1 |
|---|---:|---|---:|---:|---:|
| MFCC | 5 | `head_only` | 47.79% ± 0.60% | -1.82 ± 0.60 pp | 53.94% ± 0.46% |
| MFCC | 10 | `head_only` | 48.43% ± 1.35% | -1.19 ± 1.35 pp | 53.74% ± 2.07% |
| MFCC | 20 | `bn_head_replay` | 49.64% ± 0.35% | +0.03 ± 0.35 pp | 56.64% ± 1.87% |
| LogMel | 5 | `head_only` | 62.33% ± 0.76% | -1.23 ± 0.76 pp | 67.72% ± 0.12% |
| LogMel | 10 | `head_only` | 61.20% ± 0.81% | -2.36 ± 0.81 pp | 67.84% ± 0.05% |
| LogMel | 20 | `bn_head_replay` | 60.95% ± 0.49% | -2.61 ± 0.49 pp | 66.97% ± 0.25% |
| PCEN | 5 | `bn_head_replay` | 62.58% ± 3.37% | +9.89 ± 3.37 pp | 67.64% ± 2.64% |
| PCEN | 10 | `head_only` | 58.01% ± 2.65% | +5.32 ± 2.65 pp | 65.27% ± 3.39% |
| **PCEN** | **20** | **`full`** | **63.66% ± 0.44%** | **+10.98 ± 0.44 pp** | **70.50% ± 0.40%** |

当前正式零样本基准的LogMel链路上，四个严格策略的完整结果如下；单元格为
“DB3V Macro-F1 / Xeno Macro-F1变化”，均以百分点表示：

| shot | `head_only` | `bn_head` | `bn_head_replay` | `full` |
|---:|---:|---:|---:|---:|
| 5 | **67.72±0.12 / -1.23±0.76** | 67.28±0.15 / -1.27±0.74 | 67.38±0.08 / -1.11±0.30 | 67.34±0.17 / **-0.89±0.05** |
| 10 | **67.84±0.05 / -2.36±0.81** | 66.57±0.33 / -3.19±0.45 | 67.34±0.40 / **-1.59±0.63** | 67.00±0.38 / -3.73±1.43 |
| 20 | **67.99±0.25 / -1.83±0.55** | 66.62±0.32 / -3.38±0.69 | 66.97±0.25 / -2.61±0.49 | 67.53±1.09 / -4.39±1.29 |

严格实验的主要结论是：

- `PCEN + 20-shot + full` 是按预先定义适配分数选中且DB3V结果最高的严格策略，
  DB3V Macro-F1为70.50%±0.40%，同时Xeno Macro-F1相对基模型提高
  10.98±0.44个百分点。
- `PCEN + 20-shot + bn_head_replay` 的DB3V Macro-F1为69.32%±0.45%，但Xeno提升
  更大（+12.63±0.13个百分点），说明Replay确实改善基准域保留；其平均适配分数
  略低于`full`，因此未被选中。
- LogMel对严格全量微调较敏感；在共同held-out上，三个shot均由`head_only`取得
  最高DB3V均值。20-shot `full`的标准差增至1.09个百分点且Xeno下降4.39个百分点。
- 选中的9组策略现已完成全部三个seed的BirdSet统一公共test跨域复测和严格INT8量化；
  FP32最高的`PCEN + 20-shot + full`量化后DB3V Macro-F1降至52.52%±0.79%，
  当前DB3V严格INT8推荐改为`LogMel + 10-shot + head_only`。

完整36组聚合结果和108个逐seed结果是本节数值的直接来源：

- 训练与策略定义：`src/experiments/fine_tune_db3v.py`
- 批量执行与统计实现：`src/experiments/run_db3v_ablation_multiseed.py`
- 实验协议：`src/experiments/DB3V_fewshot_ablation_multiseed_8class/experiment_protocol.json`
- 36组均值/标准差：`src/experiments/DB3V_fewshot_ablation_multiseed_8class/aggregate.csv`
- 108个逐seed索引：`src/experiments/DB3V_fewshot_ablation_multiseed_8class/runs.csv`
- 原始模型、逐轮历史、Xeno遗忘和DB3V报告：
  `src/experiments/DB3V_fewshot_ablation_multiseed_8class/{MFCC,LogMel,PCEN}/{5shot,10shot,20shot}/{head_only,bn_head,bn_head_replay,full}/seed_<seed>/`
- 选中链路的BirdSet FP32复测和严格INT8结果：
  `src/experiments/DB3V_strict_INT8_quantization_8class/`
- DB3V数据来源与许可：[Zenodo 11544734](https://doi.org/10.5281/zenodo.11544734)；
  固定划分协议位于被Git排除的
  `src/dataset_processing/output/<feature>_DB3V_external_split*_8class/split_manifest.json`。

从项目根目录可复现或重建汇总：

```powershell
& .\.venv\Scripts\python.exe src\experiments\run_db3v_ablation_multiseed.py
& .\.venv\Scripts\python.exe src\experiments\run_db3v_ablation_multiseed.py `
  --summarize-only
```

### BirdSet grouped 5/10/20-shot严格多种子结果

严格实验使用`Head-Only`、`BN+Head`、`BN+Head+Replay`和包含BatchNorm的
`Full Fine-Tuning`，覆盖三种特征、三个shot规模和随机种子42/123/2026，
共108个模型。每个seed使用不同的support内部录音级验证变体；策略和epoch只由
support验证Top-1与Xeno保留率的乘积选择。共同BirdSet held-out和完整DB3V在
选型结束后才读取。

shot表示每类期望正五秒片段数，但划分必须保留完整长录音。5/10/20-shot support
分别包含10/11/14条长录音和684/699/1,413个片段。`Setophaga_ruticilla`
全局只有2个正片段，因此三个规模分别短缺3/8/18，不复制样本。为保证直接可比，
三种shot统一在20-shot划分隔离出的197条长录音、18,265个片段上最终复测。

下表为每个特征和shot按三seed平均适配分数选中的策略；数值是均值±样本标准差。

| 特征 | shot | 选中策略 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | Xeno Macro-F1 | DB3V Macro-F1 |
|---|---:|---|---:|---:|---:|---:|
| MFCC | 5 | `bn_head_replay` | 20.06% ± 0.57% | 13.32% ± 0.56% | 48.41% ± 0.44% | 48.45% ± 0.47% |
| MFCC | 10 | `bn_head_replay` | 19.88% ± 0.29% | 13.44% ± 0.25% | 47.89% ± 0.61% | 48.45% ± 0.23% |
| MFCC | 20 | `bn_head_replay` | 23.89% ± 0.44% | 17.57% ± 0.49% | 48.53% ± 0.40% | 49.23% ± 0.48% |
| LogMel | 5 | `bn_head_replay` | 20.31% ± 2.17% | 13.32% ± 2.35% | 56.10% ± 0.63% | 59.47% ± 1.52% |
| LogMel | 10 | `bn_head_replay` | 20.39% ± 2.02% | 13.66% ± 2.57% | 56.11% ± 1.31% | 59.33% ± 1.67% |
| LogMel | 20 | `head_only` | 18.66% ± 0.91% | 10.62% ± 0.75% | 57.81% ± 1.24% | 63.73% ± 1.05% |
| PCEN | 5 | `bn_head` | 36.11% ± 1.11% | 27.11% ± 0.40% | 55.27% ± 1.34% | 61.41% ± 1.73% |
| **PCEN** | **10** | **`bn_head`** | **37.56% ± 2.08%** | **28.06% ± 1.60%** | 54.76% ± 1.29% | 60.56% ± 1.75% |
| PCEN | 20 | `bn_head` | 29.68% ± 0.41% | 23.42% ± 0.18% | **57.85% ± 0.06%** | **63.67% ± 0.13%** |

FP32下，PCEN 10-shot `BN+Head`取得最高BirdSet实际held-out Top-1和纯单物种
Macro-F1；PCEN 20-shot更偏向Xeno保留和DB3V跨域平衡。因为shot加入的是完整
长录音而非独立同分布片段，且support内部验证构成也随规模变化，结果不保证单调。

结果与来源：

- 执行与汇总：`src/experiments/run_birdset_ablation_multiseed.py`
- 完整协议和官方链接：
  `src/experiments/BirdSet_fewshot_ablation_multiseed_8class/experiment_protocol.json`
- 108个逐seed索引与36组聚合：
  `src/experiments/BirdSet_fewshot_ablation_multiseed_8class/{runs.csv,aggregate.csv}`
- 详细说明：
  `src/experiments/BirdSet_fewshot_ablation_multiseed_8class/README.md`
- BirdSet、SSW和DB3V的官方数据来源见上文“数据来源、许可与实验角色”。

## 严格INT8量化接口与精度测试

当前标准已完成57个FP32模型的训练后静态量化（PTQ）和同口径全量复测：

- 3条零样本基准：MFCC、LogMel、PCEN；
- 27个DB3V严格小样本模型：9组按三seed平均适配分数选中的策略×3个seed；
- 27个BirdSet严格小样本模型：9组按三seed平均适配分数选中的策略×3个seed。

量化结果不用于重新选择FP32策略。代表性数据固定为256个训练窗口：零样本只使用
Xeno-canto训练特征；小样本链路在Xeno训练特征之外只加入对应DB3V或BirdSet
support。任何held-out和Xeno验证数据都不参与校准。

### 输入接口与严格性

| 特征 | 特征提取输出 | TFLite输入 | 说明 |
|---|---|---|---|
| MFCC | `float32 [batch,32,13]` | `int8 [batch,32,13,1]` | 13维MFCC |
| LogMel | `float32 [batch,32,40]` | `int8 [batch,32,40,1]` | 40维Log-Mel |
| PCEN | `float32 [batch,32,40]` | `int8 [batch,32,40,1]` | 形状同LogMel，但语义和校准数据独立 |

每条链路从TFLite模型读取自己的input scale和zero-point，按
`q=clip(round(x/scale)+zero_point,-128,127)` 量化输入，再使用输出scale和
zero-point反量化后执行原有时间聚合。当前57个标准模型全部满足：

- 仅允许 `TFLITE_BUILTINS_INT8`；
- 输入、权重、激活和输出走整数路径，输入/输出均为int8；
- 浮点张量数为0，图内没有 `QUANTIZE`/`DEQUANTIZE`；
- int32偏置和乘加累积属于标准整数内核；
- 单模型大小为49,968–49,984 bytes。

### 零样本INT8结果

BirdSet使用统一公共test的197条长录音、18,265个片段；DB3V使用完整10,658条录音。

| 特征 | Xeno Macro-F1（FP32→INT8） | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---:|---:|---:|---:|
| MFCC | 49.62% → 48.77% (-0.85 pp) | 16.99% → 15.36% (-1.63 pp) | 9.74% → 9.40% (-0.34 pp) | 50.47% → 51.25% (+0.78 pp) |
| **LogMel** | **63.56% → 59.01% (-4.55 pp)** | **17.78% → 18.42% (+0.64 pp)** | **9.88% → 10.22% (+0.34 pp)** | **67.62% → 65.94% (-1.69 pp)** |
| PCEN | 52.69% → 35.07% (-17.62 pp) | 19.52% → 21.40% (+1.87 pp) | 8.04% → 9.20% (+1.16 pp) | 59.93% → 50.24% (-9.69 pp) |

MFCC量化最稳定，但LogMel量化后的Xeno和DB3V绝对Macro-F1仍最高，因此严格INT8
零样本基准推荐保持 `Xeno-canto → LogMel → DS-CNN`。PCEN不能直接沿用FP32结论。

### DB3V严格5/10/20-shot多种子INT8结果

DB3V统一使用20-shot support之外的10,197条共同held-out；BirdSet统一使用排除
BirdSet 20-shot support后的197条长录音公共test。每个“特征×shot”按三个seed的
平均适配分数选择一个严格策略，再量化该策略的全部三个seed，共27个模型。

| 特征 | shot/策略 | Xeno Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---|---:|---:|---:|---:|
| MFCC | 5 / `head_only` | 47.79±0.60 → 46.40±0.89 | 12.87±2.22 → 14.55±2.43 | 8.37±1.04 → 8.99±0.97 | 53.94±0.46 → 53.74±0.47 |
| MFCC | 10 / `head_only` | 48.43±1.35 → 47.53±1.83 | 14.18±2.94 → 13.51±2.89 | 8.82±1.12 → 8.57±1.18 | 53.74±2.07 → 53.09±1.54 |
| MFCC | 20 / `bn_head_replay` | 49.64±0.35 → 47.83±1.88 | 16.50±0.81 → 16.35±1.47 | 10.42±0.24 → 9.66±0.41 | 56.64±1.87 → 56.25±1.98 |
| LogMel | 5 / `head_only` | 62.33±0.76 → 52.21±0.80 | 19.12±1.12 → 23.37±6.94 | 10.34±0.30 → 14.80±1.74 | 67.72±0.12 → 64.05±1.79 |
| **LogMel** | **10 / `head_only`** | **61.20±0.81 → 57.54±0.52** | **20.72±1.26 → 27.46±3.71** | **10.72±0.25 → 14.87±0.84** | **67.84±0.05 → 68.14±0.17** |
| LogMel | 20 / `bn_head_replay` | 60.95±0.49 → **58.76±0.75** | 20.29±0.07 → 26.09±1.46 | 10.27±0.07 → 14.22±0.91 | 66.97±0.25 → 66.79±0.23 |
| PCEN | 5 / `bn_head_replay` | 62.58±3.37 → 31.96±1.37 | 16.29±2.02 → 19.11±4.35 | 12.32±2.29 → 8.22±0.35 | 67.64±2.64 → 46.53±0.63 |
| PCEN | 10 / `head_only` | 58.01±2.65 → 37.09±0.64 | 12.46±5.61 → 21.96±9.50 | 8.41±0.16 → 13.05±3.34 | 65.27±3.39 → 52.05±1.57 |
| PCEN | 20 / `full` | **63.66±0.44** → 37.69±1.61 | 11.04±0.31 → **34.05±2.44** | 9.99±0.19 → **15.80±1.17** | **70.50±0.40** → 52.52±0.79 |

严格INT8部署推荐为
`Xeno-canto → LogMel → DS-CNN → DB3V 10-shot head_only`：共同held-out
Macro-F1为68.14%±0.17%，Xeno Macro-F1为57.54%±0.52%。FP32最高的PCEN
20-shot `full`量化后DB3V Macro-F1降至52.52%±0.79%，当前PTQ不可部署。

结果来源：

- `src/experiments/DB3V_strict_INT8_quantization_8class/README.md`
- `src/experiments/DB3V_strict_INT8_quantization_8class/experiment_protocol.json`
- `src/experiments/DB3V_strict_INT8_quantization_8class/summary.csv`
- `src/experiments/DB3V_strict_INT8_quantization_8class/aggregate.csv`
- `src/experiments/DB3V_strict_INT8_quantization_8class/models/<chain_id>/`

### BirdSet严格5/10/20-shot多种子INT8结果

在前述严格FP32实验中，每个“特征×shot”只按三个seed的平均适配分数选择一个
策略，再量化该策略的全部三个seed，共27个模型。校准固定使用256个代表性窗口，
在Xeno训练特征与匹配shot的BirdSet support之间等额分配；所有held-out均不参与
校准。27个模型全部为int8输入/输出、0个浮点张量、0个
`QUANTIZE`/`DEQUANTIZE`算子。

| 特征 | shot/策略 | Xeno Macro-F1 | BirdSet Top-1 | BirdSet纯单物种Macro-F1 | DB3V Macro-F1 |
|---|---|---:|---:|---:|---:|
| MFCC | 5 / `bn_head_replay` | 48.41% → 48.39% | 20.06% → 20.15% | 13.32% → 14.45% | 48.45% → 48.26% |
| MFCC | 10 / `bn_head_replay` | 47.89% → 47.81% | 19.88% → 20.69% | 13.44% → 15.20% | 48.45% → 48.07% |
| **MFCC** | **20 / `bn_head_replay`** | 48.53% → 47.72% | 23.89% → **24.12%** | 17.57% → **18.40%** | 49.23% → 48.70% |
| LogMel | 5 / `bn_head_replay` | 56.10% → 53.14% | 20.31% → 20.22% | 13.32% → 13.20% | 59.47% → 57.85% |
| LogMel | 10 / `bn_head_replay` | 56.11% → 54.40% | 20.39% → 23.09% | 13.66% → 14.19% | 59.33% → 57.84% |
| **LogMel** | **20 / `head_only`** | 57.81% → **55.62%** | 18.66% → 21.18% | 10.62% → 12.21% | 63.73% → **63.32%** |
| PCEN | 5 / `bn_head` | 55.27% → 17.97% | 36.11% → 19.78% | 27.11% → 4.25% | 61.41% → 27.80% |
| PCEN | 10 / `bn_head` | 54.76% → 19.03% | 37.56% → 19.84% | 28.06% → 4.34% | 60.56% → 27.62% |
| PCEN | 20 / `bn_head` | 57.85% → 16.85% | 29.68% → 19.28% | 23.42% → 4.18% | 63.67% → 28.53% |

严格INT8下，MFCC 20-shot在BirdSet两项指标上最高且量化最稳定；若同时重视
Xeno与DB3V绝对精度，LogMel 20-shot `Head-Only`更均衡。PCEN三个shot的
Xeno Macro-F1均下降35.73–40.99个百分点，当前PTQ不可用于部署，需要先做QAT。

新增结果来源：

- `src/experiments/BirdSet_strict_INT8_quantization_8class/README.md`
- `src/experiments/BirdSet_strict_INT8_quantization_8class/experiment_protocol.json`
- `src/experiments/BirdSet_strict_INT8_quantization_8class/summary.csv`
- `src/experiments/BirdSet_strict_INT8_quantization_8class/aggregate.csv`
- `src/experiments/BirdSet_strict_INT8_quantization_8class/models/<chain_id>/`

### 量化结论、来源与复现

- MFCC是最耐PTQ量化的特征；小样本链路的主要Macro-F1变化均在约3个百分点内。
- LogMel是严格INT8下绝对精度与稳定性的较优折中。
- PCEN各数据集输入饱和率很低，但DB3V和BirdSet两套严格多种子链路的
  Xeno/DB3V Macro-F1均严重下降，降幅依任务和shot达到约13–41个百分点，
  说明输入截断不是主因，更可能是内部激活量化敏感，仍需逐层分析确认。
- 如果必须部署PCEN，需要进行量化感知训练后重新执行全部held-out评估。

结果与实现来源：

- 输入、转换与推理接口：`src/experiments/int8_inference.py`
- 全链路评估入口：`src/experiments/evaluate_int8_experiments.py`
- 零样本严格INT8：`src/experiments/ZeroShot_strict_INT8_quantization_8class/`
- DB3V严格多种子量化：`src/experiments/DB3V_strict_INT8_quantization_8class/`
- BirdSet严格多种子量化：`src/experiments/BirdSet_strict_INT8_quantization_8class/`
- 各量化目录中的`experiment_protocol.json`、`input_interfaces.json`、
  `summary.csv`和`models/<chain_id>/`分别保存协议、接口、逐模型结果和逐粒度
  原始报告；两个严格小样本目录另含三seed `aggregate.csv`。

```powershell
& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families zero_shot `
  --output-dir src\experiments\ZeroShot_strict_INT8_quantization_8class `
  --representative-samples 256 `
  --batch-size 128 `
  --num-threads 4

& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families db3v_strict_fewshot `
  --output-dir src\experiments\DB3V_strict_INT8_quantization_8class `
  --representative-samples 256 `
  --batch-size 128 `
  --num-threads 4

& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families birdset_strict_fewshot `
  --output-dir src\experiments\BirdSet_strict_INT8_quantization_8class `
  --representative-samples 256 `
  --batch-size 128 `
  --num-threads 4
```

## 指标与粒度解释

- **Accuracy**：整体预测正确比例，受类别不平衡影响。
- **Balanced Accuracy**：各类别召回率的算术平均，更适合类别不均衡数据。
- **Macro-F1**：先分别计算每类 F1 再等权平均，是当前单标签八类任务的主要综合指标。
- **Top-3**：真实类别是否位于模型概率最高的三个类别中。
- **一秒切片级**：衡量瞬时识别能力，容易受静音、弱鸣声和局部噪声影响。
- **录音/片段级**：平均同一来源连续窗口的 softmax 概率，衡量时间聚合后的识别能力。
- **BirdSet任意目标命中率**：用于 BirdSet 内部多标签验证，表示概率最高类别是否命中任一真实标签；其定义不同于单标签 Accuracy。

当前结论仅按严格四策略、5/10/20-shot和三随机种子标准解释。DB3V适配中，
PCEN 20-shot `Full`取得最高FP32 Macro-F1 70.50%±0.40%，但PTQ后降至
52.52%±0.79%；严格INT8部署改选LogMel 10-shot `Head-Only`，DB3V Macro-F1为
68.14%±0.17%。BirdSet适配中，FP32以PCEN 10-shot `BN+Head`取得最高目标域结果，
但PCEN PTQ同样严重退化；跨域均衡INT8链路优先LogMel 20-shot `Head-Only`，
只强调BirdSet指标和量化稳定性时可选MFCC 20-shot `BN+Head+Replay`。“地区单标签
适配”和“跨数据集多标签声景适配”仍是两条独立线路，不能互相替代。
