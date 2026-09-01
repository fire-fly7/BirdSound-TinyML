# DB3V严格微调与多随机种子实验

本目录保存三条Xeno-canto DS-CNN基准链路在DB3V 5/10/20-shot support上的严格微调消融。当前标准以DB3V原始Xeno-canto来源录音为独立样本单位；同一来源切出的多个八秒块不得跨support/test，也不得在主指标中获得多票。

## 数据隔离与规模

| shot | support源录音 | support八秒块 | 共同test源录音 | 共同test八秒块 | 共同test一秒切片 | 源录音短缺 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 116 | 1,009 | 970 | 7,660 | 61,280 | 4 |
| 10 | 219 | 1,665 | 970 | 7,660 | 61,280 | 21 |
| 20 | 393 | 2,998 | 970 | 7,660 | 61,280 | 87 |

5/10/20-shot support在源录音级严格嵌套；共同test固定为最大20-shot support之外的970条独立源录音。三种特征的样本身份一致。审计同时确认support/test重叠为0，且DB3V与Xeno训练1,920条、验证480条录音的来源ID重叠均为0。

审计来源：`split_audit.json`；生成和复核入口分别为`../../dataset_processing/prepare_external_fewshot.py`与`../audit_db3v_source_split.py`。

## 严格策略

| 策略 | 可训练范围 | BatchNorm | Xeno replay |
|---|---|---|---|
| `head_only` | 最终8类softmax层 | 冻结 | 无 |
| `bn_head` | 全部BatchNorm与最终层 | 训练 | 无 |
| `bn_head_replay` | BN+Head | 训练 | support与类别均衡Xeno切片1:1 |
| `full` | 全部层 | 训练 | 无 |

实验覆盖`3特征 × 3 shot × 4策略 × 3 seed = 108`个模型，种子为42、123、2026。策略和epoch只由support内部验证与Xeno保留率组成的适配分数选择；DB3V共同test和BirdSet公共test均不参与选型。每个运行都保存微调后的Xeno遗忘指标。

## 选中策略及严格INT8结果

下表只列每个“特征×shot”按三seed平均适配分数预先选中的策略。DB3V是970条原始源录音级Macro-F1；数值为均值±样本标准差。

| 特征 | shot/策略 | Xeno Macro-F1（FP32→INT8） | DB3V Macro-F1（FP32→INT8） |
|---|---|---:|---:|
| MFCC | 5 / `head_only` | 47.73±0.31% → 47.78±0.31% | 55.17±0.20% → 54.63±0.18% |
| MFCC | 10 / `bn_head_replay` | 49.16±0.49% → 47.01±0.50% | 57.49±0.27% → 56.17±0.63% |
| MFCC | 20 / `bn_head_replay` | 48.39±0.45% → 48.02±0.55% | 57.40±0.86% → 56.99±0.92% |
| LogMel | 5 / `head_only` | 59.59±0.23% → 53.32±0.32% | 67.23±0.48% → 63.79±0.19% |
| LogMel | 10 / `bn_head_replay` | 61.00±0.26% → 59.38±0.50% | 68.12±0.38% → 66.67±0.96% |
| **LogMel** | **20 / `bn_head_replay`** | **61.36±0.21% → 60.01±1.32%** | **68.84±0.68% → 67.73±0.59%** |
| PCEN | 5 / `full` | 62.65±0.82% → 37.15±0.50% | **71.93±0.48%** → 55.20±1.10% |
| PCEN | 10 / `full` | 62.83±0.67% → 32.34±2.42% | 71.78±1.00% → 46.09±6.09% |
| PCEN | 20 / `full` | 61.64±0.65% → 28.08±4.86% | 71.27±0.20% → 37.73±9.40% |

预先选中策略中，FP32目标域最高的是PCEN 5-shot `full`；PTQ后PCEN明显失稳。严格INT8部署推荐改为LogMel 20-shot `bn_head_replay`，其DB3V Macro-F1为67.73%±0.59%，Xeno Macro-F1为60.01%±1.32%。

完整36组结果仍保存在`aggregate.csv`。其中未被适配分数选中的策略即使held-out更高，也不能在查看test后改列为推荐模型，否则会形成测试集选型偏差。

## 文件与复现

- `runs.csv`：108个逐seed结果。
- `aggregate.csv`：36个“特征×shot×策略”汇总。
- `experiment_protocol.json`：严格数据、选型和统计协议。
- `selected_cross_domain_runs.csv`与`selected_cross_domain_aggregate.csv`：选中模型在BirdSet公共test上的27个运行和9组汇总。
- `../DB3V_strict_INT8_quantization_8class/`：同一27个模型的FP32–INT8三域配对结果。
- `<feature>/<shot>shot/<policy>/seed_<seed>/`：模型、训练历史、Xeno遗忘和逐粒度评估报告。

```powershell
& .\.venv\Scripts\python.exe src\experiments\audit_db3v_source_split.py
& .\.venv\Scripts\python.exe src\experiments\run_db3v_ablation_multiseed.py --summarize-only
& .\.venv\Scripts\python.exe src\experiments\evaluate_int8_experiments.py `
  --families db3v_strict_fewshot `
  --output-dir src\experiments\DB3V_strict_INT8_quantization_8class
```

DB3V来源：[Zenodo 11544734](https://doi.org/10.5281/zenodo.11544734)。
