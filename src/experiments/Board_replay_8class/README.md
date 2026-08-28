# 板端复刻训练端测试

本实验用于检测同一个严格 INT8 TFLite 从桌面 LiteRT 移植到 STM32
TFLite Micro 后是否保持等价。它复用训练端已经使用的测试样本、标签和聚合粒度，
不把这些样本重新解释为新的独立泛化测试集。

## 测试范围

| 范围 | 完整样本 | 原始聚合粒度 | 适用链路 |
|---|---:|---|---|
| Xeno-canto validation | 3,652 个一秒片段，480 条录音 | 片段级、录音级 | 全部 57 个模型 |
| BirdSet common 20-shot held-out | 91,325 个一秒片段，18,265 个 clip，197 条长录音 | 多标签片段级、5 秒 clip 级、全局单标签 clip 级 | 全部 57 个模型 |
| DB3V full | 85,264 个一秒片段，10,658 条八秒录音 | 区域片段/录音级、合并片段/录音级 | 零样本和 BirdSet few-shot 模型 |
| DB3V common 20-shot held-out | 81,576 个一秒片段，10,197 条八秒录音 | 区域片段/录音级、合并片段/录音级 | DB3V few-shot 模型 |

三种特征的样本身份、标签和 recording/clip 索引已经逐项核对一致。BirdSet
固定 held-out 中没有 `Setophaga_ruticilla` 正样本；DB3V 固定 held-out 的区域 2
没有标签 2 和 4。这是原测试范围本身的稀疏性，不能用其他区域样本补齐后仍称为
同一测试。

## 两阶段移植判据

第一阶段直接发送训练端特征张量：

`训练端 float32 特征 -> 板端输入量化 -> TFLM INT8 推理`

LED_TEST 的 `sweep --tflite` 会保存板端 `raw_output_int8` 和桌面
`reference_raw_int8`。逐样本预测必须一致，最大输出误差必须不超过 1 LSB。
这一步隔离模型、量化、串口编码和 TFLM 内核。

第二阶段发送相同来源的一秒 PCM16：

`PCM16 -> 板端 MFCC/LogMel/PCEN -> 输入量化 -> TFLM`

生成的 raw probe 有 192 条 WAV，八类各 24 条。全部是单声道、16 kHz、
PCM16、16,000 帧，清单同时保存 WAV 和 PCM payload SHA-256。它们保留原
Xeno-canto/DB3V 来源标签，未做人耳单物种复核，所以必须使用
`source-label` 策略，`scientific_metrics_valid=false`。

相同 PCM 在训练端重建特征后，相对原测试数组的最大绝对误差为：

| 特征 | 最大绝对误差 |
|---|---:|
| MFCC | 0.00006103515625 |
| LogMel | 0.0000152587890625 |
| PCEN | 0.0000007152557373 |

## 参考运行时

原严格 INT8 报告使用 `batch=128` 和默认 TFLite 解释器；LED_TEST 的可提交
对拍固定使用 TensorFlow 2.19、`batch=1` 和 `BUILTIN_REF`。量化内核路径不同
会导致少量 INT8 输出变化。因此：

- 移植是否等价，以同一 CSV 内的 `BUILTIN_REF` 和 TFLM 逐 LSB 对拍为准；
- 旧 batch-128 指标仍保留并报告差值，但不作为板端逐 LSB 真值；
- 完整测试必须继续使用原 recording/clip 聚合，不能只比较单片段 accuracy。

## 生成与运行

在训练仓库生成可复制的数据包：

```powershell
& .\.venv\Scripts\python.exe `
  src\dataset_processing\prepare_board_replay_testset.py `
  --materialize hardlink
```

生成目录 `board_replay_testset/` 已由 Git 忽略。`hardlink` 在本机不重复占用
约 3 GB；把目录复制到 Linux 时会自然成为普通文件。

Linux 上先跑核心模型 probe：

```bash
python3 src/experiments/run_board_replay.py \
  --led-test /srv/LED_TEST \
  --pack /srv/led-firmware-pack \
  --package board_replay_testset \
  --port /dev/ttyACM0 \
  --tier probe \
  --scope core
```

核心模型通过后改为 `--scope all`。最终复算完整指标时使用
`--tier full --mode native`。运行器会逐模型核对 `INDEX.csv`、TFLite
SHA-256、板端 INFO 标签顺序和特征类型，然后烧录、执行 parity、回放
Xeno-canto/BirdSet/DB3V，并调用
[`evaluate_board_replay.py`](../evaluate_board_replay.py) 生成对应粒度的指标。

## 实现与板端来源

- 数据包生成：[`prepare_board_replay_testset.py`](../../dataset_processing/prepare_board_replay_testset.py)
- Linux 自动运行：[`run_board_replay.py`](../run_board_replay.py)
- 指标恢复：[`evaluate_board_replay.py`](../evaluate_board_replay.py)
- 板端测试规格：
  [LED_TEST BOARD_BENCHMARK_TESTSET.md](https://github.com/fire-fly7/LED_TEST/blob/073a1f1a40d689448b332bac717961475315bebf/docs/BOARD_BENCHMARK_TESTSET.md)
- 板端串口协议：
  [LED_TEST serial_model_client.py](https://github.com/fire-fly7/LED_TEST/blob/073a1f1a40d689448b332bac717961475315bebf/tools/serial_model_client.py)
- 板端全模型执行器：
  [LED_TEST run_board_benchmark.py](https://github.com/fire-fly7/LED_TEST/blob/073a1f1a40d689448b332bac717961475315bebf/tools/run_board_benchmark.py)
