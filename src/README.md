# 源码与实验目录

项目的可执行流程统一放在 `src/` 下，并按职责分为三部分：

- `row_dataset/`：数据获取脚本与本地原始数据；
- `dataset_processing/`：MFCC、Log-Mel、PCEN预处理，以及外部数据划分；
- `experiments/`：模型结构、训练、评估、模型转换和实验结果。

所有命令默认从项目根目录执行。`src/row_dataset/` 中的下载数据和
`src/dataset_processing/output/` 中的生成特征继续由 `.gitignore` 排除，只有数据
获取脚本、处理代码、模型工件与评估报告进入版本控制。
