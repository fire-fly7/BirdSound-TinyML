import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # 新增导入

def slice_and_extract_mfcc(input_dir,
                            output_train_data="train_data.npy",
                            output_train_label="train_label.npy",
                            output_test_data="test_data.npy",
                            output_test_label="test_label.npy",
                            slice_duration=1.0, sr=16000, n_mfcc=13, test_ratio=0.75):  # 默认测试占25%
    x_data = []
    y_data = []

    label_map = {}
    label_idx = 0

    print("🔍 开始处理音频文件...")

    for label_name in sorted(os.listdir(input_dir)):
        class_dir = os.path.join(input_dir, label_name)
        if not os.path.isdir(class_dir):
            continue

        if label_name not in label_map:
            label_map[label_name] = label_idx
            label_idx += 1

        for fname in tqdm(os.listdir(class_dir), desc=f"[{label_name}]"):
            if not fname.endswith(".wav"):
                continue
            fpath = os.path.join(class_dir, fname)

            try:
                y, _ = librosa.load(fpath, sr=sr)
                total_len = len(y)
                slice_len = int(sr * slice_duration)
                slices = total_len // slice_len

                for i in range(slices):
                    start = i * slice_len
                    end = start + slice_len
                    y_slice = y[start:end]

                    mfcc = librosa.feature.mfcc(y=y_slice, sr=sr, n_mfcc=n_mfcc)
                    mfcc = mfcc.T  # shape: (time, n_mfcc)

                    if mfcc.shape[0] < 10:  # 排除异常
                        continue

                    x_data.append(mfcc)
                    y_data.append(label_map[label_name])

            except Exception as e:
                print(f"❗ 处理失败: {fpath}, 错误: {e}")

    print("✅ 提取完成，开始处理数据...")

    # 统一长度（填充）
    max_len = max([x.shape[0] for x in x_data])
    x_padded = np.zeros((len(x_data), max_len, n_mfcc), dtype=np.float32)

    for i, x in enumerate(x_data):
        length = x.shape[0]
        x_padded[i, :length, :] = x

    y_array = np.array(y_data, dtype=np.int64)

    # 🎯 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x_padded, y_array, test_size=test_ratio, stratify=y_array, random_state=42
    )

    # 保存为 .npy 文件
    output_dir = "MFCC_dataset_" + input_dir[-1]
    output_dir = os.path.join("dataset_processing", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_train_data = os.path.join(output_dir, 'train_data.npy')
    output_train_label = os.path.join(output_dir, 'train_label.npy')
    output_test_data = os.path.join(output_dir, 'test_data.npy')
    output_test_label = os.path.join(output_dir, 'test_label.npy')
    np.save(output_train_data, x_train)
    np.save(output_train_label, y_train)
    np.save(output_test_data, x_test)
    np.save(output_test_label, y_test)

    print(f"📁 训练特征保存至: {output_train_data}")
    print(f"📁 训练标签保存至: {output_train_label}")
    print(f"📁 测试特征保存至: {output_test_data}")
    print(f"📁 测试标签保存至: {output_test_label}")
    print(f"🔢 类别标签映射: {label_map}")
    print(f"🧮 训练数据形状: {x_train.shape}")
    print(f"🧮 测试数据形状: {x_test.shape}")

    return label_map

# slice_and_extract_mfcc("bird_dataset")
slice_and_extract_mfcc(r"row_dataset\row_bird_dataset_D")