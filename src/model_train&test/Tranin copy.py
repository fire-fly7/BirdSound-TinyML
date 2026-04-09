import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from construct_model.BC_ResNet import create_model

# 1. 加载数据 (保持你原始的路径)
train_data = np.load('dataset_processing/output/MFCC_dataset_D/train_data.npy')
train_label = np.load('dataset_processing/output/MFCC_dataset_D/train_label.npy')
test_data = np.load('dataset_processing/output/MFCC_dataset_D/test_data.npy')
test_label = np.load('dataset_processing/output/MFCC_dataset_D/test_label.npy')

# 2. 调整形状 (MFCC 特征适配)
train_data = np.expand_dims(train_data, -1)
test_data = np.expand_dims(test_data, -1)

# 3. 标签处理
num_classes = len(np.unique(train_label))
train_label = to_categorical(train_label, num_classes)
test_label = to_categorical(test_label, num_classes)

# 4. 创建模型
input_shape = train_data.shape[1:]
model = create_model(input_shape, num_classes)

# 5. 编译
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. 训练
model.fit(train_data, train_label, batch_size=32, epochs=20, validation_split=0.2)

# 7. 保存为 SavedModel 格式 (LiteRT 推荐格式)
save_dir = r"src\model_train&test\LiteRT_model"
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, model.name)
model.save(model_save_path) # 保存为文件夹格式，包含 assets 和 variables

print(f"模型已保存至: {model_save_path}")