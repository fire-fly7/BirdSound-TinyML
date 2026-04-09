import tensorflow as tf
import numpy as np
import os

# 1️⃣ 加载模型
model_path = r"src\model_train&test\LiteRT_model\BC_ResNet" 
model = tf.keras.models.load_model(model_path)

# 2️⃣ 代表性数据集生成器 (用于整型量化校准)
# 使用你训练集的一部分数据
train_data = np.load('dataset_processing/output/MFCC_dataset_D/train_data.npy').astype(np.float32)
train_data = np.expand_dims(train_data, -1)

def representative_data_gen():
    # 随机取 100 个样本进行校准
    for i in range(100):
        data = np.expand_dims(train_data[i], axis=0)
        yield [data]

# 3️⃣ 配置 LiteRT 转换器
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# 强制输出为全整型 (适合没有 FPU 或追求极致功耗的场景)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 4️⃣ 转换并保存
litert_model = converter.convert()
output_path = os.path.join(r"src\model_train&test\LiteRT_model", f"{model.name}.tflite")

with open(output_path, "wb") as f:
    f.write(litert_model)

print(f"✅ LiteRT 量化模型已保存: {output_path}")