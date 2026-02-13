import tensorflow as tf
import os

# 1️⃣ 加载模型
model = tf.keras.models.load_model("")

# 2️⃣ 获取模型来源文件名（例如 BC_ResNet）
module_name = model.name  

# 3️⃣ 指定输出目录
output_dir = "src\model_train&test\TinyML_model"

# 4️⃣ 构建输出文件名
output_path = os.path.join(output_dir, f"{module_name}_model.tflite")

# 5️⃣ 转换为 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# 6️⃣ 保存文件
with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite 模型已保存为: {output_path}")
