import numpy as np
import os
from tensorflow.python.keras.utils.np_utils import to_categorical
# from construct_model.DS_CNN_Model import create_model
# from construct_model.CNN_Model import create_model
# from construct_model.MobileNetV2 import create_model
from construct_model.BC_ResNet import create_model

# 1. 加载数据
train_data = np.load('dataset_processing/output/MFCC_dataset_D/train_data.npy') #特征数据
train_label = np.load('dataset_processing/output/MFCC_dataset_D/train_label.npy') #标签
test_data = np.load('dataset_processing/output/MFCC_dataset_D/test_data.npy') #测试特征数据
test_label = np.load('dataset_processing/output/MFCC_dataset_D/test_label.npy') #测试标签

# 2. 调整形状
train_data = np.expand_dims(train_data, -1)
test_data = np.expand_dims(test_data, -1)

# 3. 标签one-hot
num_classes = len(np.unique(train_label))
if num_classes != len(np.unique(test_label)):
    raise ValueError("训练集和测试集的类别数不一致！")
train_label = to_categorical(train_label, num_classes)
test_label = to_categorical(test_label, num_classes)

# 4. 创建模型
input_shape = train_data.shape[1:]
model = create_model(input_shape, num_classes)

# 5. 编译
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("输入：",input_shape, num_classes)

# 6. 训练
model.fit(train_data, train_label, batch_size=32, epochs=20, validation_split=0.2)

# 7. 输出模型结构
model.summary()

# 8. 测试模型
test_loss, test_acc = model.evaluate(test_data, test_label)
print('\nTest accuracy:', test_acc)

# 9. 保存模型
module_name = model.name
output_name = f"{module_name}.h5"
save_dir = r"src\model_train&test\TinyML_model"
output_path = os.path.join(save_dir, f"{module_name}.h5")
model.save(output_path)

print("模型已保存为:", output_path)