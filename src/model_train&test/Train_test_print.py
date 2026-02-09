import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical
from CNN_model.DS_CNN_Model import create_ds_cnn_model
import io
import os
import tempfile

def load_data(data_dir):
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'))
    train_label = np.load(os.path.join(data_dir,'train_label.npy'))
    test_data = np.load(os.path.join(data_dir,'test_data.npy'))
    test_label = np.load(os.path.join(data_dir,'test_label.npy'))
    
    train_data = np.expand_dims(train_data, -1)
    test_data = np.expand_dims(test_data, -1)
    
    
    num_classes = len(np.unique(train_label))
    if num_classes != len(np.unique(test_label)):
        raise ValueError("训练集和测试集的类别数不一致！")

    train_label = to_categorical(train_label, num_classes)
    test_label = to_categorical(test_label, num_classes)
    
    return train_data, train_label, test_data, test_label, num_classes

def train_and_evaluate(model_fn, model_name, input_shape, num_classes, 
                       train_data, train_label, test_data, test_label):
    
    print(f"\n🚀 正在训练模型：{model_name}")
    model = model_fn(input_shape, num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(train_data, train_label,
                        batch_size=32, epochs=20, validation_split=0.2, verbose=0)
    train_time = time.time() - start_time

    test_loss, test_acc = model.evaluate(test_data, test_label, verbose=0)
    
    with tempfile.NamedTemporaryFile(suffix='.h5',delete=False) as temp_file:
        model_name = temp_file.name
    model.save(model_name)
    model_size_kb = os.path.getsize(model_name) / 1024
    print(f"✅ 模型 {model_name} 训练完成，准确率: {test_acc:.4f}, 损失率: {test_loss:.4f}, 训练耗时: {train_time:.1f} s, 模型大小: {model_size_kb:.1f} KB, 参数: {model.count_params():,},记录: {history}")
    os.remove(model_name)

 

# --- 主流程 ---
if __name__ == "__main__":
    train_data, train_label, test_data, test_label, num_classes = load_data("dataset_processing\MFCC_dataset_D") #input data directory
    input_shape = train_data.shape[1:]


    train_and_evaluate(create_ds_cnn_model, "DS_CNN_Model", input_shape, num_classes, train_data, train_label, test_data, test_label)