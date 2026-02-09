import tensorflow as tf

DS_model = tf.keras.models.load_model("ds_cnn_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(DS_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("DS_CNN_Model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite 模型已保存为 DS_CNN_Model.tflite")