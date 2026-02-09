import tensorflow as tf
from keras import layers, models

def create_mobilenetv2_audio(
    input_shape,
    num_classes,
    alpha=0.5,
    dropout_rate=0.3
):
    inputs = tf.keras.Input(shape=input_shape)

    # 单通道 MFCC
    x = inputs

    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=x,
        alpha=alpha,
        include_top=False,
        weights=None,      # 音频任务，别用 imagenet
        pooling=None
    )

    x = base_model.output

    # 全局平均池化（非常关键，嵌入式友好）
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(
        num_classes,
        activation='softmax'
    )(x)

    model = models.Model(inputs, outputs, name=f"MobileNetV2_alpha_{alpha}")

    return model
