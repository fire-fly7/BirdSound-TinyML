import tensorflow as tf
from keras import layers, models

def create_model(
    input_shape,
    num_classes,
    alpha=0.5,
    dropout_rate=0.3
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=x,
        alpha=alpha,
        include_top=False,
        weights=None,
        pooling=None
    )

    x = base_model.output

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(
        num_classes,
        activation='softmax'
    )(x)

    model = models.Model(inputs, outputs, name=f"MobileNetV2_alpha_{alpha}")
    model.name = "MobileNetV2"
    return model
