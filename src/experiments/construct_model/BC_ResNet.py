import tensorflow as tf
from keras import layers, models

def bc_res_block(x, filters, kernel_size=(3, 3), stride=1):

    shortcut = x

    # 1. Depthwise convolution
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 2. Pointwise convolution
    x = layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        padding='same',
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 3. Broadcast frequency convolution
    freq_dim = x.shape[2]

    x = layers.Conv2D(
        filters,
        kernel_size=(1, freq_dim),
        padding='same',
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)

    # 4. Fix shortcut
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=stride,   # ⭐ 必须同步
            padding='same',
            use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

def create_model(input_shape, num_classes, base_filters=32):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial Conv Layer
    x = layers.Conv2D(
        base_filters,
        kernel_size=(3, 3),
        padding='same',
        use_bias=False
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # BC-ResBlocks
    x = bc_res_block(x, base_filters)
    x = bc_res_block(x, base_filters)
    x = bc_res_block(x, base_filters * 2, stride=2)
    x = bc_res_block(x, base_filters * 2)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(
        num_classes,
        activation='softmax'
    )(x)

    model = models.Model(inputs, outputs, name="BC_ResNet")
    model.name = "BC_ResNet"
    return model
