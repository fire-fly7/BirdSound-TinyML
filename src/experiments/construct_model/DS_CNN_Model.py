import tensorflow as tf
from keras import layers, models

def create_model(input_shape, num_classes):

    model = models.Sequential()
    model.name = "DS_CNN_Model"

    # Initial Conv Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Depthwise Separable Convolution Block
    model.add(layers.SeparableConv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Another Depthwise Separable Convolution Block
    model.add(layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 1), padding='valid'))

    # Global Average Pooling Layer
    model.add(layers.GlobalAveragePooling2D())

    # Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

