import tensorflow as tf
from keras import layers, models

def create_model(input_shape, num_classes):

    model = models.Sequential()
    model.name = "CNN_Model"

    # Initial Conv Layer
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape, data_format='channels_last'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), data_format='channels_last'))

    # Second Conv Layer
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), data_format='channels_last'))

    # Third Conv Layer
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), data_format='channels_last'))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D(data_format='channels_last'))

    # Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
