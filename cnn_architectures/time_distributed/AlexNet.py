from tensorflow import keras


def add_alex_net_layers(model):
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                            input_shape=(5, 224, 224, 3))
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.BatchNormalization()
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.MaxPool2D(pool_size=(2, 2))
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same")
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.BatchNormalization()
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.MaxPool2D(pool_size=(3, 3))
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.BatchNormalization()
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same")
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.BatchNormalization()
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same")
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.BatchNormalization()
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.MaxPool2D(pool_size=(2, 2))
    ))
