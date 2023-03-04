from tensorflow import keras


def add_vgg16_layers(model):
    model.add(keras.layers.Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3),
                                  padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="vgg16"))
