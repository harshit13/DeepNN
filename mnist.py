# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2019-05-14 08:16:17
# @Last Modified by:   harshit
# @Last Modified time: 2019-05-14 08:54:21

from keras.datasets import mnist
from keras.layers import Dense, Input, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np


SHAPE = (28, 28)


def nn_model():
    inputs = Input((28, 28, 1))
    l0 = Flatten()(inputs)
    l1 = Dense(
        512, activation='relu', kernel_initializer="random_uniform")(l0)
    l2 = Dense(
        1024, activation='relu', kernel_initializer="random_uniform")(l1)
    l3 = Dense(
        1024, activation='relu', kernel_initializer="random_uniform")(l2)
    output = Dense(10, activation='softmax')(l3)

    model = Model(inputs=[inputs], outputs=[output])
    adm = Adam(lr=0.001)
    model.compile(
        optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def cnn_model():
    inputs = Input((28, 28, 1))

    c1 = Conv2D(16, (3, 3), activation='elu', padding='same')(inputs)
    c2 = Dropout(0.1)(c1)
    c3 = Conv2D(16, (3, 3), activation='elu', padding='same')(c2)
    c4 = MaxPooling2D((2, 2))(c3)

    c5 = Conv2D(32, (3, 3), activation='elu', padding='same')(c4)
    c6 = Dropout(0.1)(c5)
    c7 = Conv2D(32, (3, 3), activation='elu', padding='same')(c6)
    c8 = MaxPooling2D((2, 2))(c7)

    l0 = Flatten()(c8)
    l1 = Dense(
        1024, activation='relu', kernel_initializer="random_uniform")(l0)
    l2 = Dense(
        512, activation='relu', kernel_initializer="random_uniform")(l1)
    output = Dense(10, activation='sigmoid')(l2)

    model = Model(inputs=[inputs], outputs=[output])
    adm = Adam(lr=0.001)
    model.compile(
        optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    model = cnn_model()
    # setting model callbacks
    earlystopper = EarlyStopping(patience=10, verbose=2)
    checkpointer = ModelCheckpoint(
        'mnist_cnn.h5', save_best_only=True, verbose=2)

    X_train = X_train / 256
    X_test = X_test / 256
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=40, batch_size=128,
        callbacks=[earlystopper, checkpointer])

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
