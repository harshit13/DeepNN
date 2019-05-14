# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2019-05-14 07:03:11
# @Last Modified by:   harshit
# @Last Modified time: 2019-05-14 08:02:59

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os


batch_size = 1024
epochs = 10
sessions = 5


def sin():
    inputs = Input((1,))
    l0 = Dense(
        512, activation='relu', kernel_initializer="random_uniform")(inputs)
    l1 = Dense(
        1024, activation='relu', kernel_initializer="random_uniform")(l0)
    l2 = Dense(
        1024, activation='relu', kernel_initializer="random_uniform")(l1)
    output = Dense(1, activation='linear')(l2)

    model = Model(inputs=[inputs], outputs=[output])
    adm = Adam(lr=0.001)
    model.compile(optimizer=adm, loss='mean_absolute_error', metrics=['mae'])
    model.summary()
    return model


def main():

    model = sin()
    # setting model callbacks

    for sess in range(sessions):
        X = np.random.random_sample((batch_size,)) * 10 * np.pi - (5 * np.pi)
        Y = np.sin(X)
        for epoch in range(epochs):
            print("Epoch", epoch)
            model.fit(X, Y, validation_split=0.1)
            y_ = model.predict(X)

            fig, axs = plt.subplots(1, 2)
            axs[0].scatter(X, Y, color="blue")
            axs[1].scatter(X, y_, color="red")
            name = str(sess) + "_" + str(epoch) + ".png"
            fig.savefig(os.path.join("sinx_graphs", name))
            plt.close()


if __name__ == '__main__':
    main()
