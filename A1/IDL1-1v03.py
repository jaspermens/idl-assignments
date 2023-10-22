import tensorflow as tf 
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten 
from keras import backend as K

# layers, dropout+-, aantal nodes/laag
# LM experiment regularisatie + learning rate + loss + optimizer

class MLP:
    def __init__(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Flatten(input_shape=[28,28]))
        self.model.add(keras.layers.Dense(300, activation="relu"))                                          # class_weight=class_weight
        self.model.add(keras.layers.Dense(100, activation="relu"))                                          # loss = "categorical_crossentropy"
        self.model.add(keras.layers.Dense(10, activation="softmax"))                                        # history = model.fit(X_train, y_train, epochs = 30, validation_data=(X_valid, y_valid))          
    
        self.filename = 'test'

    def printf(self, input):
        with open(self.filename, 'a') as f:
            print(input, file=f)

    def train(self,X_train, y_train, X_test, y_test, X_valid, y_valid):                                             
        self.model.compile(loss= 'sparse_categorical_crossentropy', optimizer="sgd", metrics=["accuracy"])   # optimizer = keras.optimizers.SGD(lr=0.01)
        self.history = self.model.fit(X_train, y_train, epochs = 30, validation_data=(X_valid, y_valid))          # class_weight=class_weight

    def figure(self):
        pd.DataFrame(self.history.history).plot(figsize=(8,5))
        plt.grid(True)
        plt.gca().set_ylim(0,1)
        plt.show()

class CNN:
    def __init__(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(64,7, activation="relu",padding="same", input_shape=[32,32,3]))
        self.model.add(keras.layers.MaxPooling2D(2))
        self.model.add(keras.layers.Conv2D(128,3, activation="relu",padding="same"))
        self.model.add(keras.layers.Conv2D(128,3, activation="relu",padding="same"))
        self.model.add(keras.layers.MaxPooling2D(2))
        self.model.add(keras.layers.Conv2D(256,3, activation="relu",padding="same"))
        self.model.add(keras.layers.Conv2D(256,3, activation="relu",padding="same"))
        self.model.add(keras.layers.MaxPooling2D(2))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(128,activation="relu"))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(64,activation="relu"))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(10,activation="softmax"))
        self.filename = 'test'

    def printf(self, input):
        with open(self.filename, 'a') as f:
            print(input, file=f)

    def train(self,X_train, y_train, X_test, y_test, X_valid, y_valid):                                             
        self.model.compile(loss= 'sparse_categorical_crossentropy', optimizer="sgd", metrics=["accuracy"])   # optimizer = keras.optimizers.SGD(lr=0.01)
        self.history = self.model.fit(X_train, y_train, epochs = 30, validation_data=(X_valid, y_valid))          # class_weight=class_weight

    def figure(self):
        pd.DataFrame(self.history.history).plot(figsize=(8,5))
        plt.grid(True)
        plt.gca().set_ylim(0,1)
        plt.show()

def MNIST():
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:]/255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.0
    # class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"]
    return X_train, y_train, X_test, y_test, X_valid, y_valid

def CIFAR():
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    assert X_train_full.shape == (50000, 32, 32, 3)
    assert X_test.shape == (10000, 32, 32, 3)
    assert y_train_full.shape == (50000, 1)
    assert y_test.shape == (10000, 1)
    X_valid, X_train = X_train_full[:10000] / 255.0, X_train_full[10000:]/255.0
    y_valid, y_train = y_train_full[:10000], y_train_full[10000:]
    X_test = X_test / 255.0
    return X_train, y_train, X_test, y_test, X_valid, y_valid


######################## MNIST #####################################
X_train, y_train, X_test, y_test, X_valid, y_valid = MNIST()
myMLP = MLP()
myMLP.train(X_train, y_train, X_test, y_test, X_valid, y_valid)
myMLP.figure()

######################## CIFAR #####################################
X_train, y_train, X_test, y_test, X_valid, y_valid = CIFAR()
myCNN = CNN()
myCNN.train(X_train, y_train, X_test, y_test, X_valid, y_valid)
myCNN.figure()