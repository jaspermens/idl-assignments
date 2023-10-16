import tensorflow as tf 
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten 
from keras import backend

# layers, dropout+-, aantal nodes/laag
# LM experiment regularisatie + learning rate + loss + optimizer

class MLP:
    def __init__(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Flatten(input_shape=[28,28]))
        self.model.add(keras.layers.Dense(300, activation="relu"))                                          # class_weight=class_weight
        self.model.add(keras.layers.Dense(100, activation="relu"))                                          # loss = "categorical_crossentropy"
        self.model.add(keras.layers.Dense(10, activation="softmax"))                                        # history = model.fit(X_train, y_train, epochs = 30, validation_data=(X_valid, y_valid))          
    
    filename = 'test'
    def printf(input):
        with open(filename, 'a') as f:
            print(s,file=f)

    def train(self,X_train, y_train, X_test, y_test, X_valid, y_valid):                                             
        self.model.compile(loss= 'sparse_categorical_crossentropy', optimizer="sgd", metrics=["accuracy"])   # optimizer = keras.optimizers.SGD(lr=0.01)
        self.history = self.model.fit(X_train, y_train, epochs = 30, validation_data=(X_valid, y_valid))          # class_weight=class_weight

    def figure(self):
        pd.DataFrame(self.history.history).plot(figsize=(8,5))
        plt.grid(True)
        plt.gca().set_ylim(0,1)
        plt.show()

class clock_CNN:    
    def __init__(self):
        self.num_classes = 24
        self.image_shape=(150,150,1)
        self.model = keras.models.Sequential()
        self.model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=self.image_shape))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))
    
    def printf(input):
        filename = 'test'
        with open(filename, 'a') as f:
            print(input, file=f)

    def train(self,X_train, y_train, X_test, y_test, X_valid, y_valid):                                             
        self.model.compile(loss= 'sparse_categorical_crossentropy', optimizer="sgd", metrics=["accuracy"])   # optimizer = keras.optimizers.SGD(lr=0.01)
        self.history = self.model.fit(X_train, y_train, epochs = 30, validation_data=(X_valid, y_valid), batch_size=32)          # class_weight=class_weight

    def figure(self):
        pd.DataFrame(self.history.history).plot(figsize=(8,5))
        plt.grid(True)
        plt.gca().set_ylim(0,1)
        plt.show()


# class CNN:

def prepdata(X_full, y_full):
    num_val = 1000
    num_test = 3000
    # num_train = 18000
    X_valid, X_test, X_train = X_full[:num_val]/255.0, X_full[num_val:num_test]/255.0, X_full[num_test:]/255.0    
    y_valid, y_test, y_train = y_full[:num_val], y_full[num_val:num_test], y_full[num_test:]

    return X_train, label_to_class(y_train), X_test, label_to_class(y_test), X_valid, label_to_class(y_valid)

def label_to_class(labels):
    return 2*labels[:,0] + labels[:,1]//30

def MNIST():
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:]/255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.0
    # class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"]
    return X_train, y_train, X_test, y_test, X_valid, y_valid

############################ Clock #################################
images = np.load('images.npy')
labels = np.load('labels.npy')
X_train, y_train, X_test, y_test, X_valid, y_valid = prepdata(images, labels)

#X_train, y_train, X_test, y_test, X_valid, y_valid = MNIST()
myCNN = clock_CNN()
myCNN.train(X_train, y_train, X_test, y_test, X_valid, y_valid)
myCNN.figure()

######################## MNIST #####################################
# X_train, y_train, X_test, y_test, X_valid, y_valid = MNIST()
# myMLP = MLP()
# myMLP.train(X_train, y_train, X_test, y_test, X_valid, y_valid)
# myMLP.figure()



