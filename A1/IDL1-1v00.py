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

class clock_CNN:    
    def __init__(self, images, labels):
        self.num_classes = 12
        self.batch_size = 32
        self.num_epochs = 20
        self.image_shape = (150, 150, 1)
        self.filename = 'test'

        self.prep_data(images, labels)
        self.build_model()

    
    def build_model(self):
        self.model = keras.models.Sequential()
        self.model.add(Conv2D(64, kernel_size=5, activation='relu', strides=3, input_shape=self.image_shape))
        self.model.add(MaxPooling2D(pool_size=4))
        self.model.add(Conv2D(32, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))   
        self.model.add(Conv2D(16, kernel_size=2, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))   
        keras.layers.BatchNormalization()         
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.summary()
    
    def convert_labels(self,labels):
        """
        Converts tuple labels to integers
        """
        return tf.math.floor((labels[:,0] + labels[:,1]/60)/12 * self.num_classes)
        # return 2*labels[:,0] + labels[:,1]//30

    def prep_data(self, X_full, y_full):
        validation_fraction = 0.1
        test_fraction = 0.15

        num_validation = int(validation_fraction * len(y_full))
        num_testing = int(test_fraction * len(y_full))
        num_train = len(y_full) - num_validation - num_testing

        # num_val = 1000
        # num_test = 3000
        # num_train = 18000
        y_full = self.convert_labels(y_full)
        print(X_full.shape)
        X_full = X_full.reshape(len(X_full),150,150,1)/255
        print(X_full.shape)
        
        # def array_to_split_datasets(arr):
        #     db_full = tf.data.Dataset.from_tensor_slices(arr)
        #     db_full = db_full.shuffle(db_full.cardinality(), seed=10, reshuffle_each_iteration=False)

        #     db_train = db_full.take(num_train)
        #     db_remainder = db_full.skip(num_train)
        #     db_validation = db_remainder.take(num_validation)
        #     db_test = db_remainder.skip(num_validation)

        #     return db_train, db_validation, db_test
        
        # self.train_images, self.valid_images, self.test_images = array_to_split_datasets(X_full)
        # del(X_full)
        # self.train_labels, self.valid_labels, self.test_labels = array_to_split_datasets(y_full)
        # del(y_full)
        # y_full = tf.data.Dataset.from_tensor_slices(y_full)
        # X_full = tf.data.Dataset.from_tensor_slices(X_full)

        # y_full = y_full.shuffle(y_full.cardinality(), seed=10, reshuffle_each_iteration=False)
        # X_full = X_full.shuffle(X_full.cardinality(), seed=10, reshuffle_each_iteration=False)

        # X_train = X_full.take(num_train)
        # X_remainder = X_full.skip(num_train)
        # X_valid = X_remainder.take(num_validation)
        # X_test = X_remainder.skip(num_validation)

        from sklearn.model_selection import train_test_split

        train_valid_images, self.test_images, train_valid_labels, self.test_labels = train_test_split(X_full, y_full, random_state=100, test_size=0.15)
        self.train_images, self.valid_images, self.train_labels, self.valid_labels = train_test_split(train_valid_images, train_valid_labels, random_state=100, test_size=0.10)
        # X_train, X_valid, X_test = X_full[:num_train], X_full[num_train:num_train+num_validation], X_full[num_train+num_validation:]    
        # y_train, y_valid, y_test = y_full[:num_train], y_full[num_train:num_train+num_validation], y_full[num_train+num_validation:]

        # self.train_images = X_train
        # self.train_labels = y_train

        # self.test_images = X_test
        # self.test_labels = y_test
        
        # self.valid_images = X_valid
        # self.valid_labels = y_valid

        print(len(self.valid_labels), len(self.train_labels), len(self.test_labels))
        # shuffle_buffer = 100
        # self.train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=shuffle_buffer)
        # self.test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(buffer_size=shuffle_buffer)
        # self.validation_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).shuffle(buffer_size=shuffle_buffer)


    def printf(self, input):
        with open(self.filename, 'a') as f:
            print(input, file=f)

    def cyclic_loss(self, y_true, y_pred):
        print(y_true)

        linear_errors = tf.math.abs(y_true - y_pred)
        errors = tf.minimum(self.num_classes - linear_errors, linear_errors)
        return tf.reduce_mean(tf.math.pow(errors,3), axis=-1)
        # errors =  tf.math.minimum(tf.abs((y_true - y_pred) % 12), tf.abs((y_pred - y_true) % 12))
        
        # is_small_error = tf.abs(errors) < self.num_classes//4
        # squared_loss = tf.square(errors) / 2
        # linear_loss = tf.abs(errors) - 0.5

        # return tf.where(is_small_error, squared_loss, linear_loss)

    def train_model(self):                                             
        self.model.compile(
                # loss='sparse_categorical_crossentropy', 
                loss=self.cyclic_loss, 
                # optimizer = keras.optimizers.SGD(learning_rate=10),
                optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
                metrics=['accuracy'],
        )
        
        self.history = self.model.fit(self.train_images, self.train_labels, 
                                      epochs=self.num_epochs, 
                                      validation_data=(self.valid_images, self.valid_labels), 
                                      batch_size=self.batch_size,
                                      )
        

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

############################ Clock #################################
labels = np.load('labels.npy')
images = np.load('images.npy')

rng = np.random.default_rng(seed=10)
shuffled_indices = rng.permutation(len(labels))
labels = labels[shuffled_indices]
images = images[shuffled_indices]

# labels = NumpyFileIODataset('labels.npy')
# images = np.load('images.npy')
# X_train, y_train, X_test, y_test, X_valid, y_valid = prepdata(images, labels)

#X_train, y_train, X_test, y_test, X_valid, y_valid = MNIST()
myCNN = clock_CNN(images, labels)
myCNN.train_model()
# myCNN.figure()

######################## MNIST #####################################
# X_train, y_train, X_test, y_test, X_valid, y_valid = MNIST()
# myMLP = MLP()
# myMLP.train(X_train, y_train, X_test, y_test, X_valid, y_valid)
# myMLP.figure()




