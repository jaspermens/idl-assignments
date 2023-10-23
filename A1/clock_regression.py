import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras import backend as K


# DATASET_PREFIX = 'drive/MyDrive/idl/datasets/'
# DATASET_PREFIX = 'clock_data/'

# DATASET_PREFIX = '/kaggle/input/clock-images/'
DATASET_PREFIX = 'A1/'
labels = np.load(DATASET_PREFIX + 'labels.npy')
images = np.load(DATASET_PREFIX + 'images.npy')


class clock_CNN_regression:
    def __init__(self, images, labels,
                batch_size=128,
                num_epochs=20,
                ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.image_shape = (150, 150, 1)
#        self.filename = 'test'

        self.prep_data(images, labels)
        self.build_model()


    def build_model(self):
        def nonnegative_mod720(x):
            # tf.where? -> mogelijk sneller
            xmod720 = tf.math.mod(x, 720)
            return tf.math.mod(xmod720+720, 720)
        
        self.model = keras.models.Sequential([        
            Conv2D(64, kernel_size=15, strides=4, activation='relu', input_shape=self.image_shape),
            MaxPooling2D(pool_size=4, strides=4),
            BatchNormalization(),
#             Conv2D(64, kernel_size=3, strides=1, activation='relu'),
#             MaxPooling2D(pool_size=2, strides=2),
#             BatchNormalization(),
            Conv2D(16, kernel_size=3, strides=1, activation='relu'),
            MaxPooling2D(pool_size=2, strides=2),
            BatchNormalization(),
            Flatten(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            keras.layers.Lambda(nonnegative_mod720),
            Dense(1, activation='linear'),
        ])

        self.model.summary()
        
    def convert_labels(self, labels):
        """
        Converts tuple labels to integers representing minutes since 00:00
        """
        return tf.cast(labels[:,0]*60 + labels[:,1], tf.float32)

    def prep_data(self, X_full, y_full):
        test_fraction = 0.2
        y_full = self.convert_labels(y_full)
        X_full = X_full.reshape(len(X_full),*self.image_shape)/255
        
        full_dataset = tf.data.Dataset.from_tensor_slices((X_full, y_full)).batch(self.batch_size)
        self.train_dataset, self.test_dataset = keras.utils.split_dataset(full_dataset, 
                                                                          right_size=test_fraction, 
                                                                          shuffle=True, 
                                                                          seed=123)
    def cyclic_errors(self, y_true, y_pred):
        y_pred = tf.math.mod(y_pred, 720)
        error = tf.abs(y_true - y_pred)
        return tf.minimum(error, 720-error)
    
    def mean_deviation_minutes(self,y_true, y_pred):# Specific for regression task
        # not quite a loss function -> maybe book example
        mean_deviation = tf.reduce_mean(self.cyclic_errors(y_true, y_pred))
        return mean_deviation

    def train_model(self):
        self.model.compile(
                loss=self.mean_deviation_minutes,
                optimizer=keras.optimizers.Nadam(learning_rate=.01, beta_1=0.9, beta_2=0.999),
#                metrics=[self.mean_deviation_minutes],              
        )
        self.history = self.model.fit(self.train_dataset,
                                      epochs=self.num_epochs,
                                      validation_data=self.test_dataset,
                                      batch_size=self.batch_size,
                                      )


myCNN = clock_CNN_regression(images, labels, batch_size=32, num_epochs=2)
myCNN.train_model()
myCNN.model.evaluate(myCNN.test_dataset)