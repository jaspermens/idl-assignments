import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras import backend as K

DATASET_PREFIX = 'clock_data/'
labels = np.load(DATASET_PREFIX + 'labels.npy')
images = np.load(DATASET_PREFIX + 'images.npy')
mod_param = 12
downsize = 4
'''
Issues encountered
- Memory overflow / process killed => introduced downsize factor for dense layers
- 
'''
class clock_CNN_regression:
    def __init__(self, images, labels,
                num_classes=1,              # comp. classification = 12
                batch_size=128,
                num_epochs=20,
                ):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.image_shape = (150, 150, 1)
#        self.filename = 'test'

        self.prep_data(images, labels)
        self.build_model()


    def build_model(self):
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
            Dense(64/downsize, activation='relu'),
            Dropout(0.2),
            Dense(128/downsize, activation='relu'),
            Dense(self.num_classes, activation='softmax'),
        ])

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
        
        rng = np.random.default_rng(seed=10)
        shuffled_indices = rng.permutation(len(labels))
        y_full = y_full[shuffled_indices]
        X_full = X_full[shuffled_indices]

        y_full = self.convert_labels(y_full)
        y_full = keras.utils.to_categorical(y_full, num_classes = self.num_classes)

        X_full = X_full.reshape(len(X_full),*self.image_shape)/255

        X_train, X_valid, X_test = X_full[:num_train], X_full[num_train:num_train+num_validation], X_full[num_train+num_validation:]
        y_train, y_valid, y_test = y_full[:num_train], y_full[num_train:num_train+num_validation], y_full[num_train+num_validation:]

        self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.batch_size)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(self.batch_size)
        self.valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(self.batch_size)

    def mod_loss(y_true, y_pred):                       # Specific for regression task
        error = y_true - y_pred
        is_close = error <= 6
        return tf.where(is_close, error, mod_param-error)

#    def cyclic_loss(self, y_true, y_pred):
#        linear_errors = tf.math.abs(y_true - y_pred)
#        errors = tf.minimum(self.num_classes - linear_errors, linear_errors)
#        # mean_error = tf.reduce_mean(tf.math.pow(errors,3), axis=0)
#        mean_error = tf.reduce_mean(errors, axis=0)
#        return mean_error

    def mean_deviation_minutes(self, y_true, y_pred):
        "returns mean error in minutes"
        y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), tf.float32)
        y_true = tf.cast(tf.math.argmax(y_true, axis=-1), tf.float32)

        linear_errors = tf.math.abs(y_true - y_pred)
        errors = tf.minimum(self.num_classes - linear_errors, linear_errors)
        
        mean_error = tf.reduce_mean(errors, axis=0) / self.num_classes * 720
        return mean_error
    
    def train_model(self):
        self.model.compile(
                loss='mod_loss',                        # Comp categorical_crossentropy
                optimizer=keras.optimizers.Nadam(learning_rate=.001, beta_1=0.9, beta_2=0.999),
#                metrics=['accuracy', self.mean_deviation_minutes],              #######################################################
        )
        self.history = self.model.fit(self.train_dataset,
                                      epochs=self.num_epochs,
                                      validation_data=self.valid_dataset,
                                      batch_size=self.batch_size,
                                      )

myCNN = clock_CNN_regression(images, labels, num_classes=12, batch_size=128, num_epochs=20)
myCNN.train_model()
myCNN.model.evaluate(myCNN.test_dataset)
