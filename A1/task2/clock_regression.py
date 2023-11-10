import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras.layers as kl
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras import backend as K
import pickle as pk

# DATASET_PREFIX = 'drive/MyDrive/idl/datasets/'
DATASET_PREFIX = '/kaggle/input/clock-images/'


class ClockRegressor:
    FILENAME_LABELS = DATASET_PREFIX + 'labels.npy'
    FILENAME_IMAGES = DATASET_PREFIX + 'images.npy'
    def __init__(self,
                batch_size=128,
                num_epochs=20,
                test_fraction=.2,
                ):
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.image_shape = (150, 150, 1)
        
        self.build_model()

    @staticmethod
    def read_dataset():
        return np.load(ClockRegressor.FILENAME_IMAGES), np.load(ClockRegressor.FILENAME_LABELS)
    
    def build_model(self):
        self.model = keras.models.Sequential([        
            Conv2D(64, kernel_size=4, strides=2, activation='relu', input_shape=self.image_shape),
            Conv2D(64, kernel_size=4, strides=2, activation='relu'),
            MaxPooling2D(pool_size=4, strides=2),
            BatchNormalization(),
            Conv2D(64, kernel_size=2, strides=1, activation='relu'),
            Conv2D(64, kernel_size=2, strides=1, activation='relu'),
            MaxPooling2D(pool_size=2, strides=2),
            BatchNormalization(),
            Flatten(),
            Dropout(0.3),
            Dense(360, activation='relu'),
            Dense(360, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(360, activation='relu'),
            Dense(360, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(360, activation='relu'),
            Dense(360, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='linear'),
        ])

        self.model.summary()
        
    @staticmethod
    def labels_to_float(labels):
        """
        Converts tuple labels to integers for classification
        """
        labels_base_720 = labels[:,0]*60 + labels[:,1]
        return tf.cast(labels_base_720, tf.float32)
    
    def shuffle_data(self):
        images, labels = ClockRegressor.read_dataset()
        rng = np.random.default_rng(seed=10)
        indices = rng.permutation(len(labels))
        
        labs = labels[indices]
        
        ims = images.reshape(len(images),*self.image_shape)/255
        ims = ims[indices]
        
        return ims, labs
    
    def make_datasets(self):
        images, labels = self.shuffle_data()
        labels_float = ClockRegressor.labels_to_float(labels)
        
        full_dataset = tf.data.Dataset.from_tensor_slices((images, labels_float)).batch(self.batch_size)
        train_dataset, test_dataset = keras.utils.split_dataset(
                                        full_dataset, 
                                        right_size=self.test_fraction, 
                                        shuffle=False,
                                        )

        return train_dataset, test_dataset
    
    def testset_for_test_acc(self):
        images, labels = self.shuffle_data()
        labels_base720 = ClockRegressor.labels_to_float(labels)
        
        num_test = int(self.test_fraction * len(labels_base720))
        
        test_labels = labels_base720[-num_test:]
        test_images = images[-num_test:]
        
        return test_images, test_labels   
        
    @property
    def final_test_accuracy(self):
        test_images, test_labels = self.testset_for_test_acc()
        return self.model.evaluate(test_images, test_labels)[1]
        
    @property
    def num_steps(self):
        train_dataset, test_dataset = self.read_dataset()
        return len(train_dataset)//self.batch_size
        
    def final_common_sense_accuracy(self):
        """prints/returns the mean deviation in minutes of the final model on the test set"""
        test_images, y_true = self.testset_for_test_acc()
        
        output = self.model.predict(test_images)
        y_pred = tf.math.mod(output, 720)
#         y_pred = (tf.cast(tf.math.argmax(output, axis=1), tf.float64) + .5) * 720/self.num_classes
        
        linear_error = tf.math.abs(y_true - y_pred)
        cyclic_error = tf.cast(tf.minimum(720 - linear_error, linear_error), tf.float32)
        mean_error = tf.reduce_mean(cyclic_error)
        
        print(mean_error)
        plt.hist(cyclic_error)
        plt.show()
        
        return mean_error
    
    def common_sense_error(self, y_true, y_pred):
        y_pred = tf.math.mod(y_pred, 720)
        
        linear_error = tf.math.abs(y_true - y_pred)
        cyclic_error = tf.cast(tf.minimum(720 - linear_error, linear_error), tf.float32)

        return cyclic_error
    
    def common_sense_accuracy(self, y_true, y_pred):
        error = self.common_sense_error(y_true, y_pred)
        return tf.reduce_mean(error)
        
    def train_model(self):
        train_dataset, test_dataset = self.make_datasets()
        
        batches_per_epoch = len(list(train_dataset.as_numpy_iterator()))
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=2e-4,
            decay_steps=20*batches_per_epoch,
            decay_rate=0.5)
        
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                          monitor='loss',
                                                          restore_best_weights=True)
        
        self.model.compile(
                loss='mae',
                optimizer=keras.optimizers.Nadam(learning_rate=lr_schedule, 
                                                 beta_1=0.9, beta_2=0.999),
                metrics=[self.common_sense_accuracy],
        )
        
        self.history = self.model.fit(
                train_dataset,
                epochs=self.num_epochs,
                validation_data=test_dataset,
                batch_size=self.batch_size,
                callbacks=[early_stopping_cb],
        )
myCNN = ClockRegressor(batch_size=32, num_epochs=2)
myCNN.train_model()
myCNN.model.evaluate(myCNN.test_dataset)