#%%
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras.layers as kl
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras import backend as K

# DATASET_PREFIX = 'drive/MyDrive/idl/datasets/'
# DATASET_PREFIX = 'clock_data/'

# DATASET_PREFIX = '/kaggle/input/clock-images/'
DATASET_PREFIX = 'clock_75/'
labels = np.load(DATASET_PREFIX + 'labels.npy')
images = np.load(DATASET_PREFIX + 'images.npy')

#%%
class ClockMultihead:
    FILENAME_LABELS = DATASET_PREFIX + 'labels.npy'
    FILENAME_IMAGES = DATASET_PREFIX + 'images.npy'
    def __init__(self,
                batch_size=128,
                num_epochs=20,
                ):
        self.batch_size = batch_size
        self.test_fraction = .2
        self.num_epochs = num_epochs
        self.image_shape = (150, 150, 1)

        self.build_model()

    def build_model(self):
        input_ = kl.Input(shape=self.image_shape)
        x = kl.Conv2D(64, kernel_size=4, strides=2, activation='relu')(input_)
        x = kl.Conv2D(64, kernel_size=4, strides=2, activation='relu')(x)
        x = kl.MaxPooling2D(pool_size=4, strides=2)(x)
        x = kl.BatchNormalization()(x)
        
        x = kl.Conv2D(64, kernel_size=2, strides=1, activation='relu')(x)
        x = kl.Conv2D(64, kernel_size=2, strides=1, activation='relu')(x)
        x = kl.MaxPooling2D(pool_size=2, strides=2)(x)
        x = kl.BatchNormalization()(x)
            
        flatten = kl.Flatten()(x)

        # hours bit:
        x = kl.Dropout(0.5)(flatten)
        x = kl.Dense(360, activation='relu')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.5)(x)
        x = kl.Dense(360, activation='relu')(x)
        x = kl.BatchNormalization()(x)
        h_output = kl.Dense(12, activation='softmax', name='hours')(x)

        # minutes bit:
        x = kl.Dropout(0.3)(flatten)
        x = kl.Dense(360, activation='relu')(x)
        x = kl.Dense(360, activation='relu')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.3)(x)
        x = kl.Dense(360, activation='relu')(x)
        x = kl.Dense(360, activation='relu')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.3)(x)
        x = kl.Dense(360, activation='relu')(x)
        x = kl.Dense(360, activation='relu')(x)
        x = kl.BatchNormalization()(x)
        m_output = kl.Dense(1, activation='linear', name='minutes')(x)

        self.model = keras.Model(inputs=[input_], outputs=[h_output, m_output])
        self.model.summary()
            
    @staticmethod
    def read_dataset():
        return np.load(ClockMultihead.FILENAME_IMAGES), np.load(ClockMultihead.FILENAME_LABELS)
    
    @staticmethod
    def labels_to_float(labels):
        """
        Converts tuple labels to integers for classification
        """
        labels_base_720 = labels[:,0]*60 + labels[:,1]
        return tf.cast(labels_base_720, tf.float32)
    
    def shuffle_data(self):
        images, labels = ClockMultihead.read_dataset()
        rng = np.random.default_rng(seed=10)
        indices = rng.permutation(len(labels))
        
        labs = labels[indices]
        
        ims = images.reshape(len(images),*self.image_shape)/255
        ims = ims[indices]
        
        return ims, labs
    
    @staticmethod
    def labels_to_hrs_mins(labels):
        """
        returns hours and minutes as separate labels (maybe doesn't have to be a function lol)
        """
        return labels[:,0], labels[:,1].astype('float32')
    
    @staticmethod
    def hrs_mins_to_base720(hrs, mins):
        return hrs*12 + mins
    
    def prep_data(self):
        images, labels = self.shuffle_data()
        hours_full, minutes_full = ClockMultihead.labels_to_hrs_mins(labels)

        num_testing = int(self.test_fraction * len(hours_full))
        
        test_hours, test_minutes, test_images = hours_full[:num_testing], minutes_full[:num_testing], images[:num_testing]
        train_hours, train_minutes, train_images = hours_full[num_testing:], minutes_full[num_testing:], images[num_testing:]

        return (train_images, [train_hours, train_minutes]), (test_images, [test_hours, test_minutes])
    
    def common_sense_accuracy(self):
        _, (test_images, test_labels) = self.prep_data()
        hrs_pred, mins_pred = self.model.predict(test_images)
        hrs_pred = np.argmax(hrs_pred, axis=-1)
        time_predict = (hrs_pred * 12 + mins_pred.T).reshape((-1))
        time_true = test_labels[0] * 12 + test_labels[1]

        linear_error = tf.math.abs(time_predict - time_true)
        cyclic_error = tf.math.minimum(linear_error, 720-linear_error)
        return cyclic_error.numpy()


    def train_model(self):
        trainset, testset = self.prep_data()
        
        batches_per_epoch = len(trainset[0])//self.batch_size
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=2e-4,
            decay_steps=20 * batches_per_epoch,
            decay_rate=0.5)

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                          monitor='loss',
                                                          restore_best_weights=True)
        self.model.compile(
            loss=['sparse_categorical_crossentropy', 'mae'], 
            loss_weights=[.99, .01], 
            optimizer=keras.optimizers.Nadam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999),
#             optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.1),
            )

        self.history = self.model.fit(*trainset,
                                      epochs=self.num_epochs,
                                      validation_data=testset,
                                      batch_size=self.batch_size,
                                      callbacks=[early_stopping_cb],
                                      )
#%%

myCNN = clock_CNN_multihead(images, labels, batch_size=128, num_epochs=2)
myCNN.train_model()