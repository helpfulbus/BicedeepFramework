import keras
import pandas as pd
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation
import time
from Common import Logging

import tensorflow as tf

sess = tf.Session()


# Read given file from filepath and return
def readData(filePath):
    data_frame = pd.read_csv(filePath)
    return data_frame


# Create a neural network architecture
def modelArchitectureRegression(input_dimension):
    with tf.device('/gpu:0'):
        model = Sequential()
        model.add(Dense(64, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, kernel_initializer='normal'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


def report_test(filePath):
    data_frame = readData(filePath)
    number_of_features = len(data_frame.columns) - 1
    dataset = data_frame.values
    # split data as X: features, Y: labels
    X = dataset[:, 0: number_of_features]
    Y = dataset[:, number_of_features]
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=42)
    model = modelArchitectureRegression(number_of_features)
    model.fit(x_train, y_train, nb_epoch=20, batch_size=128, verbose=0)
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    Logging.Logging.write_log_to_file(str(loss_and_metrics))
