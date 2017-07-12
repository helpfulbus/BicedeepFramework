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
from sklearn import model_selection
import time
import gc
import resource
import os
import math
import datetime as dt
from dateutil import parser
from dateutil.parser import parse
from Common import Logging

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.Session()


# Read given file from filepath and return
def readData(filePath):
    data_frame = pd.read_csv(filePath)
    return data_frame


def is_date(string):
    try:
        parse(string)
        return True
    except ValueError:
        return False


def convertToNumber (s):
    return int.from_bytes(s.encode(), 'little')


def convertFromNumber (n):
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()


def getModelNumber(number):
    strNumber = str(number)
    return ('00000' + strNumber)[len(strNumber):]


# Convert number to datetime, and string:
# data_df['Country']= data_df['Country'].map(dt.datetime.fromordinal)
# data_df['Country']= data_df['Country'].strftime('%m/%d/%Y')
def preprocess_data(df):
    string_columns = []
    date_columns = []
    data_length = len(df.columns)
    for i in range(data_length):
        col_name = df.columns[i]
        # Check if there is a column with string values
        if (isinstance(df.iloc[0, i], str)):
            # Check if string value is a date
            if (is_date(df.iloc[0, i])):
                df[col_name] = pd.to_datetime(df[col_name])
                df[col_name] = df[col_name].map(dt.datetime.toordinal)
                date_columns.append(col_name)
            else:
                df[col_name] = df[col_name].map(convertToNumber)
                string_columns.append(col_name)

    return string_columns, date_columns


#Return a categorical data from given data
def data_to_categorical(data, label_unique_values, number_of_unique_values):
    len_of_data = len(data)
    categorical_data = np.zeros((len_of_data, number_of_unique_values))
    dict = {}
    for i in range(0, len(label_unique_values)):
        dict[label_unique_values[i]] = i
    for i in range(0, len_of_data):
        categorical_data[i][dict[data[i]]] = 1
    return categorical_data


# Create a neural network architecture
def modelArchitectureRegression(input_dimension):
    with tf.device('/gpu:0'):
        model = Sequential()
        model.add(Dense(64, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, kernel_initializer='normal'))

        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model


def modelArchitectureClassification(input_dimension, output_dimension):
    with tf.device('/gpu:0'):
        model = Sequential()
        model.add(Dense(64, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(output_dimension, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


def create_report(file_path, file_name, desired_columns_as_label, reports_path, outputs_path):
    data_frame = readData(file_path)

    preprocess_start = time.time()
    Logging.Logging.write_log_to_file("Starting Pre Process")
    string_columns, date_columns = preprocess_data(data_frame)
    Logging.Logging.write_log_to_file("Pre Process Ended")

    preprocess_end = time.time()
    Logging.Logging.write_log_to_file("Pre process took %.2f seconds " % (preprocess_end - preprocess_start))

    ##############Feature Selection Part#############
    Logging.Logging.write_log_to_file('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    ft_sel_time_start = time.time()

    with tf.device('/gpu:0'):
        label_suggestion = {}
        label_mse_score = {}
        number_of_columns = len(desired_columns_as_label)
        early_end = False;
        for i in range(0, number_of_columns):
            if (early_end):
                break

            # child_pid = os.fork()
            # Logging.Logging.write_log_to_file("child_pid : " + str(child_pid))
            # if child_pid == 0:

            # copy data frame to make changes on copy
            data_frame_copy = data_frame.copy(deep=True)
            # get each column as label
            label_column_index = data_frame_copy.columns.get_loc(desired_columns_as_label[i])
            label_column = data_frame_copy.iloc[:, label_column_index]
            # decide whether regression or classification will be applied to the column
            classification = False
            column_unique_values = label_column.unique()
            number_of_unique_values = len(label_column.unique())
            number_of_samples = len(label_column.values)
            if ((number_of_unique_values / number_of_samples) < 0.2):
                classification = True
            label_column_name = data_frame_copy.columns[label_column_index]
            # drop label column from data frame
            data_frame_copy.drop(data_frame_copy.columns[[label_column_index]], axis=1, inplace=True)
            # add label column as last column to the data_frame
            label_column_series = pd.Series(label_column.values, name=label_column_name)
            data_frame_copy = pd.concat([data_frame_copy, label_column_series], axis=1)

            best_minimumMSEValue = float('inf')  # value to keep min mse value
            minMSEFeatureList = []  # value to keep list of feature where min mse occurs

            while (len(data_frame_copy.columns) > 2):
                if (early_end):
                    break

                number_of_features = len(data_frame_copy.columns) - 1

                modelClass = modelArchitectureClassification(number_of_features - 1, number_of_unique_values)
                modelReg = modelArchitectureRegression(number_of_features - 1)

                remove_feature_index = -1
                local_minimumMSEValue = float('inf')
                for j in range(-1, number_of_features):
                    if (early_end):
                        break

                    Logging.Logging.write_log_to_file('Column Start 1 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                    # convert data_frame_copy into numpy matrix
                    dataset = data_frame_copy.values
                    # get names of features
                    feature_set_column_names = data_frame_copy.columns[0:number_of_features].values
                    # at each turn remove one of the features

                    if (j >= 0):
                        dataset = np.delete(dataset, [j], axis=1)
                        feature_set_column_names = np.delete(feature_set_column_names, [j])
                    # split data as X: features, Y: labels
                    X = dataset[:, 0: len(dataset[0]) - 1]
                    Y = dataset[:, len(dataset[0]) - 1]
                    # evaluate the model
                    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

                    # decide to model based on whether it is classification or not
                    if (classification):
                        if (j >= 0):
                            model = modelClass
                        else:
                            model = modelArchitectureClassification(number_of_features, number_of_unique_values)
                        model.reset_states()

                        Logging.Logging.write_log_to_file(
                            'Column Start 2 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                        y_train = data_to_categorical(y_train, column_unique_values, number_of_unique_values)
                        model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=0)

                        y_test = data_to_categorical(y_test, column_unique_values, number_of_unique_values)

                        accuracy = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
                        Logging.Logging.write_log_to_file("Loss: {}, Accuracy: {} ".format(accuracy[0], accuracy[1]))
                        Logging.Logging.write_log_to_file(feature_set_column_names)
                        Logging.Logging.write_log_to_file('Column Start 3 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


                        if accuracy[0] <= local_minimumMSEValue:
                            local_minimumMSEValue = accuracy[0]
                            remove_feature_index = j
                            Logging.Logging.write_log_to_file('local')
                            if local_minimumMSEValue <= best_minimumMSEValue:
                                Logging.Logging.write_log_to_file('best')
                                best_minimumMSEValue = local_minimumMSEValue
                                minMSEFeatureList = feature_set_column_names
                                save_model = model
                                modelClass = modelArchitectureClassification(number_of_features - 1,
                                                                             number_of_unique_values)

                    else:
                        if (j >= 0):
                            model = modelReg
                        else:
                            model = modelArchitectureRegression(number_of_features)
                        model.reset_states()

                        Logging.Logging.write_log_to_file(
                            'Column Start 2 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                        model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=0)

                        loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
                        Logging.Logging.write_log_to_file(loss_and_metrics)
                        Logging.Logging.write_log_to_file(feature_set_column_names)
                        Logging.Logging.write_log_to_file('Column Start 3 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                        if loss_and_metrics <= local_minimumMSEValue:
                            local_minimumMSEValue = loss_and_metrics
                            remove_feature_index = j
                            Logging.Logging.write_log_to_file('local')
                            if local_minimumMSEValue <= best_minimumMSEValue:
                                Logging.Logging.write_log_to_file('best')
                                best_minimumMSEValue = local_minimumMSEValue
                                minMSEFeatureList = feature_set_column_names
                                save_model = model
                                modelReg = modelArchitectureRegression(number_of_features - 1)

                if (remove_feature_index >= 0):
                    Logging.Logging.write_log_to_file('removed feature: {}'.format(remove_feature_index))
                    data_frame_copy.drop(data_frame_copy.columns[[remove_feature_index]], axis=1, inplace=True)

            # add to dict best mse and best feature list, save the model
            label_suggestion[label_column_name] = minMSEFeatureList
            label_mse_score[label_column_name] = best_minimumMSEValue
            model_save_file_number = getModelNumber(i + 1)
            model_save_file_name = outputs_path + "/" + file_name + "." + model_save_file_number + ".h5"
            Logging.Logging.write_log_to_file(len(save_model.layers))
            save_model.save(model_save_file_name)

            Logging.Logging.write_log_to_file(label_suggestion)
            Logging.Logging.write_log_to_file("End")
            ft_sel_time_end = time.time()
            Logging.Logging.write_log_to_file("Feature Selection took %.2f seconds with gpu" % (ft_sel_time_end - ft_sel_time_start))
            Logging.Logging.write_log_to_file('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            data_frame_copy = None
            # get each column as label
            label_column = None
            gc.collect()
            Logging.Logging.write_log_to_file("gc")
            Logging.Logging.write_log_to_file('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            # pid,status = os.waitpid(child_pid,0)

