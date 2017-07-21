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
import resource
import os
import math
import datetime as dt
import json
from collections import OrderedDict
from dateutil import parser
from dateutil.parser import parse
from Common import Logging

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=128))

class LastUpdatedOrderedDict(OrderedDict):
    'Store items in the order the keys were last added'

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)

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


def get_nearest_power_of_2(number):
    return math.ceil(2**math.ceil(math.log(number, 2)))

def getModelNumber(number):
    strNumber = str(number)
    return ('00000' + strNumber)[len(strNumber):]

def create_report_file(report_path, file_name, label_mse_score, label_suggestion):
    part_predictability = []
    file_counter = 1
    for key, value in label_mse_score.items():
        data = {}
        data["id"] = file_counter
        data["part_name"] = key
        data["predictability"] = value[0]
        data["classification"] = value[1]
        if value[1]:
            data["accuracy"] = value[2]
        part_predictability.append(data)
        file_counter += 1

    suggestions = []
    file_counter = 1
    for key, value in label_suggestion.items():
        data = {}
        data["id"] = file_counter
        data["part"] = key
        if isinstance(value, list):
            data["using"] = value
        else:
            data["using"] = value.tolist()
        suggestions.append(data)
        file_counter += 1

    json_file = {}
    json_file["part_predictability"] = part_predictability
    json_file["suggestions"] = suggestions

    with open(report_path + "/" + file_name + ".json", 'w') as outfile:
        json.dump(json_file, outfile)

def create_details_file(output_path, file_name, label_suggestion, label_types, label_category_dict):
    # Generate details
    file_counter = 1
    for key, value in label_suggestion.items():
        suggestions = []
        data = {}
        data["id"] = file_counter
        data["part"] = key
        data["using"] = value.tolist()
        data["type"] = label_types[key]

        if len(label_category_dict[key]) > 0:
            category_data = {}
            for category_key, category_value in label_category_dict[key].items():
                category_data[str(category_key)] = str(category_value)
            data["categories"] = category_data

        suggestions.append(data)
        json_file = {}
        json_file["suggestions"] = suggestions
        filename = output_path + "/" + file_name + "." + getModelNumber(file_counter) + ".h5.json"
        with open(filename, 'w') as outfile:
            json.dump(json_file, outfile)
        file_counter += 1

# Convert number to datetime, and string:
# data_df['Country']= data_df['Country'].map(dt.datetime.fromordinal)
# data_df['Country']= data_df['Country'].strftime('%m/%d/%Y')
def preprocess_data(df):
    string_columns = []
    date_columns = []
    big_number_columns = []
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
        else:
            if df.iloc[0, i] > df[col_name].mean() > 10000:
                big_number_columns.append(col_name)

    return string_columns, date_columns, big_number_columns


#Return a categorical data from given data
def data_to_categorical(data, label_unique_values, number_of_unique_values, dict = {}):
    len_of_data = len(data)
    categorical_data = np.zeros((len_of_data, number_of_unique_values))
    if(len(dict) == 0):
        for i in range(0, len(label_unique_values)):
            dict[label_unique_values[i]] = i
    for i in range(0, len_of_data):
        categorical_data[i][dict[data[i]]] = 1
    return categorical_data, dict


# Create a neural network architecture
def modelArchitectureRegression(input_dimension, optimizer_input):
    with tf.device('/cpu:0'):
        model = Sequential()
        model.add(Dense(64, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, kernel_initializer='normal'))

        model.compile(loss='mean_absolute_error', optimizer=optimizer_input)
        return model


def modelArchitectureClassification(input_dimension, output_dimension, optimizer_input):
    with tf.device('/cpu:0'):
        model = Sequential()
        model.add(Dense(64, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(output_dimension, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=optimizer_input, metrics=['accuracy'])
        return model


def create_report(file_path, file_name, desired_columns_as_label, reports_path, outputs_path):
    data_frame = readData(file_path)

    preprocess_start = time.time()
    Logging.Logging.write_log_to_file_selectable("Starting Pre Process")
    string_columns, date_columns, big_number_columns = preprocess_data(data_frame)
    Logging.Logging.write_log_to_file_selectable("Pre Process Ended")

    preprocess_end = time.time()
    Logging.Logging.write_log_to_file_selectable("Pre process took %.2f seconds " % (preprocess_end - preprocess_start))

    ##############Feature Selection Part#############
    Logging.Logging.write_log_to_file_selectable('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    ft_sel_time_start = time.time()

    with tf.device('/cpu:0'):
        label_suggestion = LastUpdatedOrderedDict()
        label_types = LastUpdatedOrderedDict()
        label_mse_score = LastUpdatedOrderedDict()
        label_category_dict = LastUpdatedOrderedDict()
        number_of_columns = len(desired_columns_as_label)
        for i in range(0, number_of_columns):
            early_end = False;

            # child_pid = os.fork()
            # Logging.Logging.write_log_to_file_selectable("child_pid : " + str(child_pid))
            # if child_pid == 0:

            # copy data frame to make changes on copy
            data_frame_copy = data_frame.copy(deep=True)
            # get each column as label
            label_column_index = data_frame_copy.columns.get_loc(desired_columns_as_label[i])
            label_column = data_frame_copy.iloc[:, label_column_index]
            # decide whether regression or classification will be applied to the column
            classification = False
            column_unique_values = label_column.unique()
            number_of_unique_values = len(column_unique_values)
            number_of_samples = len(label_column.values)

            if ((number_of_unique_values / number_of_samples) < 0.2):
                classification = True
            label_column_name = data_frame_copy.columns[label_column_index]

            categorical_dict = {}
            preset_batch_size = get_nearest_power_of_2(number_of_samples / 100);
            preset_optimizer = "adam"
            #if the values are big numbers, increase learning rate reduce batch size
            if(label_column_name in date_columns or label_column_name in big_number_columns):
                preset_batch_size = 2;
                preset_optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.01)


            # drop label column from data frame
            data_frame_copy.drop(data_frame_copy.columns[[label_column_index]], axis=1, inplace=True)
            # add label column as last column to the data_frame
            label_column_series = pd.Series(label_column.values, name=label_column_name)
            data_frame_copy = pd.concat([data_frame_copy, label_column_series], axis=1)

            best_minimumMSEValue = float('inf')  # value to keep min mse value
            minMSEFeatureList = []  # value to keep list of feature where min mse occurs

            isFirstTime = True
            while (len(data_frame_copy.columns) >= 2):
                if (early_end):
                    break

                number_of_features = len(data_frame_copy.columns) - 1

                if(number_of_features - 1 > 0):
                    modelClass = modelArchitectureClassification(number_of_features - 1, number_of_unique_values, preset_optimizer)
                    modelReg = modelArchitectureRegression(number_of_features - 1, preset_optimizer)

                remove_feature_index = -1
                local_minimumMSEValue = float('inf')

                if (isFirstTime):
                    loop_start = -1
                    isFirstTime = False
                else:
                    loop_start = 0
                for j in range(loop_start, number_of_features):
                    if (early_end):
                        break

                    Logging.Logging.write_log_to_file_selectable('Column Start 1 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                    # convert data_frame_copy into numpy matrix
                    dataset = data_frame_copy.values
                    # get names of features
                    feature_set_column_names = data_frame_copy.columns[0:number_of_features].values
                    # at each turn remove one of the features

                    if (j >= 0):
                        dataset = np.delete(dataset, [j], axis=1)
                        feature_set_column_names = np.delete(feature_set_column_names, [j])
                        if(dataset.shape[1] < 2):
                            early_end = True
                            break
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
                            model = modelArchitectureClassification(number_of_features, number_of_unique_values, preset_optimizer)
                        model.reset_states()

                        Logging.Logging.write_log_to_file_selectable(
                            'Column Start 2 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                        y_train, categorical_dict = data_to_categorical(y_train, column_unique_values, number_of_unique_values, categorical_dict)

                        model.fit(x_train, y_train, epochs=10, batch_size=preset_batch_size, verbose=0)

                        y_test, categorical_dict = data_to_categorical(y_test, column_unique_values, number_of_unique_values, categorical_dict)
                        accuracy = model.evaluate(x_test, y_test, batch_size=preset_batch_size, verbose=0)

                        Logging.Logging.write_log_to_file_selectable("Loss: {}, Accuracy: {} ".format(accuracy[0], accuracy[1]))
                        Logging.Logging.write_log_to_file_selectable(feature_set_column_names)
                        Logging.Logging.write_log_to_file_selectable('Column Start 3 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


                        if accuracy[0] <= local_minimumMSEValue:
                            local_minimumMSEValue = accuracy[0]
                            remove_feature_index = j
                            Logging.Logging.write_log_to_file_selectable('local')
                            if local_minimumMSEValue <= best_minimumMSEValue:
                                Logging.Logging.write_log_to_file_selectable('best')
                                best_minimumMSEValue = local_minimumMSEValue
                                best_acc = accuracy[1]
                                minMSEFeatureList = feature_set_column_names
                                save_model = model

                                if (number_of_features - 1 > 0):
                                    modelClass = modelArchitectureClassification(number_of_features - 1,
                                                                             number_of_unique_values, preset_optimizer)

                    else:
                        if (j >= 0):
                            model = modelReg
                        else:
                            model = modelArchitectureRegression(number_of_features, preset_optimizer)
                        model.reset_states()

                        Logging.Logging.write_log_to_file_selectable(
                            'Column Start 2 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                        model.fit(x_train, y_train, epochs=15, batch_size=preset_batch_size, verbose=0)

                        loss_and_metrics = model.evaluate(x_test, y_test, batch_size=preset_batch_size, verbose=0)

                        Logging.Logging.write_log_to_file_selectable(loss_and_metrics)
                        Logging.Logging.write_log_to_file_selectable(feature_set_column_names)
                        Logging.Logging.write_log_to_file_selectable('Column Start 3 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                        if loss_and_metrics <= local_minimumMSEValue:
                            local_minimumMSEValue = loss_and_metrics
                            remove_feature_index = j
                            Logging.Logging.write_log_to_file_selectable('local')
                            if local_minimumMSEValue <= best_minimumMSEValue:
                                Logging.Logging.write_log_to_file_selectable('best')
                                best_minimumMSEValue = local_minimumMSEValue
                                minMSEFeatureList = feature_set_column_names
                                save_model = model

                                if (number_of_features - 1 > 0):
                                    modelReg = modelArchitectureRegression(number_of_features - 1, preset_optimizer)

                if (remove_feature_index >= 0):
                    Logging.Logging.write_log_to_file_selectable('removed feature: {}'.format(remove_feature_index))
                    data_frame_copy.drop(data_frame_copy.columns[[remove_feature_index]], axis=1, inplace=True)

            # add to dict best mse and best feature list, save the model
            label_suggestion[label_column_name] = minMSEFeatureList

            label_types[label_column_name] = "number"
            if label_column_name in string_columns:
                label_types[label_column_name] = "string"
            elif label_column_name in date_columns:
                label_types[label_column_name] = "date"

            if classification:
                label_mse_score[label_column_name] = (best_minimumMSEValue, classification, best_acc)
                label_category_dict[label_column_name] = categorical_dict
            else:
                label_mse_score[label_column_name] = (best_minimumMSEValue, classification)
                label_category_dict[label_column_name] = {}
            model_save_file_number = getModelNumber(i + 1)
            model_save_file_name = outputs_path + "/" + file_name + "." + model_save_file_number + ".h5"
            save_model.save(model_save_file_name)

            Logging.Logging.write_log_to_file_selectable(label_suggestion)
            Logging.Logging.write_log_to_file_selectable("End")
            ft_sel_time_end = time.time()
            Logging.Logging.write_log_to_file_selectable("Feature Selection took %.2f seconds with gpu" % (ft_sel_time_end - ft_sel_time_start))
            Logging.Logging.write_log_to_file_selectable('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            # pid,status = os.waitpid(child_pid,0)


    create_report_file(reports_path, file_name, label_mse_score, label_suggestion)
    create_details_file(outputs_path, file_name, label_suggestion, label_types, label_category_dict)

    Logging.Logging.write_log_to_file_selectable_flush()
