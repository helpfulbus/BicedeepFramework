# <copyright company="Bicedeep, Inc.">
# Copyright (c) 2016-2018 All Rights Reserved
# </copyright>

import keras
import pandas as pd
import numpy as np
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
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
import heapq
import copy
import locale
from collections import OrderedDict
from dateutil import parser
from dateutil.parser import parse
from decimal import *
from Common import Logging
from Common import GoogleStorage
from tensorflow.python.client import device_lib
#from keras.utils.training_utils import multi_gpu_model

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=128, intra_op_parallelism_threads=128))

class LastUpdatedOrderedDict(OrderedDict):
    'Store items in the order the keys were last added'

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)

# Read given file from filepath and return
def readData(filePath):
    try:
        data_frame = pd.read_csv(filePath)
    except:
        data_frame = pd.read_csv(filePath, encoding="latin_1")
    return data_frame

def is_date(string):
    if (pd.isnull(string)):
        return True
    try:
        parse(string)
        return True
    except:
        return False

def is_number(input):
    if (pd.isnull(input)):
        return True
    try:
        int(input.replace(',', ''))
        return True
    except:
        return False

def to_comma_int(input):
    try:
        return int(input.replace(',', ''))
    except:
        return 0


def insert_second_dot(input):
    return input[:1] + "." + input[1:]


def convertToNumber(s):
    int_value = int.from_bytes(str(s).encode(), 'big')
    return Decimal(insert_second_dot(str(int_value)))


def convertFromNumber (n):
    if(n == Decimal("nan")):
        return "NAN"
    int_value = int(str(n).replace(".", ""))
    return int_value.to_bytes(math.ceil(int_value.bit_length() / 8), 'big').decode(errors='ignore')


def get_nearest_power_of_2(number):
    return math.ceil(2**math.ceil(math.log(number, 2)))

def getModelNumber(number):
    strNumber = str(number)
    return ('00000' + strNumber)[len(strNumber):]

def is_number_classifiction(number_of_unique_values, number_of_samples):
    if((number_of_unique_values / number_of_samples) < 0.01 and number_of_unique_values < 256 and number_of_samples * number_of_unique_values < 100000000):
        return True
    else:
        return False

def is_string_classifiction(number_of_unique_values, number_of_samples):
    if(number_of_samples * number_of_unique_values < 100000000):
        return True
    else:
        return False

def get_available_gpu_number():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

#def make_model_multigpu(model):
#    gpuNum = get_available_gpu_number()
#    if (gpuNum >= 2):
#        model = multi_gpu_model(model, gpus=gpuNum)
#    return model

def create_report_file(report_path, file_name, label_mse_score, label_suggestion, label_types, label_mse_score_best_epochs):
    part_predictability_best_epochs = []
    file_counter = 1
    for key, value in label_mse_score_best_epochs.items():
        data = {}
        data["id"] = file_counter
        data["part_name"] = key
        data["predictability"] = value[0]

        data["classification"] = value[1]
        if value[1]:
            data["accuracy"] = value[2]
        part_predictability_best_epochs.append(data)
        file_counter += 1

    part_predictability = []
    file_counter = 1
    for key, value in label_mse_score.items():
        data = {}
        data["id"] = file_counter
        data["part_name"] = key
        data["type"] = label_types[key]
        data["predictability"] = value[0]

        #if(data["type"] == "string"):
            #data["predictability_conv"] = convertFromNumber(Decimal(data["predictability"]))
        if(data["type"] == "date"):
            data["predictability_conv"] = str(dt.datetime.fromordinal(int(round(data["predictability"]))))

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
    json_file["part_predictability_best_epochs"] = part_predictability_best_epochs
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

def is_string(inp):
    if (pd.isnull(inp)):
        return True
    return isinstance(inp, str)

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

        is_string_column = df[col_name].apply(lambda x: is_string(x)).all()
        is_number_column = df[col_name].apply(lambda x: is_number(x)).all()

        if (is_string_column and not is_number_column):

            is_date_column = df[col_name].apply(lambda x: is_date(x)).all()
            # Check if string value is a date
            if (is_date_column):
                df[col_name] = df[col_name].fillna("01-01-2000")
                df[col_name] = pd.to_datetime(df[col_name])
                df[col_name] = df[col_name].map(dt.datetime.toordinal)
                date_columns.append(col_name)
            else:
                df[col_name] = df[col_name].fillna("nan")
                df[col_name] = df[col_name].map(convertToNumber)
                string_columns.append(col_name)
        elif is_number_column:
            df[col_name] = df[col_name].map(to_comma_int)
            try:
                if df[col_name].mean() > 100000:
                    big_number_columns.append(col_name)
            except:
                df[col_name] = df[col_name].map(convertToNumber)
                string_columns.append(col_name)
        else:
            df[col_name] = df[col_name].fillna(0)
            try:
                if df[col_name].mean() > 100000:
                    big_number_columns.append(col_name)
            except:
                df[col_name] = df[col_name].map(convertToNumber)
                string_columns.append(col_name)

    return string_columns, date_columns, big_number_columns

def get_categorical_dict(label_unique_values):
    cate_dict = {}
    for i in range(0, len(label_unique_values)):
        cate_dict[label_unique_values[i]] = i
    return cate_dict

#Return a categorical data from given data
def data_to_categorical(data, label_unique_values, number_of_unique_values, data_to_cate_dict):

    if(number_of_unique_values == 1):
        number_of_unique_values = 2
        np.append(label_unique_values, label_unique_values[0] + 1)

    len_of_data = len(data)
    categorical_data = np.zeros((len_of_data, number_of_unique_values))
    for i in range(0, len_of_data):
        try:
            categorical_data[i][data_to_cate_dict[data[i]]] = 1
        except:
            print("error")
    return categorical_data


# Create a neural network architecture
def modelArchitectureRegression(input_dimension, optimizer_input):
    with tf.device('/cpu:0'):
        model = Sequential()
        model.add(Dense(64, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, kernel_initializer='normal'))

        #model = make_model_multigpu(model)

        model.compile(loss='mean_absolute_error', optimizer=optimizer_input)
        return model


def modelArchitectureClassification(input_dimension, output_dimension, optimizer_input):
    with tf.device('/cpu:0'):
        if(output_dimension < 2):
            output_dimension = 2
        model = Sequential()
        model.add(Dense(64, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(output_dimension, activation='softmax'))

        #model = make_model_multigpu(model)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer_input, metrics=['accuracy'])
        return model


def create_report(file_path, file_name, desired_columns_as_label, reports_path, outputs_path, email, file_name_full):

    status = 0.0 / 100.0
    data_frame = readData(file_path)
    Logging.Logging.write_log_to_file_status(str(status))

    preprocess_start = time.time()
    Logging.Logging.write_log_to_file_selectable("Starting Pre Process")
    string_columns, date_columns, big_number_columns = preprocess_data(data_frame)
    Logging.Logging.write_log_to_file_selectable("Pre Process Ended")

    preprocess_end = time.time()

    status = 2.0 / 100.0

    Logging.Logging.write_log_to_file_selectable("Pre process took %.2f seconds " % (preprocess_end - preprocess_start))
    Logging.Logging.write_log_to_file_status(str(status))
    Logging.Logging.write_log_to_file_status_flush()

    ##############Feature Selection Part#############
    Logging.Logging.write_log_to_file_selectable('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    Logging.Logging.write_log_to_file_selectable_flush()

    try:
        GoogleStorage.upload_logs(email, file_name_full)
    except Exception as e:
        Logging.Logging.write_log_to_file_selectable("Logs Upload Failed : " + str(e))

    ft_sel_time_start = time.time()

    with tf.device('/cpu:0'):
        label_suggestion = LastUpdatedOrderedDict()
        label_types = LastUpdatedOrderedDict()
        label_mse_score = LastUpdatedOrderedDict()
        label_mse_score_best_epochs = LastUpdatedOrderedDict()
        label_category_dict = LastUpdatedOrderedDict()
        number_of_columns = len(desired_columns_as_label)
        desired_columns_as_label = list(map((lambda x: x.strip("\n ,")), desired_columns_as_label))

        status_column_increase = (78.0 / number_of_columns) / 100

        for i in range(0, number_of_columns):
            early_end = False;

            # copy data frame to make changes on copy
            data_frame_copy = data_frame.copy(deep=True)
            # get each column as label

            dfc_columns = list(map((lambda x: x.strip("\n ,")), data_frame_copy.columns))

            label_column_index = dfc_columns.index(desired_columns_as_label[i])
            label_column = data_frame_copy.iloc[:, label_column_index]
            # decide whether regression or classification will be applied to the column
            classification = False
            column_unique_values = label_column.unique()
            number_of_unique_values = len(column_unique_values)
            number_of_samples = len(label_column.values)

            label_column_name = data_frame_copy.columns[label_column_index]

            classification = is_number_classifiction(number_of_unique_values, number_of_samples)

            if (label_column_name in string_columns or label_column_name in date_columns):
                classification = is_string_classifiction(number_of_unique_values, number_of_samples)

            if(classification):
                categorical_dict = get_categorical_dict(column_unique_values)

            preset_batch_size = get_nearest_power_of_2(number_of_samples / 100);
            preset_optimizer = "adam"
            #if the values are big numbers, increase learning rate reduce batch size
            if(label_column_name in big_number_columns):
                preset_batch_size = 2;
                preset_optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.01)


            # drop label column from data frame
            data_frame_copy.drop(data_frame_copy.columns[[label_column_index]], axis=1, inplace=True)
            # add label column as last column to the data_frame
            label_column_series = pd.Series(label_column.values, name=label_column_name)
            data_frame_copy = pd.concat([data_frame_copy, label_column_series], axis=1)

            best_minimumMSEValue = float('inf')  # value to keep min mse value
            best_maxAccuracy = 0.0
            minMSEFeatureList = []  # value to keep list of feature where min mse occurs

            isFirstTime = True

            status_feature_inc = (status_column_increase / float(len(data_frame_copy.columns) - 1))

            while (len(data_frame_copy.columns) >= 2):
                if (early_end):
                    break

                number_of_features = len(data_frame_copy.columns) - 1
                feature_reduce_rate = int(number_of_features / 50) + 1
                remove_heap = []

                if(number_of_features - 1 > 0):
                    modelClass = modelArchitectureClassification(number_of_features - 1, number_of_unique_values, preset_optimizer)
                    modelReg = modelArchitectureRegression(number_of_features - 1, preset_optimizer)

                remove_feature_index = -1
                local_minimumMSEValue = float('inf')
                local_maxAccuracy = 0.0

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

                        y_train = data_to_categorical(y_train, column_unique_values, number_of_unique_values, categorical_dict)

                        checkpointerClass = EarlyStopping(monitor='acc', min_delta=0, patience=0, verbose=0, mode='auto')
                        model.fit(x_train, y_train, epochs=15, batch_size=preset_batch_size, verbose=0, callbacks=[checkpointerClass])

                        y_test = data_to_categorical(y_test, column_unique_values, number_of_unique_values, categorical_dict)
                        accuracy = model.evaluate(x_test, y_test, batch_size=preset_batch_size, verbose=0)

                        Logging.Logging.write_log_to_file_selectable(best_maxAccuracy)

                        Logging.Logging.write_log_to_file_selectable("Loss: {}, Accuracy: {} ".format(accuracy[0], accuracy[1]))
                        Logging.Logging.write_log_to_file_selectable(feature_set_column_names)
                        Logging.Logging.write_log_to_file_selectable('Column Start 3 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


                        if(np.isnan(accuracy[0])):
                            accuracy[0] = float('inf')

                        if j >= 0:
                            remove_heap.append((accuracy[1], j))

                        if accuracy[1] > local_maxAccuracy or (accuracy[1] == local_maxAccuracy and accuracy[0] <= local_minimumMSEValue):
                            local_minimumMSEValue = accuracy[0]
                            local_maxAccuracy = accuracy[1]
                            remove_feature_index = j
                            Logging.Logging.write_log_to_file_selectable('local')
                            if local_maxAccuracy > best_maxAccuracy or (local_maxAccuracy == best_maxAccuracy and local_minimumMSEValue <= best_minimumMSEValue):
                                Logging.Logging.write_log_to_file_selectable('best')
                                best_minimumMSEValue = local_minimumMSEValue
                                best_maxAccuracy = accuracy[1]
                                minMSEFeatureList = feature_set_column_names
                                save_model = model
                                save_x_train = x_train
                                save_y_train = y_train
                                save_x_test = x_test
                                save_y_test = y_test

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

                        checkpointerReg = EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto')
                        model.fit(x_train, y_train, epochs=15, batch_size=preset_batch_size, verbose=0, callbacks=[checkpointerReg])

                        loss_and_metrics = model.evaluate(x_test, y_test, batch_size=preset_batch_size, verbose=0)

                        Logging.Logging.write_log_to_file_selectable(best_minimumMSEValue)

                        Logging.Logging.write_log_to_file_selectable(loss_and_metrics)
                        Logging.Logging.write_log_to_file_selectable(feature_set_column_names)
                        Logging.Logging.write_log_to_file_selectable('Column Start 3 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                        if(np.isnan(loss_and_metrics)):
                            loss_and_metrics = float('inf')

                        if j >= 0:
                            remove_heap.append((loss_and_metrics, j))

                        if loss_and_metrics <= local_minimumMSEValue:
                            local_minimumMSEValue = loss_and_metrics
                            remove_feature_index = j
                            Logging.Logging.write_log_to_file_selectable('local')
                            if local_minimumMSEValue <= best_minimumMSEValue:
                                Logging.Logging.write_log_to_file_selectable('best')
                                best_minimumMSEValue = local_minimumMSEValue
                                minMSEFeatureList = feature_set_column_names
                                save_model = model
                                save_x_train = x_train
                                save_y_train = y_train
                                save_x_test = x_test
                                save_y_test = y_test

                                if (number_of_features - 1 > 0):
                                    modelReg = modelArchitectureRegression(number_of_features - 1, preset_optimizer)

                if (remove_feature_index >= 0):

                    heapq.heapify(remove_heap)
                    if(classification):
                        drop_list = heapq.nlargest(feature_reduce_rate, remove_heap)
                    else:
                        drop_list = heapq.nsmallest(feature_reduce_rate, remove_heap)

                    Logging.Logging.write_log_to_file_selectable('removed feature: {}'.format(str(drop_list)))
                    data_frame_copy.drop(data_frame_copy.columns[[k[1] for k in drop_list]], axis=1, inplace=True)

                status = status + status_feature_inc
                Logging.Logging.write_log_to_file_status(str(status))
                Logging.Logging.write_log_to_file_status_flush()

                try:
                    GoogleStorage.upload_logs(email, file_name_full)
                except Exception as e:
                    Logging.Logging.write_log_to_file_selectable("Logs Upload Failed : " + str(e))

            # add to dict best mse and best feature list, save the model
            label_suggestion[label_column_name] = minMSEFeatureList

            label_types[label_column_name] = "number"
            if label_column_name in string_columns:
                label_types[label_column_name] = "string"
            elif label_column_name in date_columns:
                label_types[label_column_name] = "date"

            if classification:
                checkpointer = EarlyStopping(monitor='acc', min_delta=0, patience=2, verbose=1, mode='auto')
            else:
                checkpointer = EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=1, mode='auto')

            if(classification):
                save_model_cand = modelArchitectureClassification(np.shape(save_x_train)[1], number_of_unique_values, preset_optimizer)
            else:
                save_model_cand = modelArchitectureRegression(np.shape(save_x_train)[1], preset_optimizer)


            save_model_cand.fit(save_x_train, save_y_train, epochs=50, batch_size=preset_batch_size, verbose=1, callbacks=[checkpointer])
            save_val_losses = save_model_cand.evaluate(save_x_test, save_y_test, batch_size=preset_batch_size, verbose=1)

            if classification:
                if best_maxAccuracy < save_val_losses[1] or (best_maxAccuracy == save_val_losses[1] and best_minimumMSEValue > save_val_losses[0]):
                    label_mse_score_best_epochs[label_column_name] = (save_val_losses[0], classification, save_val_losses[1])
                    save_model = save_model_cand
                else:
                    label_mse_score_best_epochs[label_column_name] = (best_minimumMSEValue, classification, best_maxAccuracy)
                label_mse_score[label_column_name] = (best_minimumMSEValue, classification, best_maxAccuracy)
                label_category_dict[label_column_name] = categorical_dict
            else:
                if best_minimumMSEValue > save_val_losses:
                    label_mse_score_best_epochs[label_column_name] = (save_val_losses, classification)
                    save_model = save_model_cand
                else:
                    label_mse_score_best_epochs[label_column_name] = (best_minimumMSEValue, classification)
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
            Logging.Logging.write_log_to_file_selectable_flush()

    create_report_file(reports_path, file_name, label_mse_score, label_suggestion, label_types, label_mse_score_best_epochs)
    create_details_file(outputs_path, file_name, label_suggestion, label_types, label_category_dict)

    Logging.Logging.write_log_to_file_selectable_flush()
