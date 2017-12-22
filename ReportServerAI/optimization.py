# <copyright company="Bicedeep, Inc.">
# Copyright (c) 2016-2018 All Rights Reserved
# </copyright>

import json
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn import model_selection
import time
import gc
import os
import random
from math import log
from math import ceil
from numpy import argsort
from keras.layers import Dropout
from ReportServerAI import report
from Common import Logging
from Common import GoogleStorage

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=128))

# Build list that contains all possible architecture structures
def get_architecture_list():
    architecture_list = []
    arch1 = [2, 32, 32]
    architecture_list.append(arch1)
    arch2 = [2, 64, 64]
    architecture_list.append(arch2)
    arch3 = [2, 128, 128]
    architecture_list.append(arch3)
    arch4 = [2, 64, 32]
    architecture_list.append(arch4)
    arch5 = [2, 32, 64]
    architecture_list.append(arch5)
    arch6 = [2, 128, 64]
    architecture_list.append(arch6)
    arch7 = [2, 128, 32]
    architecture_list.append(arch7)
    arch8 = [3, 128, 64, 32]
    architecture_list.append(arch8)
    arch9 = [3, 128, 64, 64]
    architecture_list.append(arch9)
    arch10 = [3, 128, 32, 64]
    architecture_list.append(arch10)
    arch11 = [4, 128, 32, 64, 32]
    architecture_list.append(arch11)
    arch12 = [4, 256, 128, 64, 32]
    architecture_list.append(arch12)
    arch13 = [4, 128, 64, 64, 32]
    architecture_list.append(arch13)
    arch14 = [4, 256, 128, 64, 64]
    architecture_list.append(arch14)
    return architecture_list


# Return random hyperparameters
# parameters[0] = architecture structure
# parameters[1] = optimizers
# parameters[2] = activations
# parameters[3] = regularization
# parameters[4] = batch_size
def get_random_hyperparameter_configuration():
    parameters = []
    arch_list = get_architecture_list()
    parameters.append(random.choice(arch_list))
    optimizers = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
    parameters.append(random.choice(optimizers))
    activations = ['relu', 'sigmoid', 'tanh', 'softmax']
    parameters.append(random.choice(activations))
    regularization = [0, 1]  # 0: dropout, 1: batch_normalization
    parameters.append(random.choice(regularization))
    batch_sizes = [16, 32, 64, 128, 512, 1024]
    parameters.append(random.choice(batch_sizes))

    return parameters


def run_then_return_val_loss(hyperparameters, input_dim, is_classification, output_dimension, x_train, y_train, x_test, y_test, epochs, callbacks, verb):
    with tf.device('/cpu:0'):
        number_of_layers = hyperparameters[0][0]
        model = Sequential()
        for i in range(0, number_of_layers):
            if (i == 0):
                model.add(Dense(hyperparameters[0][i + 1], input_dim=input_dim, kernel_initializer='normal',
                                activation=hyperparameters[2]))
            else:
                model.add(Dense(hyperparameters[0][i + 1], activation=hyperparameters[2]))
            if (hyperparameters[3]):
                model.add(BatchNormalization())
            else:
                model.add(Dropout(0.1))
        if (is_classification):
            if(output_dimension < 2):
                output_dimension = 2
            model.add(Dense(output_dimension, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=hyperparameters[1], metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=epochs, batch_size=hyperparameters[4], verbose=verb, callbacks=callbacks)
            loss_and_metrics = model.evaluate(x_test, y_test, batch_size=hyperparameters[4], verbose=verb)
            return loss_and_metrics, model
        else:
            model.add(Dense(1, kernel_initializer='normal'))
            model.compile(loss='mean_absolute_error', optimizer=hyperparameters[1])

            model.fit(x_train, y_train, epochs=epochs, batch_size=hyperparameters[4], verbose=verb, callbacks=callbacks)
            loss_and_metrics = model.evaluate(x_test, y_test, batch_size=hyperparameters[4], verbose=verb)
            return loss_and_metrics, model


def contains(array, item):
    for i in range(0, len(array)):
        if (array[i] == item):
            return True
    return False


def get_remove_label_list(label_list, suggestions, label_name):
    remove_list = []
    for i in range(0, len(label_list)):
        if (contains(suggestions, label_list[i]) == False and label_list[i] != label_name):
            remove_list.append(label_list[i])
    return remove_list

def do_optimization(file_path, file_name, reports_path, outputs_path, email, file_name_full):

    data_frame = report.readData(file_path)

    Logging.Logging.write_log_to_file_selectable("Starting Pre Process")
    string_columns, date_columns, big_number_columns = report.preprocess_data(data_frame)
    Logging.Logging.write_log_to_file_selectable("Pre Process Ended")

    with open(reports_path + "/" + file_name + ".json") as data_file:
        data = json.load(data_file)

    id_name_dict = report.LastUpdatedOrderedDict()
    for i in range(0, len(data['part_predictability'])):
        id_name_dict[i] = data['part_predictability'][i]['part_name']

    label_mse_score = {}
    label_types = {}
    label_mse_score_best_epochs = {}
    for i in range(0, len(data['part_predictability'])):
        score = data['part_predictability'][i]['predictability']
        classification = data['part_predictability'][i]['classification']

        if(classification):
            accuracy = data['part_predictability'][i]['accuracy']
            label_mse_score[data['part_predictability'][i]['part_name']] = (score, classification, accuracy)
        else:
            label_mse_score[data['part_predictability'][i]['part_name']] = (score, classification)
        label_types[data['part_predictability'][i]['part_name']] = data['part_predictability'][i]['type']

    for i in range(0, len(data['part_predictability_best_epochs'])):
        score = data['part_predictability_best_epochs'][i]['predictability']
        classification = data['part_predictability_best_epochs'][i]['classification']

        if (classification):
            accuracy = data['part_predictability_best_epochs'][i]['accuracy']
            label_mse_score_best_epochs[data['part_predictability_best_epochs'][i]['part_name']] = (score, classification, accuracy)
        else:
            label_mse_score_best_epochs[data['part_predictability_best_epochs'][i]['part_name']] = (score, classification)

    desired_columns_as_label = []
    label_suggestion = {}
    for i in range(0, len(data['suggestions'])):
        suggestions = data['suggestions'][i]['using']
        label_suggestion[data['suggestions'][i]['part']] = suggestions
        desired_columns_as_label.append(data['suggestions'][i]['part'])


    Logging.Logging.write_log_to_file_optimization("Optimization Has Started")
    with tf.device('/cpu:0'):
        gc.enable()
        num_of_cols = len(desired_columns_as_label)
        status = 80.0 / 100.0
        desired_columns_as_label = list(map((lambda x: x.strip("\n ,")), desired_columns_as_label))

        status_column_increase = (10.0 / num_of_cols) / 100

        for index in range(0, num_of_cols):
            data_frame_copy = data_frame.copy(deep=True)

            dfc_columns = list(map((lambda x: x.strip("\n ,")), data_frame_copy.columns))

            label_column_index = dfc_columns.index(desired_columns_as_label[index])
            label_column = data_frame_copy.iloc[:, label_column_index]
            # decide whether regression or classification will be applied to the column
            classification = False
            column_unique_values = label_column.unique()
            number_of_unique_values = len(label_column.unique())
            number_of_samples = len(label_column.values)

            label_column_name = data_frame_copy.columns[label_column_index]

            classification = report.is_number_classifiction(number_of_unique_values, number_of_samples)

            if (label_column_name in string_columns or label_column_name in date_columns):
                classification = report.is_string_classifiction(number_of_unique_values, number_of_samples)

            if (classification):
                categorical_dict = report.get_categorical_dict(column_unique_values)

            # drop label column from data frame
            data_frame_copy.drop(data_frame_copy.columns[[label_column_index]], axis=1, inplace=True)
            # add label column as last column to the data_frame
            label_column_series = pd.Series(label_column.values, name=label_column_name)
            data_frame_copy = pd.concat([data_frame_copy, label_column_series], axis=1)

            # get name of columns to remove
            label_list = data_frame_copy.columns
            rem_list = get_remove_label_list(label_list, label_suggestion[label_column_name], label_column_name)
            # remove columns that are not suggested
            for j in range(0, len(rem_list)):
                del data_frame_copy[rem_list[j]]

            Logging.Logging.write_log_to_file_optimization(data_frame_copy.columns)
            dataset = data_frame_copy.values
            number_of_features = len(data_frame_copy.columns) - 1
            X = dataset[:, 0: number_of_features]
            Y = dataset[:, number_of_features]
            x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

            if (classification):
                y_train = report.data_to_categorical(y_train, column_unique_values, number_of_unique_values, categorical_dict)
                y_test = report.data_to_categorical(y_test, column_unique_values, number_of_unique_values, categorical_dict)

            Logging.Logging.write_log_to_file_optimization("Optimization Start for : {}".format(label_column_name))

            if classification:
                checkpointer = EarlyStopping(monitor='acc', min_delta=0, patience=0, verbose=0, mode='auto')
            else:
                checkpointer = EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto')

            model_save_file_number = report.getModelNumber(index + 1)
            model_save_file_name = outputs_path + "/" + file_name + "." + model_save_file_number + ".h5"
            ##Hyperband algorithm##
            max_iter = 32
            eta = 2
            logeta = lambda x: log(x) / log(eta)
            s_max = int(logeta(max_iter))
            B = (s_max + 1) * max_iter

            for s in reversed(range(s_max + 1)):
                n = int(ceil(B / max_iter / (s + 1) * eta ** s))
                r = max_iter * eta ** (-s)
                T = [get_random_hyperparameter_configuration() for i in range(n)]
                for i in range(s):
                    start = time.time()
                    n_i = n * eta ** (-i)
                    r_i = r * eta ** (i)
                    val_losses = []
                    models = []
                    hypers = []

                    for t in T:
                        loss, mod = run_then_return_val_loss(t, number_of_features, classification,
                                                             number_of_unique_values, x_train, y_train, x_test, y_test, 15, [checkpointer], 0)
                        Logging.Logging.write_log_to_file_optimization(t)
                        Logging.Logging.write_log_to_file_optimization(loss)
                        if (classification):
                            if (np.isnan(loss[0])):
                                loss[0] = float('inf')
                            val_losses.append((loss[0], loss[1]))
                        else:
                            if (np.isnan(loss)):
                                loss = float('inf')
                            val_losses.append((loss, 0))

                        models.append(mod)
                        hypers.append(t)

                    T = [T[m] for m in (argsort(val_losses, axis=0)[:,0])[0:int(n_i / eta)]]

                    if len(T) == 0:
                        continue

                    hypers = [hypers[j] for j in (argsort(val_losses, axis=0)[:,0])]
                    models = [models[j] for j in (argsort(val_losses, axis=0)[:,0])]
                    val_losses = sorted(val_losses, key=lambda x: x[0])
                    end = time.time()
                    for k in range(0, int(n_i / eta)):
                        Logging.Logging.write_log_to_file_optimization("Parameters: {}, Loss: {}".format(T[k], val_losses[k]))

                    if(classification):
                        best_accuracy = label_mse_score[label_column_name][2]
                        found_better = False
                        for i in range(0, len(val_losses)):
                            found_better_local = val_losses[i][1] > best_accuracy
                            if val_losses[i][1] == best_accuracy:
                                found_better_local = val_losses[i][0] < label_mse_score[label_column_name][0]
                            if(found_better_local):
                                best_accuracy = val_losses[i][1]
                                used_model_id = i
                                found_better = True
                    else:
                        found_better = val_losses[0][0] < label_mse_score[label_column_name][0]
                        used_model_id = 0
                    if (len(val_losses) > 0 and len(models) > 0 and found_better):
                        Logging.Logging.write_log_to_file_optimization('Found better')

                        #is it better than old trained best
                        if classification:
                            if label_mse_score_best_epochs[label_column_name][2] < val_losses[used_model_id][1] or (label_mse_score_best_epochs[label_column_name][2] == val_losses[used_model_id][1] and label_mse_score_best_epochs[label_column_name][0] > val_losses[used_model_id][0]):
                                label_mse_score_best_epochs[label_column_name] = (val_losses[used_model_id][0], classification, val_losses[used_model_id][1])
                                label_mse_score[label_column_name] = (val_losses[used_model_id][0], classification, val_losses[used_model_id][1])
                                models[used_model_id].save(model_save_file_name)
                        else:
                            if label_mse_score_best_epochs[label_column_name][0] > val_losses[used_model_id][0]:
                                label_mse_score_best_epochs[label_column_name] = (val_losses[used_model_id][0], classification)
                                label_mse_score[label_column_name] = (val_losses[used_model_id][0], classification)
                                models[used_model_id].save(model_save_file_name)


                        if classification:
                            checkpointer = EarlyStopping(monitor='acc', min_delta=0, patience=2, verbose=1,mode='auto')
                        else:
                            checkpointer = EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=1,mode='auto')


                        save_val_losses, save_model_cand = run_then_return_val_loss(hypers[used_model_id], number_of_features, classification,
                                                             number_of_unique_values, x_train, y_train, x_test, y_test, 50, [checkpointer], 1)

                        if classification:
                            if label_mse_score_best_epochs[label_column_name][2] < save_val_losses[1] or (label_mse_score_best_epochs[label_column_name][2] == save_val_losses[1] and label_mse_score_best_epochs[label_column_name][0] > save_val_losses[0]):
                                label_mse_score_best_epochs[label_column_name] = (save_val_losses[0], classification, save_val_losses[1])
                                save_model_cand.save(model_save_file_name)
                        else:
                            if label_mse_score_best_epochs[label_column_name][0] > save_val_losses:
                                label_mse_score_best_epochs[label_column_name] = (save_val_losses, classification)
                                save_model_cand.save(model_save_file_name)

                    Logging.Logging.write_log_to_file_optimization("Time elapsed: {} seconds".format(end - start))
                    Logging.Logging.write_log_to_file_optimization("*****")
                Logging.Logging.write_log_to_file_optimization("End of iteration: {}".format(i))
                gc.collect()
                break

            status = status + status_column_increase
            Logging.Logging.write_log_to_file_status(str(status))
            Logging.Logging.write_log_to_file_status_flush()
            Logging.Logging.write_log_to_file_optimization_flush()

            try:
                GoogleStorage.upload_logs(email, file_name_full)
            except Exception as e:
                Logging.Logging.write_log_to_file_selectable("Logs Upload Failed : " + str(e))

    label_mse_score_ordered = report.LastUpdatedOrderedDict()
    label_suggestion_ordered = report.LastUpdatedOrderedDict()

    for key, value in id_name_dict.items():
        label_mse_score_ordered[value] = label_mse_score_best_epochs[value]
        label_suggestion_ordered[value] = label_suggestion[value]

    Logging.Logging.write_log_to_file_optimization("Result : " + str(label_mse_score_ordered) )

    report.create_report_file(reports_path, file_name, label_mse_score_ordered, label_suggestion_ordered, label_types, report.LastUpdatedOrderedDict())
    Logging.Logging.write_log_to_file_optimization_flush()