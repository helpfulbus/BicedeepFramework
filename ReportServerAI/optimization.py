import json
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
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
    optimizers = ['sgd', 'Adam', 'Nadam', 'adadelta']
    parameters.append(random.choice(optimizers))
    activations = ['relu', 'sigmoid']
    parameters.append(random.choice(activations))
    regularization = [0, 1]  # 0: dropout, 1: batch_normalization
    parameters.append(random.choice(regularization))
    batch_sizes = [2, 32, 128, 512, 1024]
    parameters.append(random.choice(batch_sizes))

    return parameters


def run_then_return_val_loss(num_iters, hyperparameters, input_dim, is_classification, output_dimension, x_train, y_train, x_test, y_test):
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

            model.fit(x_train, y_train, epochs=10, batch_size=hyperparameters[4], verbose=0)
            loss_and_metrics = model.evaluate(x_test, y_test, batch_size=hyperparameters[4], verbose=0)
            return loss_and_metrics, model
        else:
            model.add(Dense(1, kernel_initializer='normal'))
            model.compile(loss='mean_absolute_error', optimizer=hyperparameters[1])

            model.fit(x_train, y_train, epochs=15, batch_size=hyperparameters[4], verbose=0)
            loss_and_metrics = model.evaluate(x_test, y_test, batch_size=hyperparameters[4], verbose=0)
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


def do_optimization(file_path, file_name, reports_path, outputs_path):

    data_frame = report.readData(file_path)

    Logging.Logging.write_log_to_file_selectable("Starting Pre Process")
    string_columns, date_columns, big_number_columns = report.preprocess_data(data_frame)
    Logging.Logging.write_log_to_file_selectable("Pre Process Ended")

    with open(reports_path + "/" + file_name + ".json") as data_file:
        data = json.load(data_file)

    label_mse_score = {}
    label_types = {}
    for i in range(0, len(data['part_predictability'])):
        score = data['part_predictability'][i]['predictability']
        classification = data['part_predictability'][i]['classification']

        if(classification):
            accuracy = data['part_predictability'][i]['accuracy']
            label_mse_score[data['part_predictability'][i]['part_name']] = (score, classification, accuracy)
        else:
            label_mse_score[data['part_predictability'][i]['part_name']] = (score, classification)
        label_types[data['part_predictability'][i]['part_name']] = data['part_predictability'][i]['type']

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
        for index in range(0, num_of_cols):
            data_frame_copy = data_frame.copy(deep=True)
            label_column_index = data_frame_copy.columns.get_loc(desired_columns_as_label[index])
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
                y_train, category_dict = report.data_to_categorical(y_train, column_unique_values, number_of_unique_values)
                y_test, category_dict = report.data_to_categorical(y_test, column_unique_values, number_of_unique_values)

            Logging.Logging.write_log_to_file_optimization("Optimization Start for : {}".format(label_column_name))

            model_save_file_number = report.getModelNumber(index + 1)
            model_save_file_name = outputs_path + "/" + file_name + "." + model_save_file_number + ".h5"
            ##Hyperband algorithm##
            max_iter = 27
            eta = 3
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

                    for t in T:
                        loss, mod = run_then_return_val_loss(int(r_i), t, number_of_features, classification,
                                                             number_of_unique_values, x_train, y_train, x_test, y_test)
                        Logging.Logging.write_log_to_file_optimization(t)
                        Logging.Logging.write_log_to_file_optimization(loss)
                        if (classification):
                            val_losses.append((loss[0], loss[1]))
                        else:
                            val_losses.append((loss, 0))

                        models.append(mod)

                    T = [T[m] for m in (argsort(val_losses, axis=0)[:,0])[0:int(n_i / eta)]]

                    if len(T) == 0:
                        continue
                    models = [models[j] for j in (argsort(val_losses, axis=0)[:,0])[0:int(n_i / eta)]]
                    val_losses = sorted(val_losses, key=lambda x: x[0])
                    end = time.time()
                    for k in range(0, int(n_i / eta)):
                        Logging.Logging.write_log_to_file_optimization("Parameters: {}, Loss: {}".format(T[k], val_losses[k]))
                    if (len(val_losses) > 0 and len(models) > 0 and val_losses[0][0] < label_mse_score[label_column_name][0]):
                        Logging.Logging.write_log_to_file_optimization('Found better')
                        save_model = models[0]
                        save_model.save(model_save_file_name)
                        if classification:
                            label_mse_score[label_column_name] = (val_losses[0][0], classification, val_losses[0][1])
                        else:
                            label_mse_score[label_column_name] = (val_losses[0][0], classification)

                    Logging.Logging.write_log_to_file_optimization("Time elapsed: {} seconds".format(end - start))
                    Logging.Logging.write_log_to_file_optimization("*****")
                Logging.Logging.write_log_to_file_optimization("End of iteration: {}".format(i))
                gc.collect()
                break

            Logging.Logging.write_log_to_file_optimization_flush()

    report.create_report_file(reports_path, file_name, label_mse_score, label_suggestion, label_types)
    Logging.Logging.write_log_to_file_optimization_flush()