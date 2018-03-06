# <copyright company="Bicedeep, Inc.">
# Copyright (c) 2016-2018 All Rights Reserved
# </copyright>

import math
import json
import numpy as np
import pandas as pd
import datetime as dt
import os
import locale
from Common import Logging
from ReportServerAI import report
from keras.models import load_model
from dateutil.parser import parse
from decimal import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def result_string_from_regression(predict, result_type):
    result = float(predict)

    if (result_type == "string"):
        result = report.convertFromNumber(Decimal(result))
    elif (result_type == "date"):
        result = str(dt.datetime.fromordinal(int(result)))
    else:
        result = str(result)

    return result


def result_string_from_category(predictArr, category_dict, result_type):
    result_string = ""

    for i in range(0, len(predictArr)):

        best_index = np.argmax(predictArr)
        for key, value in category_dict.items():
            if (value == str(best_index)):
                result = key
                result_percent = "{:.1%}".format(predictArr[best_index])
                predictArr[best_index] = 0
                break

        if (result_type == "string"):
            result = report.convertFromNumber(Decimal(result))
        elif (result_type == "date"):
            result = str(dt.datetime.fromordinal(int(result)))
        else:
            result = str(result)

        result_string = result_string + result + "(" + result_percent + ")  "

    return str(result_string)


def calculate_query_from_json(query_file_path, model_file_path, model_details_path):
    model = load_model(model_file_path)

    with open(query_file_path) as data_file:
        data = json.load(data_file)

    x_query = np.array([])
    for x in data['query_using']:
        x_value = x['value']

        try:
            x_value = float(x_value)
        except:
            if report.is_number(x_value):
                x_value = report.to_comma_int(x_value)
            elif report.is_date(x_value):
                x_value = pd.to_datetime(x_value).toordinal()
            else:
                x_value = report.convertToNumber(x_value)

        x_query = np.insert(x_query, len(x_query), x_value)

    x_query = np.reshape(x_query, (-1, len(x_query)))

    predict = model.predict(x_query, batch_size=1)

    if (model.outputs[0].shape[1] > 1):  # classification
        with open(model_details_path) as data_file:
            details_data = json.load(data_file)
            category_dict = details_data['suggestions'][0]['categories']

        data['results'] = result_string_from_category(predict[0], category_dict, data['result_type'])

    else:
        data['results'] = result_string_from_regression(predict[0][0], data['result_type'])

    Logging.Logging.write_log_to_file_queueserver(str(data))
    Logging.Logging.write_log_to_file_queueserver_flush()
    with open(query_file_path, 'w') as output_file:
        json.dump(data, output_file)


def calculate_query_from_csv(query_file_path, model_file_path, model_details_path, result_column_name, result_column_type):

    model = load_model(model_file_path)
    data_frame = report.readData(query_file_path)
    data_frame_copy = data_frame.copy(deep=True)

    string_columns, date_columns, big_number_columns = report.preprocess_data(data_frame)

    predict_frame = model.predict(data_frame.values, batch_size=64)

    result_column_series = pd.DataFrame({result_column_name:predict_frame.tolist()})

    if (model.outputs[0].shape[1] > 1):  # classification
        with open(model_details_path) as data_file:
            details_data = json.load(data_file)
            category_dict = details_data['suggestions'][0]['categories']

        result_column_series[result_column_name] = result_column_series[result_column_name].apply(lambda x: result_string_from_category(x, category_dict, result_column_type))

    else:
        result_column_series[result_column_name] = result_column_series[result_column_name].apply(lambda x: result_string_from_regression(x[0], result_column_type))

    data_frame = pd.concat([data_frame_copy, result_column_series], axis=1)
    data_frame.to_csv(query_file_path, index=False)


def calculate_query(query_file_path, model_file_path, model_details_path, selected_headers):
    query_file_path_split = query_file_path.split('.')

    if query_file_path_split[-1] == "csv":
        calculate_query_from_csv(query_file_path, model_file_path, model_details_path, selected_headers[0], selected_headers[1])
    else:
        calculate_query_from_json(query_file_path, model_file_path, model_details_path)
