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


def calculate_query(quey_file_path, model_file_path, model_details_path):

    model = load_model(model_file_path)

    with open(quey_file_path) as data_file:
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


    if(model.outputs[0].shape[1] > 1): # classification
        with open(model_details_path) as data_file:
            details_data = json.load(data_file)
            category_dict = details_data['suggestions'][0]['categories']

        best_index = predict[0].argmax()
        for key, value in category_dict.items():
            if(value == str(best_index)):
                result = key

    else:
        result = float(predict[0][0])

    if (data['result_type'] == "string"):
        result = report.convertFromNumber(Decimal(result))
    elif (data['result_type'] == "date"):
        result = str(dt.datetime.fromordinal(int(result)))
    else:
        result = str(result)

    data['results'] = result

    Logging.Logging.write_log_to_file_queueserver(str(data))
    with open(quey_file_path, 'w') as output_file:
        json.dump(data, output_file)

