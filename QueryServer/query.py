import math
import json
import numpy as np
import pandas as pd
import datetime as dt
from keras.models import load_model
from dateutil.parser import parse


def is_date(string):
    try:
        parse(string)
        return True
    except:
        return False


def convertToNumber(s):
    return int.from_bytes(s.encode(), 'little')


def convertFromNumber(n):
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()


def calculate_query(quey_file_path, model_file_path):

    model = load_model(model_file_path)

    with open(quey_file_path) as data_file:
        data = json.load(data_file)

    x_query = np.array([])
    for x in data['query_using']:
        x_value = x['value']

        try:
            x_value = float(x_value)
        except:
            if is_date(x_value):
                x_value = pd.to_datetime(x_value).toordinal()
            else:
                x_value = convertToNumber(x_value)

        x_query = np.insert(x_query, len(x_query), x_value)

    x_query = np.reshape(x_query, (-1, len(x_query)))

    predict = model.predict(x_query, batch_size=1)

    result = predict[0][0]

    if (data['result_type'] == "string"):
        result = convertFromNumber(result)
    elif (data['result_type'] == "date"):
        result = str(dt.datetime.date.fromordinal(result))
    else:
        result = str(result)

    data['results'] = result

    print(str(data))
    with open(quey_file_path, 'w') as output_file:
        json.dump(data, output_file)

