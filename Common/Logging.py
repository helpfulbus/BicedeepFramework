# <copyright company="Bicedeep, Inc.">
# Copyright (c) 2016-2018 All Rights Reserved
# </copyright>

import datetime
import os
import config
from google.cloud import logging

class Logging:
    f = open('reportserver.log', 'a')
    f2 = open('selectablefeature.log', 'a')
    f3 = open('optimization.log', 'a')
    f4 = open('queueserver.log', 'a')

    @staticmethod
    def init():
        path = './logs'
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.system("sudo mkdir " + dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

    @staticmethod
    def write_log_to_file(input_string):
        input_string = str(input_string)
        input_string = str(datetime.datetime.now()) + "  :  " + input_string + "\n"
        print(input_string)
        Logging.f.write(input_string)
        #Logging.f.flush()

    @staticmethod
    def write_log_to_file_flush():
        Logging.f.flush()

    @staticmethod
    def write_log_to_file_selectable(input_string):
        input_string = str(input_string)
        input_string = str(datetime.datetime.now()) + "  :  " + input_string + "\n"
        print(input_string)
        Logging.f2.write(input_string)
        # Logging.f.flush()

    @staticmethod
    def write_log_to_file_selectable_flush():
        Logging.f2.flush()

    @staticmethod
    def write_log_to_file_optimization(input_string):
        input_string = str(input_string)
        input_string = str(datetime.datetime.now()) + "  :  " + input_string + "\n"
        print(input_string)
        Logging.f3.write(input_string)
        # Logging.f.flush()

    @staticmethod
    def write_log_to_file_optimization_flush():
        Logging.f3.flush()

    @staticmethod
    def write_log_to_file_queueserver(input_string):
        input_string = str(input_string)
        input_string = str(datetime.datetime.now()) + "  :  " + input_string + "\n"
        print(input_string)
        Logging.f4.write(input_string)
        # Logging.f.flush()

    @staticmethod
    def write_log_to_file_queueserver_flush():
        Logging.f4.flush()

    @staticmethod
    def remoteLog(log_name, input_string):
        logging_client = logging.Client(project=config.PROJECT_NAME)
        logger = logging_client.logger(log_name)
        logger.log_text(input_string)
