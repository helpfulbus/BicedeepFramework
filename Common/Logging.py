import datetime
import os

class Logging:
    f = open('logs/reportserver.log', 'a')
    f2 = open('logs/selectablefeature.log', 'a')
    f3 = open('logs/optimization.log', 'a')
    f4 = open('logs/queueserver.log', 'a')

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
        #print(input_string)
        Logging.f.write(str(datetime.datetime.now()) + "  :  " + input_string + "\n")
        #Logging.f.flush()

    @staticmethod
    def write_log_to_file_flush():
        Logging.f.flush()

    @staticmethod
    def write_log_to_file_selectable(input_string):
        input_string = str(input_string)
        # print(input_string)
        Logging.f2.write(str(datetime.datetime.now()) + "  :  " + input_string + "\n")
        # Logging.f.flush()

    @staticmethod
    def write_log_to_file_selectable_flush():
        Logging.f2.flush()

    @staticmethod
    def write_log_to_file_optimization(input_string):
        input_string = str(input_string)
        # print(input_string)
        Logging.f3.write(str(datetime.datetime.now()) + "  :  " + input_string + "\n")
        # Logging.f.flush()

    @staticmethod
    def write_log_to_file_optimization_flush():
        Logging.f3.flush()

    @staticmethod
    def write_log_to_file_queueserver(input_string):
        input_string = str(input_string)
        # print(input_string)
        Logging.f4.write(str(datetime.datetime.now()) + "  :  " + input_string + "\n")
        # Logging.f.flush()

    @staticmethod
    def write_log_to_file_queueserver_flush():
        Logging.f4.flush()