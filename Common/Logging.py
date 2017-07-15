import datetime


class Logging:
    f = open('./reportserver.log', 'a')
    f2 = open('./selectablefeature.log', 'a')
    f3 = open('./optimization.log', 'a')

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
