import datetime


class Logging:
    f = open('./reportserver.log', 'a')

    @staticmethod
    def write_log_to_file(input_string):
        input_string = str(input_string)
        Logging.f.write(str(datetime.datetime.now()) + "  :  " + input_string + "\n")
        Logging.f.flush()
