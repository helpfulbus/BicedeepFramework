# <copyright company="Bicedeep, Inc.">
# Copyright (c) 2016-2018 All Rights Reserved
# </copyright>

from multiprocessing import Process
import os
import posix
import config
import time
from Common import Timing
from Common import GoogleStorage
from Common import Aws
from Common import Logging


def query_calculate(query_file_path, model_file_path, model_details_path, selected_headers):
    from QueryServer import query
    os.nice(-20)
    os.setpriority(posix.PRIO_PROCESS, os.getpid(), -20)
    query.calculate_query(query_file_path, model_file_path, model_details_path, selected_headers)


def queue_server_run(message):
    Logging.Logging.write_log_to_file_queueserver("Read message from aws queue")

    try:
        [query_file_name, model_file_name, selected_headers] = Aws.deserialize_queue_message(message)
    except Exception as e:
        Logging.Logging.write_log_to_file_queueserver(str(e))
        return False

    Logging.Logging.write_log_to_file_queueserver("aws queue deserialize")

    file_path = ""
    try:
        [query_file_path, model_file_path, model_details_path] = GoogleStorage.download_query_file(query_file_name, model_file_name)
    except Exception as e:
        Logging.Logging.write_log_to_file_queueserver(str(e))
        return False

    Logging.Logging.write_log_to_file_queueserver("Query file downloaded : " + query_file_name)
    Logging.Logging.write_log_to_file_queueserver("Model file downloaded : " + model_file_name)
    Logging.Logging.write_log_to_file_queueserver("Model details file downloaded : " + model_file_name + ".json")

    # Calculations
    try:
        p = Process(target=query_calculate, args=(query_file_path, model_file_path, model_details_path, selected_headers,))
        p.start()
        p.join()
    except Exception as e:
        Logging.Logging.write_log_to_file_queueserver(str(e))
        return False

    Logging.Logging.write_log_to_file_queueserver("Query calculation completed")

    # Google upload results
    try:
        GoogleStorage.upload_query_results(query_file_name)
    except Exception as e:
        Logging.Logging.write_log_to_file_queueserver(str(e))

    # Delete local files
    try:
        GoogleStorage.delete_local_dir()
    except Exception as e:
        Logging.Logging.write_log_to_file_queueserver(str(e))

    try:
        Aws.send_completed_message(message.body, 'QueueFileCompleted')
        Logging.Logging.write_log_to_file("Queue File completed message sent to aws")
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))

    try:
        message.delete()
        Logging.Logging.write_log_to_file_queueserver("Query Message Deleted")
    except Exception as e:
        Logging.Logging.write_log_to_file_queueserver(str(e))
        try:
            message_re_read = Aws.read_queue(config.QUEUE_QUEUE_NAME)
            if(message.body == message_re_read.body):
                message_re_read.delete()
                Logging.Logging.write_log_to_file_queueserver("Query Message Deleted")
        except Exception as e:
            Logging.Logging.write_log_to_file_queueserver(str(e))
            return True
        return True
    return True


def main():
    Logging.Logging.init()
    Logging.Logging.write_log_to_file_queueserver("Queue Server Started")

    try:
        Logging.Logging.remoteLog("Query_Server_Alive", "Query_Server_Alive ID : " + config.QUEUE_INSTANCE_ID)
    except Exception as e:
        Logging.Logging.write_log_to_file_queueserver(str(e))

    start_time = time.time()
    while True:
        end_time = time.time()
        if end_time - start_time > 3600:
            try:
                Logging.Logging.remoteLog("Query_Server_Alive", "Query_Server_Alive ID : " + config.QUEUE_INSTANCE_ID)
            except Exception as e:
                Logging.Logging.write_log_to_file_queueserver(str(e))
            start_time = time.time()


        Logging.Logging.write_log_to_file_queueserver_flush()
        try:
            message = Aws.read_queue(config.QUEUE_QUEUE_NAME)
        except Exception as e:
            Logging.Logging.write_log_to_file_queueserver(str(e))
            continue

        if message is not None:
            result = queue_server_run(message)
            if(result == False):
                Aws.add_message_to_exception_queue(message, config.QUEUE_QUEUE_NAME, config.QUERY_EXCEPTION_QUEUE_NAME)
        else:
            if Timing.Timing.DoShutDown(Logging.Logging.write_log_to_file_queueserver, config.QUEUE_INSTANCE_ID):
                Logging.Logging.write_log_to_file_queueserver("Shutting down the instance")
                try:
                    Aws.stop_instance(config.QUEUE_INSTANCE_ID)
                    Logging.Logging.write_log_to_file_queueserver_flush()
                except Exception as e:
                    Logging.Logging.write_log_to_file_queueserver(str(e))





if __name__ == "__main__":
    main()