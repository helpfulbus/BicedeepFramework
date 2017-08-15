from multiprocessing import Process
import os
import posix
import config
from Common import Timing
from Common import GoogleStorage
from Common import Aws
from Common import Logging


def query_calculate(quey_file_path, model_file_path, model_details_path):
    from QueryServer import query
    os.nice(-20)
    os.setpriority(posix.PRIO_PROCESS, os.getpid(), -20)
    query.calculate_query(quey_file_path, model_file_path, model_details_path)


def queue_server_run(message):
    Logging.Logging.write_log_to_file_queueserver("Read message from aws queue")

    try:
        [query_file_name, model_file_name] = Aws.deserialize_queue_message(message)
    except Exception as e:
        Logging.Logging.write_log_to_file_queueserver(str(e))
        return False

    file_path = ""
    try:
        [quey_file_path, model_file_path, model_details_path] = GoogleStorage.download_query_file(query_file_name, model_file_name)
    except Exception as e:
        Logging.Logging.write_log_to_file_queueserver(str(e))
        return False

    Logging.Logging.write_log_to_file_queueserver("Query file downloaded : " + query_file_name)
    Logging.Logging.write_log_to_file_queueserver("Model file downloaded : " + model_file_name)
    Logging.Logging.write_log_to_file_queueserver("Model details file downloaded : " + model_file_name + ".json")

    # Calculations
    try:
        p = Process(target=query_calculate, args=(quey_file_path, model_file_path, model_details_path,))
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
    while True:
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
            if Timing.Timing.DoShutDown(Logging.Logging.write_log_to_file_queueserver):
                Logging.Logging.write_log_to_file_queueserver("Shutting down the instance")
                try:
                    Aws.stop_instance(config.QUEUE_INSTANCE_ID)
                    Logging.Logging.write_log_to_file_queueserver_flush()
                except Exception as e:
                    Logging.Logging.write_log_to_file_queueserver(str(e))





if __name__ == "__main__":
    main()