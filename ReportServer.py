# <copyright company="Bicedeep, Inc.">
# Copyright (c) 2016-2018 All Rights Reserved
# </copyright>

from multiprocessing import Process
import os
import posix
import time
from Common import Timing
from Common import GoogleStorage
from Common import Aws
from Common import Logging


import config


def selectable_feature(file_path, file_name, selected_headers, reports_path, outputs_path):
    from ReportServerAI import report
    os.nice(-20)
    os.setpriority(posix.PRIO_PROCESS, os.getpid(), -20)
    report.create_report(file_path, file_name, selected_headers, reports_path, outputs_path)


def optimization(file_path, file_name, reports_path, outputs_path):
    from ReportServerAI import optimization
    os.nice(-20)
    os.setpriority(posix.PRIO_PROCESS, os.getpid(), -20)
    optimization.do_optimization(file_path, file_name, reports_path, outputs_path)

def report_server_run(message):
    Logging.Logging.write_log_to_file("Read message from aws queue")

    try:
        [email, file_name, selected_headers] = Aws.deserialize_message(message)
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))
        return False

    file_path = ""
    try:
        [file_path, reports_path, outputs_path] = GoogleStorage.download_file(file_name, email)
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))
        return False

    Logging.Logging.write_log_to_file("Data file downloaded : " + file_name)

    # Calculations
    try:
        p = Process(target=selectable_feature, args=(file_path, file_name.split("/")[-1], selected_headers, reports_path, outputs_path,))
        p.start()
        p.join()
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))
        return False

    Logging.Logging.write_log_to_file("Feature selection completed")

    try:
        p2 = Process(target=optimization, args=(file_path, file_name.split("/")[-1], reports_path, outputs_path,))
        p2.start()
        p2.join()
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))
        return False

    Logging.Logging.write_log_to_file("Optimization completed")

    # Google upload results
    while True:
        try:
            GoogleStorage.upload_results(email)
            break
        except Exception as e:
            Logging.Logging.write_log_to_file(str(e))

    # Delete local files
    try:
        GoogleStorage.delete_local_dir()
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))

    try:
        Aws.send_report_completed_message(message.body)
        Logging.Logging.write_log_to_file("Report completed message sent to aws")
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))

    Aws.try_to_delete_message(message, config.REPORT_QUEUE_NAME)
    return True

def main():
    Logging.Logging.init()
    Logging.Logging.write_log_to_file("Report Server Started")
    try:
        Logging.Logging.remoteLog("Report_Server_Alive", "Report_Server_Alive ID : " + config.REPORT_INSTANCE_ID)
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))

    start_time = time.time()
    while True:

        end_time = time.time()
        if end_time - start_time > 3600:
            try:
                Logging.Logging.remoteLog("Report_Server_Alive", "Report_Server_Alive ID : " + config.REPORT_INSTANCE_ID)
            except Exception as e:
                Logging.Logging.write_log_to_file(str(e))
            start_time = time.time()

        Logging.Logging.write_log_to_file_flush()

        try:
            message = Aws.read_queue(config.REPORT_QUEUE_NAME)
        except Exception as e:
            Logging.Logging.write_log_to_file(str(e))
            continue

        if message is not None:
            result = report_server_run(message)
            if(result == False):
                Aws.add_message_to_exception_queue(message, config.REPORT_QUEUE_NAME, config.REPORT_EXCEPTION_QUEUE_NAME)
        else:
            if Timing.Timing.DoShutDown(Logging.Logging.write_log_to_file, config.REPORT_INSTANCE_ID):
                Logging.Logging.write_log_to_file("Shutting down the instance")
                try:
                    Aws.stop_instance(config.REPORT_INSTANCE_ID)
                    Logging.Logging.write_log_to_file_flush()
                except Exception as e:
                    Logging.Logging.write_log_to_file(str(e))



if __name__ == "__main__":
    main()
