from multiprocessing import Process
import os
import posix
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
        return

    file_path = ""
    try:
        [file_path, reports_path, outputs_path] = GoogleStorage.download_file(file_name, email)
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))
        return

    Logging.Logging.write_log_to_file("Data file downloaded : " + file_name)

    # Calculations
    try:
        p = Process(target=selectable_feature, args=(file_path, file_name.split("/")[-1], selected_headers, reports_path, outputs_path,))
        p.start()
        p.join()
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))
        return

    Logging.Logging.write_log_to_file("Feature selection completed")

    try:
        p2 = Process(target=optimization, args=(file_path, file_name.split("/")[-1], reports_path, outputs_path,))
        p2.start()
        p2.join()
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))

    Logging.Logging.write_log_to_file("Optimization completed")

    # Google upload results
    try:
        GoogleStorage.upload_results(email)
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))
        return

    # Delete local files
    try:
        GoogleStorage.delete_local_dir()
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))
        return

    try:
        Aws.send_report_completed_message(message.body)
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))
        return

    Logging.Logging.write_log_to_file("Report completed message sent to aws")

    try:
        message.delete()
        Logging.Logging.write_log_to_file("Message Deleted")
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))

        try:
            message_re_read = Aws.read_report_queue()
            message_re_read.delete()
            Logging.Logging.write_log_to_file("Message Deleted")
        except Exception as e:
            Logging.Logging.write_log_to_file(str(e))
            return
        return


def main():
    Logging.Logging.init()
    Logging.Logging.write_log_to_file("Report Server Started")
    while True:
        Logging.Logging.write_log_to_file_flush()
        try:
            message = Aws.read_report_queue()
        except Exception as e:
            Logging.Logging.write_log_to_file(str(e))
            continue

        if message is not None:
            report_server_run(message)
        else:
            if Timing.Timing.DoShutDown():
                Logging.Logging.write_log_to_file("Shutting down the instance")
                try:
                    Aws.stop_instance(config.REPORT_INSTANCE_ID)
                    Logging.Logging.write_log_to_file_flush()
                except Exception as e:
                    Logging.Logging.write_log_to_file(str(e))





if __name__ == "__main__":
    main()
