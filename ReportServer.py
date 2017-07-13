from Common import Timing
from Common import GoogleStorage
from Common import Aws
from Common import Logging
from ReportServerAI import report
import config


def main():
    Logging.Logging.write_log_to_file("Report Server Started")
    while True:

        try:
            message = Aws.read_report_queue()
        except Exception as e:
            Logging.Logging.write_log_to_file(str(e))
            continue

        if message is not None:
            Logging.Logging.write_log_to_file("Read message from aws queue")

            try:
                [email, file_name, selected_headers] = Aws.deserialize_message(message)
            except Exception as e:
                Logging.Logging.write_log_to_file(str(e))
                continue

            file_path = ""
            try:
                [file_path, reports_path, outputs_path] = GoogleStorage.download_file(file_name, email)
            except Exception as e:
                Logging.Logging.write_log_to_file(str(e))
                continue

            Logging.Logging.write_log_to_file("Data file downloaded : " + file_name)

            # Calculations
            try:
                report.create_report(file_path, file_name.split("/")[-1], selected_headers, reports_path, outputs_path)
            except Exception as e:
                Logging.Logging.write_log_to_file(str(e))
                continue

            # Google upload results
            try:
                GoogleStorage.upload_results(email)
            except:
                Logging.Logging.write_log_to_file(str(e))
                continue

            # Delete local files
            try:
                GoogleStorage.delete_local_dir()
            except:
                Logging.Logging.write_log_to_file(str(e))
                continue

            try:
                Aws.send_report_completed_message(message.body)
            except Exception as e:
                Logging.Logging.write_log_to_file(str(e))
                continue

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
                    continue
                continue

        else:
            if Timing.Timing.DoShutDown():
                Logging.Logging.write_log_to_file("Shutting down the instance")
                try:
                    Aws.stop_instance(config.REPORT_INSTANCE_ID)
                except Exception as e:
                    Logging.Logging.write_log_to_file(str(e))





if __name__ == "__main__":
    main()
