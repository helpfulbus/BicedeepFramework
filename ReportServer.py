from Common import Timing
from Common import GoogleStorage
from Common import Aws


def main():
    while True:
        if Timing.Timing.DoShutDown():
            file_name = "h@h.com/data/population.csv"
            instance_id = "i-0bb171b2b649b9c2b"
            GoogleStorage.download_file(file_name)

            # Do Calculation
            
            Aws.send_report_completed_message(file_name)
            Aws.stop_instance(instance_id)


if __name__ == "__main__":
    main()
