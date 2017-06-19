import datetime
import subprocess

WAIT_UNTIL_MIN = 45

class Timing:
    def __init__(self):
        pass

    @staticmethod
    def DoShutDown():
        threshold = WAIT_UNTIL_MIN
        last_reboot_cmd = "last -F | grep reboot"
        reboot_times = subprocess.check_output(last_reboot_cmd, shell=True)
        reboot_times_split = reboot_times.split(' ')

        last_start_time = datetime.datetime.strptime(reboot_times_split[10] + "-" +
                                                     reboot_times_split[11] + "-" +
                                                     reboot_times_split[12] + "-" +
                                                     reboot_times_split[13],
                                                     "%b-%d-%H:%M:%S-%Y")

        time_diff_mod = ((datetime.datetime.now() - last_start_time).seconds // 60) % 60
        if time_diff_mod > threshold:
            return True
        else:
            return False
