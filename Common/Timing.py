import datetime
import subprocess
from dateutil.parser import parse

WAIT_UNTIL_MIN = 45

class Timing:
    def __init__(self):
        pass

    @staticmethod
    def DoShutDown():
        try:
            threshold = WAIT_UNTIL_MIN
            last_reboot_cmd = "last -F | grep reboot"
            reboot_times = subprocess.check_output(last_reboot_cmd, shell=True)
            reboot_times_split = reboot_times.decode().split('  ')

            dd = reboot_times_split[3] + " " + reboot_times_split[4]
            try:
                last_start_time = parse(dd)
            except Exception as e:
                last_start_time = parse(reboot_times_split[3])

            time_diff_mod = ((datetime.datetime.now() - last_start_time).seconds // 60) % 60
            if time_diff_mod > threshold:
                return True
            else:
                return False
        except Exception as e:
            return False

