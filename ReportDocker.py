# <copyright company="Bicedeep, Inc.">
# Copyright (c) 2016-2018 All Rights Reserved
# </copyright>

import sys
import datetime
from ReportServerAI import report
from ReportServerAI import optimization

print("filename : " + sys.argv[1])
filename = sys.argv[1]
filepath = "./filesystem/data/" + filename
selectedHeaders = []

for i in range(2, len(sys.argv)):
    selectedHeaders.append(sys.argv[i])

f= open("./filesystem/logs/" + filename + "/status.log","w")

f.write(str(datetime.datetime.now()) + " Create Report Filtering Started\n")
f.flush()

try:
    report.create_report(filepath, filename, selectedHeaders, './filesystem/reports', './filesystem/models', '',filename)
except Exception as e:
    f.write(str(datetime.datetime.now()) + str(e))
    f.flush()

f.write(str(datetime.datetime.now()) + " Create Report Filtering Completed\n")
f.write(str(datetime.datetime.now()) + " Optimization Started\n")
f.flush()

try:
    optimization.do_optimization(filepath, filename, './filesystem/reports', './filesystem/models', '', filename)
except Exception as e:
    f.write(str(datetime.datetime.now()) + str(e))
    f.flush()

f.write(str(datetime.datetime.now()) + " Optimization Completed\n")
f.flush()

f.close()