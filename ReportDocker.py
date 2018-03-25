# <copyright company="Bicedeep, Inc.">
# Copyright (c) 2016-2018 All Rights Reserved
# </copyright>

import sys
from ReportServerAI import report
from ReportServerAI import optimization

filepath = sys.argv[1]
filename = filepath.split('/')[-1]
selectedHeaders = []

for i in range(2, len(sys.argv)):
    selectedHeaders.append(sys.argv[i])

report.create_report(filepath, filename, selectedHeaders, './filesystem/reports', './filesystem/models', '', filename)
optimization.do_optimization(filepath, filename, './filesystem/reports', './filesystem/models', '', filename)