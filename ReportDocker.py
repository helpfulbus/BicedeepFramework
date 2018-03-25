# <copyright company="Bicedeep, Inc.">
# Copyright (c) 2016-2018 All Rights Reserved
# </copyright>

import sys
from ReportServerAI import report
from ReportServerAI import optimization

filename = sys.argv[1]
selectedHeaders = []

for i in range(2, len(sys.argv)):
    selectedHeaders.append(sys.argv[i])

report.create_report(file_path, file_name, selected_headers, reports_path, outputs_path, email, file_name_full)
optimization.do_optimization(file_path, file_name, reports_path, outputs_path, email, file_name_full)