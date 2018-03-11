# -*- encoding:utf-8 -*-
import subprocess

cmd = 'python3 csv_2_libsvm.py'
subprocess.call(cmd, shell=True)

# cmd = 'python3 csv_2_svm.py'
# subprocess.call(cmd, shell=True)

# cmd = 'python3 xgb_lr.py'
# subprocess.call(cmd, shell=True)

cmd = 'python3 xgb_gbdt.py'

subprocess.call(cmd, shell=True)

cmd = 'python3 evaluate.py'
subprocess.call(cmd, shell=True)
