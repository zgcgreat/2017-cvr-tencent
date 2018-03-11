# -*- encoding:utf-8 -*-
import subprocess

cmd = 'python3 csv_2_ffm.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 ffm.py'
subprocess.call(cmd, shell=True)

# cmd = 'python3 evaluate.py'
# subprocess.call(cmd, shell=True)
