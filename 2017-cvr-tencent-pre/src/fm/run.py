# -*- encoding:utf-8 -*-
import subprocess

cmd = 'python3 data2fm.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 libfm.py'
subprocess.call(cmd, shell=True)

# cmd = 'python3 evaluate.py'
# subprocess.call(cmd, shell=True)
