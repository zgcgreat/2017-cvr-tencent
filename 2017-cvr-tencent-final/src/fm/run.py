# -*- encoding:utf-8 -*-
import subprocess

path = '../../data/data_no_header/'
cmd = '~/ffm/triple_format_to_libfm.pl -in {path}train.csv,{path}test.csv -target 0 -separator ","'.format(path=path)
subprocess.call(cmd, shell=True)

# cmd = 'python3 data2fm.py'
# subprocess.call(cmd, shell=True)
#
cmd = 'python3 libfm.py'
subprocess.call(cmd, shell=True)

# cmd = 'python3 evaluate.py'
# subprocess.call(cmd, shell=True)
