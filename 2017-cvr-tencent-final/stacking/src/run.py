import subprocess

cmd = 'python3 ./data_process/split_data.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 ./lgb/run.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 ./xgb/run.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 ./新特征.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 xgb_stacking.py'
subprocess.call(cmd, shell=True)
