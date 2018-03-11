import subprocess

cmd = 'python3 tr-te-data.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 frq_features.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 statical_feature.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 merge_featured_data.py'
subprocess.call(cmd, shell=True)

