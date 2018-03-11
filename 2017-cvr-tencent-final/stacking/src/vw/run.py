# _*_ coding: utf-8 _*_
import subprocess

cmd = 'python3 csv2vw.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 vw.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 evaluate.py'
subprocess.call(cmd, shell=True)
