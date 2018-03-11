# _*_ coding: utf-8 _*_

import math
import sys
import subprocess

data_path = '../../output/fm/'
result_path = '../../output/fm/'

cmd = './libFM -task r -train {train} -test {test} -out {out} -dim \'1,1,8\' -iter 50 -validation {train}'.format(
    train=data_path + 'train.fm', test=data_path + 'test.fm', out=result_path + 'preds.txt')
subprocess.call(cmd, shell=True, stdout=sys.stdout)


def zygmoid(x):
    return 1 / (1 + math.exp(-x))


with open(result_path + 'submission.csv', 'w') as outfile:
    outfile.write('instanceID,prob\n')
    t = 0
    for line in open(result_path + 'preds.txt'):
        outfile.write('{0},{1}\n'.format(t, zygmoid(float(line.rstrip()))))
        t += 1

# cmd = 'rm {0}preds.txt'.format(result_path)
# subprocess.call(cmd, shell=True)

# with open(result_path + 'submission.csv', 'w') as outfile:
#     outfile.write('Id,Predicted\n')
#     t = 0
#     for line in open(result_path + 'preds.txt'):
#         outfile.write('{0},{1}\n'.format(t, float(line.rstrip())))
#         t += 1
