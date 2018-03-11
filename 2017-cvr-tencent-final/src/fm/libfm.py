# _*_ coding: utf-8 _*_

import math
import sys
import subprocess

data_path = '../../data/data_no_header/'
result_path = '../../output/fm/'

cmd = '~/ffm/libFM -task r -train {train} -test {test} -out {out} -method sgda -learn_rate 0.1 -dim \'1,1,8\' -iter 500 '\
      '-validation {train} -verbosity 0'.format(train=data_path + 'train.csv.libfm', test=data_path + 'test.csv.libfm', out=result_path + 'preds.txt')
subprocess.call(cmd, shell=True, stdout=sys.stdout)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


with open(result_path + 'submission.csv', 'w') as outfile:
    outfile.write('instanceID,prob\n')
    for t, line in enumerate(open(result_path + 'preds.txt'), start=1):
        outfile.write('{0},{1}\n'.format(t, sigmoid(float(line.rstrip()))))

# cmd = 'rm {0}preds.txt'.format(result_path)
# subprocess.call(cmd, shell=True)


