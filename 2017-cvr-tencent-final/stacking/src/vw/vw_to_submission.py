# _*_ coding: utf-8 _*_

import math
import sys

'''
将预测值转换为0-1之间的数
'''

path = sys.argv[1]

submission = path + 'submission.csv'
preds = path + 'preds.txt'


def sigmod(x):
    return 1 / (1 + math.exp(-x))


with open(submission, 'w') as outfile:
    outfile.write('instanceID,prob\n')
    for t, line in enumerate(open(preds, 'r')):
        row = line.strip().split(' ')
        pro = sigmod(float(line))
        outfile.write('%s,%f\n' % (t, pro))
