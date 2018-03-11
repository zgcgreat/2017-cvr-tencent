# _*_ coding: utf-8 _*_

import os
import shutil
import subprocess
from csv import DictReader

'''
交叉验证
'''

solution = 'xgb'
FOLD = 5

data_path = '../output/cross_validation_split/'
results_path = '../output/results/' + solution + '/'


# 先删除已存在的结果文件目录, 这样增加FOLD不会有问题
if os.path.exists(results_path):
    shutil.rmtree(results_path)
# 建立目录
if not os.path.exists(results_path):
    for i in range(FOLD):
        os.makedirs(results_path + 'split_{0}/'.format(i))

# 测试
for i in range(FOLD):
    print('running ' + solution + ', round: ' + str(i))
    cmd = 'python3 {solution}.py {data} {results}'.format(solution=solution, data=data_path + 'split_{0}/'
                                                         .format(i), results=results_path + 'split_{0}/'.format(i))
    subprocess.call(cmd, shell=True)

    # 计算性能评价指标
    cmd = 'python3 evaluate.py {data} {result}'.format(data=data_path + 'split_{0}/'.format(i),
                                                         result=results_path + 'split_{0}/'.format(i))
    subprocess.call(cmd, shell=True)


