# _*_ coding: utf-8 _*_

import subprocess
import sys

data_path = '../../data/validation/'
result_path = '../../output/vw/'

# 训练
cmd = 'vw {path}train.vw -f {path}model --sgd -c -k --passes 300 -b 20 ' \
      '--loss_function logistic --early_terminate 10'.format(path=result_path)
subprocess.call(cmd, shell=True)

# 测试
cmd = 'vw {path}test.vw -t -i {path}model -p {path}preds.txt --loss_function logistic '.format(path=result_path)
subprocess.call(cmd, shell=True)

# # 删除中间数据文件
cmd = 'rm {path}train.vw.cache'.format(path=result_path)
subprocess.call(cmd, shell=True)

# 将预测值转换为0-1之间的数
cmd = 'python3 vw_to_submission.py {path}'.format(path=result_path)
subprocess.call(cmd, shell=True)
