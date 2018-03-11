
import os
import shutil
import subprocess

path = '../rr/'
# 先删除已存在的结果文件目录, 这样增加FOLD不会有问题
# if os.path.exists(path):
#     shutil.rmtree(path)
# # 建立目录
# if not os.path.exists(path):
#     for i in range(2):
#         os.makedirs(path + 'split_{0}/'.format(i))

cmd = 'rm -rf {0}'.format(path)
subprocess.call(cmd, shell=True)
