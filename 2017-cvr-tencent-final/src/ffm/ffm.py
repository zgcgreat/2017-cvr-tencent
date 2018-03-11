import subprocess
from datetime import datetime


path = '../../output/ffm/'

start = datetime.now()

# 训练
cmd = '~/ffm/ffm-train -p {save}train.ffm -l 0.0000002 -k 4 -t 50 -s 4 -r 0.2 --auto-stop {save}train.ffm ' \
      '{save}model'.format(save=path)
subprocess.call(cmd, shell=True)
# 预测
cmd = '~/ffm/ffm-predict {save}test.ffm {save}model {save}test.out'.format(save=path)
subprocess.call(cmd, shell=True)

with open(path + 'submission.csv', 'w') as fo:
    fo.write('instanceID,prob\n')
    for i, row in enumerate(open(path + 'test.out'), start=1):
        fo.write('{0},{1}'.format(i, row))

# cmd = 'rm {path}model {path}test.out'.format(path=path)
# subprocess.call(cmd, shell=True)

print('时间: {0}'.format(datetime.now() - start))
