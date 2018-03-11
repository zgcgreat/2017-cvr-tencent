import subprocess
from datetime import datetime

start = datetime.now()

for i in range(1, 6):
    print('第{0}次...\n'.format(i))
    cmd = 'python3 xgb_gbdt.py {0}'.format(i)
    subprocess.call(cmd, shell=True)

print('时间:', datetime.now()-start)

# for i in range(1, 6):
#     print('第{0}次...\n'.format(i))
#     cmd = 'python3 用已知模型预测.py {0}'.format(i)
#     subprocess.call(cmd, shell=True)