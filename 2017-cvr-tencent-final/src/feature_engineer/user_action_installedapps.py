from csv import DictReader
import collections
import pandas as pd

'''
统计app_action.csv和appinstalled.csv中用户安装app的数量
'''
input_path = '../../data_ori'
output_path = '../../data'

user_app_action = '{0}/user_app_actions.csv'.format(input_path)
user_installedapp = '{0}/user_installedapps.csv'.format(input_path)

app_action = collections.defaultdict(lambda: 0)

installed_app = collections.defaultdict(lambda: 0)

for t, row in enumerate(DictReader(open(user_app_action, 'r')), start=1):
    # app_action[row['userID'] + '_' + row['appID']] += 1
    app_action[row['userID']] += 1
    if t % 1000000 == 0:
        print('Line processed:', t)


for t, row in enumerate(DictReader(open(user_installedapp, 'r')), start=1):
    # installed_app[row['userID'] + '_' + row['appID']] += 1
    installed_app[row['userID']] += 1
    if t % 1000000 == 0:
        print('Line processed:', t)


with open('../../output/feature_data/app_action_cnt.csv', 'w') as fo:
    fo.write('userID,app_action_cnt\n')
    for key in app_action.keys():
        fo.write(key + ',' + str(app_action[key]) + '\n')

with open('../../output/feature_data/installed_app_cnt.csv', 'w') as fo:
    fo.write('userID,installed_app_cnt\n')
    for key in installed_app.keys():
        fo.write(key + ',' + str(installed_app[key]) + '\n')


