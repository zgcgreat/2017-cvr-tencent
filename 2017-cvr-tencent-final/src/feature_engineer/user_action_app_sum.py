'''
当天之前用户安装的app数量,不加前16天的统计和加上前16天的统计
'''

import collections
from csv import DictReader

user_app_install_sum = {}
for row in DictReader(open('../../output/feature_data/installed_app_cnt.csv', 'r')):
    user_app_install_sum[row['userID']] = row['installed_app_cnt']


user_app_sum = {}

for cur_date in range(24, 32):
    user_app_sum[str(cur_date)] = collections.defaultdict(lambda: 0)
    for row in DictReader(open('../../data_ori/user_app_actions.csv', 'r')):
        date = int(row['installTime'][:2])
        if date < cur_date:
            user_app_sum[str(cur_date)][row['userID']] += 1

    print(cur_date)

print(user_app_sum['24'])
print('train set...')
with open('../../output/feature_data/tr_user_app_sum.csv', 'w') as fo:
    fo.write('label,clickTime,userID,user_app_action_sum,user_app_action_installed_sum\n')
    for row in DictReader(open('../../data/train.csv', 'r')):
        date = int(row['date'])
        userID = row['userID']
        # app_action_count = 0  # 从第17天开始， 到当天前一天用户安装app的数量
        # app_action_installed_count = 0  # 前16天用户安装app的数量 + 从第17天开始， 到当天前一天用户安装app的数量
        if 28 <= date <= 29:
            if userID in user_app_sum[str(date)].keys():
                app_action_count = user_app_sum[str(date)][userID]   # 从第17天开始， 到当天前一天用户安装app的数量
            else:
                app_action_count = 0

            if userID in user_app_install_sum.keys():
                app_installed_count = int(user_app_install_sum[userID])  # 前16天用户安装app的数量
            else:
                app_installed_count = 0

            app_action_installed_count = app_action_count + app_installed_count  # 前16天用户安装app的数量 + 从第17天开始， 到当天前一天用户安装app的数量

            if app_action_count == 0:
                app_action_count = -1
            if app_installed_count == 0:
                app_installed_count = -1
            if app_action_installed_count == 0:
                app_action_installed_count = -1

            fo.write(row['label'] + ',' + row['date'] + ',' + userID + ',' + str(app_action_count) + ',' +
                     str(app_action_installed_count) + '\n')

print('test set...')
with open('../../output/feature_data/te_user_app_sum.csv', 'w') as fo:
    fo.write('label,clickTime,userID,user_app_action_sum,user_app_action_installed_sum\n')
    for row in DictReader(open('../../data/test.csv', 'r')):
        date = int(row['date'])
        userID = row['userID']
        app_action_count = 0
        app_action_installed_count = 0
        if date == 31:
            if userID in user_app_sum[str(date)].keys():
                app_action_count = user_app_sum[str(date)][userID]
            else:
                app_action_count = 0

            if userID in user_app_install_sum.keys():
                app_installed_count = int(user_app_install_sum[userID])
            else:
                app_installed_count = 0

            app_action_installed_count = app_action_count + app_installed_count

            if app_action_count == 0:
                app_action_count = -1
            if app_installed_count == 0:
                app_installed_count = -1
            if app_action_installed_count == 0:
                app_action_installed_count = -1

            fo.write(row['label'] + ',' + row['date'] + ',' + userID + ',' + str(app_action_count) + ',' +
                     str(app_action_installed_count) + '\n')
