from csv import DictReader

'''
用户两次点击时间的差
'''

last_time_userID = {}
last_time_userID_creativeID = {}
last_time_userID_appID = {}
last_time_creativeID_appID = {}


def fun(row, fo):
    date = row['date']
    hour = row['hour']
    minute = row['minute']

    time = int(date) * 24 * 60 + int(hour) * 60 + int(minute)

    if row['userID'] in last_time_userID.keys():
        user_time_interval = int(time) - int(last_time_userID[row['userID']])
    else:
        user_time_interval = ''

    if row['userID'] + row['creativeID'] in last_time_userID_creativeID.keys():
        uc_time_interval = int(time) - int(last_time_userID_creativeID[row['userID'] + row['creativeID']])
    else:
        uc_time_interval = ''

    if row['userID'] + row['appID'] in last_time_userID_appID.keys():
        ua_time_interval = int(time) - int(last_time_userID_appID[row['userID'] + row['appID']])
    else:
        ua_time_interval = ''

    if row['creativeID'] + row['appID'] in last_time_userID_appID.keys():
        ca_time_interval = int(time) - int(last_time_userID_appID[row['creativeID'] + row['appID']])
    else:
        ca_time_interval = ''

    last_time_userID[row['userID']] = time
    last_time_userID_creativeID[row['userID'] + row['creativeID']] = time
    last_time_userID_appID[row['userID'] + row['appID']] = time
    last_time_creativeID_appID[row['creativeID'] + row['appID']] = time

    fo.write(row['label'] + ',' + row['date'] + ',' + row['userID'] + ',' + str(user_time_interval) + ',' +
             str(uc_time_interval) + ',' + str(ua_time_interval) + ',' + str(ca_time_interval) + '\n')


def tr(fo):
    for t, row in enumerate(DictReader(open('../../data/train.csv', 'r')), start=1):
        if 28 <= int(row['date']) <= 29:
            fun(row, fo)
        if t % 1000000 == 0:
            print('Line processed:', t)


def te(fo):
    for t, row in enumerate(DictReader(open('../../data/test.csv', 'r')), start=1):
        fun(row, fo)
        if t % 1000000 == 0:
            print('Line processed:', t)

with open('../../output/feature_data/tr_time_interval.csv', 'w') as fo:
    header = 'label,clickTime,userID,user_time_interval,userID_creativeID_time_interval,userID_appID_time_interval,' \
             'creativeID_appID_time_interval'
    fo.write(header + '\n')
    tr(fo)

with open('../../output/feature_data/te_time_interval.csv', 'w') as fo:
    fo.write(header + '\n')
    te(fo)
