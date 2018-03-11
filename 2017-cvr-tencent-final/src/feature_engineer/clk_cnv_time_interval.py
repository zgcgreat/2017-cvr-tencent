from csv import DictReader


'''
当前点击时间与上次转化时间的差
'''

last_cnv_time = {}


def tr(filename, fo):
    for t, row in enumerate(DictReader(open('../../data/{0}.csv'.format(filename), 'r')), start=1):
        if 28 <= int(row['date']) <= 29:
            clk_time = int(row['date']) * 24 * 60 + int(row['hour']) * 60 + int(row['minute'])

            if row['userID'] in last_cnv_time.keys():
                time_interval = int(clk_time) - int(last_cnv_time[row['userID']])
            else:
                time_interval = ''

            if row['conversionTime'] != '':
                cnv_time = int(row['conversionTime'][:2]) * 24 * 60 + int(row['conversionTime'][2:4]) * 60 + \
                           int(row['conversionTime'][4:6])
                last_cnv_time[row['userID']] = cnv_time

            fo.write(row['label'] + ',' + row['date'] + ',' + row['userID'] + ',' + str(time_interval) + '\n')
        if t % 1000000 == 0:
            print('Line processed:', t)


def te(filename, fo):
    for t, row in enumerate(DictReader(open('../../data/{0}.csv'.format(filename), 'r')), start=1):
        clk_time = int(row['date']) * 24 * 60 + int(row['hour']) * 60 + int(row['minute'])

        if row['userID'] in last_cnv_time.keys():
            time_interval = int(clk_time) - int(last_cnv_time[row['userID']])
        else:
            time_interval = ''

        fo.write(row['label'] + ',' + row['date'] + ',' + row['userID'] + ',' + str(time_interval) + '\n')
        if t % 1000000 == 0:
            print('Line processed:', t)


with open('../../output/feature_data/tr_clk-cnv.csv', 'w') as fo:
    fo.write('label,date,userID,clk-cnv\n')
    tr('train', fo)

with open('../../output/feature_data/te_clk-cnv.csv', 'w') as fo:
    fo.write('label,date,userID,clk-cnv\n')
    te('test', fo)

print(last_cnv_time)
