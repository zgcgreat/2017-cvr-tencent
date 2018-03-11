from csv import DictReader

'''
转化时间与点击时间的差
'''

cnv_clk_interval = {}


def tr():
    with open('../../output/feature_data/tr_cnv-clk.csv', 'w') as fo:
        fo.write('label,clickTime,userID,cnv-clk\n')
        for t, row in enumerate(DictReader(open('../../data_ori/train.csv', 'r')), start=1):
            clk_time = int(row['clickTime'][:2]) * 24 * 60 + int(row['clickTime'][2:4]) * 60 + int(row['clickTime'][4:6])
            if row['conversionTime'] != '':
                cnv_time = int(row['conversionTime'][:2]) * 24 * 60 + int(row['conversionTime'][2:4]) * 60 + \
                           int(row['conversionTime'][4:6])
                time_interval = cnv_time - clk_time
                cnv_clk_interval[row['userID']] = time_interval
            else:
                time_interval = ''
            fo.write(row['label'] + ',' + row['clickTime'] + ',' + row['userID'] + ',' + str(time_interval) + '\n')
            if t % 1000000 == 0:
                print('Line processed:', t)


def te():
    with open('../../output/feature_data/te_cnv-clk.csv', 'w') as fo:
        fo.write('label,clickTime,userID,cnv-clk\n')
        for t, row in enumerate(DictReader(open('../../data_ori/test.csv', 'r')), start=1):
            if row['userID'] in cnv_clk_interval.keys():
                time_interval = cnv_clk_interval[row['userID']]
            else:
                time_interval = ''

            fo.write(row['label'] + ',' + row['clickTime'] + ',' + row['userID'] + ',' + str(time_interval) + '\n')


if __name__ == '__main__':
    tr()

    te()

