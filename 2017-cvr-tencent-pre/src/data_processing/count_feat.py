# _*_ coding: utf-8 _*_

"""

"""

import collections
import csv
from datetime import datetime

field = ['clickTime', 'creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory']

header = 'label,clickTime-cnt,creativeID-cnt,positionID-cnt,connectionType-cnt,telecomsOperator-cnt,age-cnt,' \
         'gender-cnt,education-cnt,marriageStatus-cnt,haveBaby-cnt,hometown-cnt,residence-cnt,sitesetID-cnt,' \
         'positionType-cnt,adID-cnt,camgaignID-cnt,advertiserID-cnt,appID-cnt,appPlatform-cnt,appCategory-cnt'


def tr_feat(start_date, fo):
    # 创建一个默认字典
    counts = collections.defaultdict(lambda: [0, 0, 0])
    for i, row in enumerate(csv.DictReader(open('../../data/train.csv')), start=1):
        label = row['label']
        date = int(row['clickTime'][0:2])
        if start_date - 5 <= date < start_date:
            row['clickTime'] = row['clickTime'][2:4]
            for feat in field:
                value = row[feat]
                if label == '0':
                    counts[feat + '-' + value][0] += 1
                else:
                    counts[feat + '-' + value][1] += 1
                counts[feat + '-' + value][2] += 1
                # if i % 100000 == 0:
                #     print(datetime.now(), 'Line read:', i)
        if date == start_date:
            print(date)
            break

    for i, row in enumerate(csv.DictReader(open('../../data/train.csv')), start=1):
        date = int(row['clickTime'][0:2])
        if start_date == date:
            features = []
            row['clickTime'] = row['clickTime'][2:4]
            for k in field:
                if len(k) > 0:
                    v = row[k]
                    key = k + '-' + v
                    if key in counts.keys():
                        if counts[key][2] > 0:
                            # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                            pos_cnt = counts[key][1]
                            features.append(str(pos_cnt))
                        else:
                            features.append(str(0))
                    else:
                        features.append(str(0))
            fo.write(row['label'] + ',' + ','.join(features) + '\n')
        if date > start_date:
            break


def te_feat(fo):
    # 创建一个默认字典
    counts = collections.defaultdict(lambda: [0, 0, 0])
    for i, row in enumerate(csv.DictReader(open('../../data/train.csv')), start=1):
        label = row['label']
        date = int(row['clickTime'][0:2])
        if 26 <= date < 31:
            row['clickTime'] = row['clickTime'][2:4]
            for feat in field:
                value = row[feat]
                if label == '0':
                    counts[feat + '-' + value][0] += 1
                else:
                    counts[feat + '-' + value][1] += 1
                counts[feat + '-' + value][2] += 1
                # if i % 100000 == 0:
                #     print(datetime.now(), 'Line read:', i)

    for i, row in enumerate(csv.DictReader(open('../../data/test.csv')), start=1):
        features = []
        row['clickTime'] = row['clickTime'][2:4]
        for k in field:
            if len(k) > 0:
                v = row[k]
                key = k + '-' + v
                if key in counts.keys():
                    if counts[key][2] > 0:
                        # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                        pos_cnt = counts[key][1]
                        features.append(str(pos_cnt))
                else:
                    features.append(str(0))
        fo.write(row['label'] + ',' + ','.join(features) + '\n')


if __name__ == '__main__':

    ftrain = open('../../data/train-cnt.csv', 'w')
    ftrain.write(header + '\n')
    for date in range(22, 31):
        tr_feat(date, ftrain)

    print('test...')
    ftest = open('../../data/test-cnt.csv', 'w')
    ftest.write(header + '\n')
    te_feat(ftest)

