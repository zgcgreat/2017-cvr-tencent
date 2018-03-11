# _*_ coding: utf-8 _*_

"""
label,clickTime,creativeID,positionID,connectionType,telecomsOperator,age,gender,education,marriageStatus,haveBaby,
hometown,residence,hometown_city,hometown_province,residence_city,residence_province,sitesetID,positionType,adID,
camgaignID,advertiserID,appID,appPlatform,appCategory

"""

import collections
import csv
from datetime import datetime

field = ['creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'hour', 'cate_1', 'cate_2']

header = 'label,creativeID-ctr,positionID-ctr,connectionType-ctr,telecomsOperator-ctr,age-ctr,' \
         'gender-ctr,education-ctr,marriageStatus-ctr,haveBaby-ctr,hometown-ctr,residence-ctr,sitesetID-ctr,' \
         'positionType-ctr,adID-ctr,camgaignID-ctr,advertiserID-ctr,appID-ctr,appPlatform-ctr,hour-ctr' \
         ',cate_1-ctr, cate_2-ctr'


def tr_feat(start_date, fo):
    # 创建一个默认字典
    counts = collections.defaultdict(lambda: [0, 0, 0])
    count = 0
    for i, row in enumerate(csv.DictReader(open('../../data/train.csv')), start=1):
        label = row['label']
        date = int(row['date'])

        if start_date - 5 <= date < start_date:
            for feat in field:
                value = row[feat]
                if label == '0':
                    counts[feat + '-' + value][0] += 1
                else:
                    counts[feat + '-' + value][1] += 1
                counts[feat + '-' + value][2] += 1
            count += 1
        if date == start_date:
            print(date)
            break

    for i, row in enumerate(csv.DictReader(open('../../data/train.csv')), start=1):
        date = int(row['date'])
        if start_date == date:
            features = []

            for k in field:
                if len(k) > 0:
                    v = row[k]
                    key = k + '-' + v
                    if key in counts.keys():
                        if counts[key][2] > 0:
                            # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                            ratio = counts[key][1] / counts[key][2]
                            p = counts[key][2] / count
                            # features.append(str(round(float(ratio) * p, 5)))
                            features.append(str(round(float(p), 5)))
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
    count = 0
    for i, row in enumerate(csv.DictReader(open('../../data/train.csv')), start=1):
        label = row['label']
        date = int(row['date'])
        if 25 <= date < 30:

            for feat in field:
                value = row[feat]
                if label == '0':
                    counts[feat + '-' + value][0] += 1
                else:
                    counts[feat + '-' + value][1] += 1
                counts[feat + '-' + value][2] += 1
            count += 1

    for i, row in enumerate(csv.DictReader(open('../../data/test.csv')), start=1):
        features = []
        for k in field:
            if len(k) > 0:
                v = row[k]
                key = k + '-' + v
                if key in counts.keys():
                    if counts[key][2] > 0:
                        # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                        ratio = counts[key][1] / counts[key][2]
                        p = counts[key][2] / count
                        # features.append(str(round(float(ratio) * p, 5)))
                        features.append(str(round(float(p), 5)))
                else:
                    features.append(str(0))
        fo.write(row['label'] + ',' + ','.join(features) + '\n')


if __name__ == '__main__':

    ftrain = open('../../data/train-p.csv', 'w')
    ftrain.write(header + '\n')
    for date in range(28, 30):
        tr_feat(date, ftrain)

    print('31')
    ftest = open('../../data/test-p.csv', 'w')
    ftest.write(header + '\n')
    te_feat(ftest)

