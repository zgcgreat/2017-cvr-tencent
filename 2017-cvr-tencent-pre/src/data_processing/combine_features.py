# _*_ coding: utf-8 _*_

"""
label,clickTime,creativeID,positionID,connectionType,telecomsOperator,age,gender,education,marriageStatus,haveBaby,
hometown,residence,hometown_city,hometown_province,residence_city,residence_province,sitesetID,positionType,adID,
camgaignID,advertiserID,appID,appPlatform,appCategory

"""

import collections
import csv
from datetime import datetime

field = ['clickTime', 'creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory', 'appID']

com_field = ['positionID', 'clickTime', 'creativeID', 'adID', 'camgaignID', 'advertiserID', 'hometown',
             'residence', 'age', 'appCategory',  'connectionType']
# com_field = []

header = 'label,clickTime-ctr,creativeID-ctr,positionID-ctr,connectionType-ctr,telecomsOperator-ctr,age-ctr,' \
         'gender-ctr,education-ctr,marriageStatus-ctr,haveBaby-ctr,hometown-ctr,residence-ctr,sitesetID-ctr,' \
         'positionType-ctr,adID-ctr,camgaignID-ctr,advertiserID-ctr,appID-ctr,appPlatform-ctr,appCategory-ctr,' \
         'creat_pos-ctr'


def com_feat(row):
    com_feats = {}
    for fa in com_field:
        for fb in com_field:
            if fa != fb and com_field.index(fa) < com_field.index(fb):
                com_feats[fa + '_' + fb] = row[fa] + row[fb]  #
    return com_feats


def tr_feat(start_date, fo):
    # 创建一个默认字典
    counts = collections.defaultdict(lambda: [0, 0, 0])
    for i, row in enumerate(csv.DictReader(open('../../data/train.csv')), start=1):
        label = row['label']
        date = int(row['clickTime'][0:2])

        if start_date - 7 <= date < start_date:
            row['clickTime'] = row['clickTime'][2:4]
            for feat in field:
                value = row[feat]
                if label == '0':
                    counts[feat + '-' + value][0] += 1
                else:
                    counts[feat + '-' + value][1] += 1
                counts[feat + '-' + value][2] += 1

            com_feats = com_feat(row)

            for feat in com_feats.keys():
                value = com_feats[feat]

                if label == '0':
                    counts[feat + '-' + value][0] += 1
                else:
                    counts[feat + '-' + value][1] += 1
                counts[feat + '-' + value][2] += 1

        if date == start_date:
            print(date)
            break

    for i, row in enumerate(csv.DictReader(open('../../data/train.csv')), start=1):
        date = int(row['clickTime'][0:2])
        if start_date == date:
            features = {}
            row['clickTime'] = row['clickTime'][2:4]

            cur_row = row['label']

            for k in field:
                if len(k) > 0:
                    v = row[k]
                    key = k + '-' + v
                    if key in counts.keys():
                        if counts[key][2] > 0:
                            # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                            ratio = counts[key][1] / counts[key][2]
                            features[k] = (str(round(float(ratio), 5)))
                            cur_row += ',' + (str(round(float(ratio), 5)))
                        else:
                            features[k] = (str(0))
                            cur_row += ',' + (str(0))
                    else:
                        features[k] = (str(0))
                        cur_row += ',' + (str(0))

            com_feats = com_feat(row)
            for k in com_feats.keys():
                if len(k) > 0:
                    v = com_feats[k]
                    key = k + '-' + v
                    if key in counts.keys():
                        if counts[key][2] > 0:
                            # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                            ratio = counts[key][1] / counts[key][2]
                            features[k] = (str(round(float(ratio), 5)))
                            cur_row += ',' + (str(round(float(ratio), 5)))
                        else:
                            features[k] = (str(0))
                            cur_row += ',' + (str(0))
                    else:
                        features[k] = (str(0))
                        cur_row += ',' + (str(0))
            # fo.write(row['label'] + ',' + ','.join(features[k] for k in features.keys()) + '\n')
            # print(cur_row)
            fo.write(cur_row + '\n')
        if date > start_date:
            break


def te_feat(fo):
    # 创建一个默认字典
    counts = collections.defaultdict(lambda: [0, 0, 0])
    for i, row in enumerate(csv.DictReader(open('../../data/train.csv')), start=1):
        label = row['label']
        date = int(row['clickTime'][0:2])

        if 24 <= date < 30:
            row['clickTime'] = row['clickTime'][2:4]
            for feat in field:
                value = row[feat]
                if label == '0':
                    counts[feat + '-' + value][0] += 1
                else:
                    counts[feat + '-' + value][1] += 1
                counts[feat + '-' + value][2] += 1

            com_feats = com_feat(row)
            for feat in com_feats.keys():
                value = com_feats[feat]
                if label == '0':
                    counts[feat + '-' + value][0] += 1
                else:
                    counts[feat + '-' + value][1] += 1
                counts[feat + '-' + value][2] += 1

    for i, row in enumerate(csv.DictReader(open('../../data/test.csv')), start=1):
        features = []
        row['clickTime'] = row['clickTime'][2:4]

        cur_row = row['label']

        for k in field:
            if len(k) > 0:
                v = row[k]
                key = k + '-' + v
                if key in counts.keys():
                    if counts[key][2] > 0:
                        # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                        ratio = counts[key][1] / counts[key][2]
                        features.append(str(round(float(ratio), 5)))
                        cur_row += ',' + (str(round(float(ratio), 5)))
                else:
                    features.append(str(0))
                    cur_row += ',' + (str(0))

        com_feats = com_feat(row)
        for k in com_feats.keys():
            if len(k) > 0:
                v = com_feats[k]
                key = k + '-' + v
                if key in counts.keys():
                    if counts[key][2] > 0:
                        # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                        ratio = counts[key][1] / counts[key][2]
                        features.append(str(round(float(ratio), 5)))
                        cur_row += ',' + (str(round(float(ratio), 5)))
                else:
                    features.append(str(0))
                    cur_row += ',' + (str(0))
        # fo.write(row['label'] + ',' + ','.join(features) + '\n')
        fo.write(cur_row + '\n')


if __name__ == '__main__':
    from datetime import datetime

    start = datetime.now()
    header = 'label'
    for feat in field:
        header += ',' + feat

    for fa in com_field:
        for fb in com_field:
            if fa != fb and com_field.index(fa) < com_field.index(fb):
                header += ',' + fa + '_' + fb
    print(header)
    ftrain = open('../../data/trainctr.csv', 'w')
    ftrain.write(header + '\n')
    for date in range(24, 30):
        tr_feat(date, ftrain)

    print('31')
    ftest = open('../../data/testctr.csv', 'w')
    ftest.write(header + '\n')
    te_feat(ftest)
    print(datetime.now() - start)
