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
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'hour', 'cate_1', 'cate_2',
         'hometown_city', 'hometown_province', 'residence_city', 'residence_province']

# com_field = ['positionID', 'clickTime', 'creativeID', 'adID', 'camgaignID', 'advertiserID', 'hometown',
#              'residence', 'age', 'connectionType']
# com_field = field
com_field1 = ['positionID', 'appID', 'hour']
com_field = ['age', 'gender', 'marriageStatus']
# fi = open('../../output/xgb/feat_importance.csv', 'r')
# next(fi)
# for t, line in enumerate(fi, start=1):
#     feat = line.split(',')[0]
#     com_field.append(feat)
#     if t == 14:
#         break
# fi.close()
# print(com_field)


def com_feat(row):
    com_feats = {}
    for fa in com_field1:
        for fb in com_field:
            # if fa != fb and com_field.index(fa) < com_field.index(fb):
            com_feats[fa + '_' + fb] = row[fa] + row[fb]  #
    return com_feats


def fo_row(label, ctr_feats, clk_feats, cnv_feats):
    row = label
    for k in ctr_feats.keys():
        row += ',' + str(ctr_feats[k])
    for k in clk_feats.keys():
        row += ',' + str(clk_feats[k])
    for k in cnv_feats.keys():
        row += ',' + str(cnv_feats[k])
    return row


def tr_feat(start_date, fo):
    # 创建一个默认字典
    counts = collections.defaultdict(lambda: [0, 0, 0])
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
        date = int(row['date'])

        if start_date == date:
            features = {}
            clk_feats = {}
            cnv_feats = {}

            for k in field:
                if len(k) > 0:
                    v = row[k]
                    key = k + '-' + v
                    if key in counts.keys():
                        if counts[key][2] > 0:
                            # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                            ratio = counts[key][1] / counts[key][2]
                            features[k] = (str(round(float(ratio), 5)))
                            clk_feats[k] = counts[key][2]
                            cnv_feats[k] = counts[key][1]

                        else:
                            features[k] = (str(0))
                            clk_feats[k] = (str(0))
                            cnv_feats[k] = (str(0))

                    else:
                        features[k] = (str(0))
                        clk_feats[k] = (str(0))
                        cnv_feats[k] = (str(0))

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
                            clk_feats[k] = counts[key][2]
                            cnv_feats[k] = counts[key][1]

                        else:
                            features[k] = (str(0))
                            clk_feats[k] = (str(0))
                            cnv_feats[k] = (str(0))

                    else:
                        features[k] = (str(0))
                        clk_feats[k] = (str(0))
                        cnv_feats[k] = (str(0))

            features = {}
            outrow = fo_row(row['label'], features, clk_feats, cnv_feats)

            fo.write(outrow + '\n')

        if date > start_date:
            break


def te_feat(fo):
    # 创建一个默认字典
    counts = collections.defaultdict(lambda: [0, 0, 0])
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

            com_feats = com_feat(row)
            for feat in com_feats.keys():
                value = com_feats[feat]
                if label == '0':
                    counts[feat + '-' + value][0] += 1
                else:
                    counts[feat + '-' + value][1] += 1
                counts[feat + '-' + value][2] += 1

    for i, row in enumerate(csv.DictReader(open('../../data/test.csv')), start=1):
        features = {}
        clk_feats = {}
        cnv_feats = {}

        for k in field:
            if len(k) > 0:
                v = row[k]
                key = k + '-' + v
                if key in counts.keys():
                    if counts[key][2] > 0:
                        # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                        ratio = counts[key][1] / counts[key][2]
                        features[k] = (str(round(float(ratio), 5)))
                        clk_feats[k] = counts[key][2]
                        cnv_feats[k] = counts[key][1]

                    else:
                        features[k] = (str(0))
                        clk_feats[k] = (str(0))
                        cnv_feats[k] = (str(0))

                else:
                    features[k] = (str(0))
                    clk_feats[k] = (str(0))
                    cnv_feats[k] = (str(0))

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
                        clk_feats[k] = counts[key][2]
                        cnv_feats[k] = counts[key][1]

                    else:
                        features[k] = (str(0))
                        clk_feats[k] = (str(0))
                        cnv_feats[k] = (str(0))

                else:
                    features[k] = (str(0))
                    clk_feats[k] = (str(0))
                    cnv_feats[k] = (str(0))

        features = {}
        outrow = fo_row(row['label'], features, clk_feats, cnv_feats)
        fo.write(outrow + '\n')


def get_header():
    header = 'label'
    # for feat in field:
    #     header += ',' + feat + '-ctr'
    #
    # for fa in com_field:
    #     for fb in com_field:
    #         if fa != fb and com_field.index(fa) < com_field.index(fb):
    #             header += ',' + fa + '_' + fb + '-ctr'

    for feat in field:
        header += ',' + feat + '-clk'

    for fa in com_field1:
        for fb in com_field:
            # if fa != fb and com_field.index(fa) < com_field.index(fb):
            header += ',' + fa + '_' + fb + '-clk'
    for feat in field:
        header += ',' + feat + '-cnv'

    for fa in com_field1:
        for fb in com_field:
            # if fa != fb and com_field.index(fa) < com_field.index(fb):
            header += ',' + fa + '_' + fb + '-cnv'
    return header


if __name__ == '__main__':
    from datetime import datetime

    start = datetime.now()
    header = get_header()
    print(header)
    ftrain = open('../../output/feature_data/tr_clk_cnv.csv', 'w')
    ftrain.write(header + '\n')
    for date in range(28, 30):
        tr_feat(date, ftrain)

    print('31')
    ftest = open('../../output/feature_data/te_clk_cnv.csv', 'w')
    ftest.write(header + '\n')
    te_feat(ftest)
    print(datetime.now() - start)
