# _*_ coding: utf-8 _*_
import collections
from csv import DictReader

path = '../../data/'

fields = ['clickTime', 'creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
          'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
          'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory']


# THRESHOLD = 10

# [positive][total][index]
# table = collections.defaultdict(lambda: [0, 0, 0])

# for i, row in enumerate(DictReader(open('../../data/validation/train.csv'))):
#     label = row['label']
#     for field in fields:
#         value = row[field]
#         if label == '1':
#             table[field + '-' + value][0] += 1  # 0 特征值被点击的次数
#         table[field + '-' + value][1] += 1  # 特征值总的出现次数
#         table[field + '-' + value][2] = len(table)
#     if i % 100000 == 0:
#         print('Line processed: {0}'.format(i))


# 读取频繁特征
def read_frequent_feats(threshold):
    frequent_feats = {}
    # fc.trav.t10.txt为出现频率超过10的表
    for row in DictReader(open('../../data/fc.trav.csv')):
        if int(row['Total']) < threshold:
            continue
        frequent_feats[row['Field'] + '-' + row['Value']] = row['Ratio']
    return frequent_feats


threshold = 0

# 出现频率超过10次的特征集合
frequent_feats = read_frequent_feats(threshold)

with open(path + 'train-ccc.csv', 'w') as fo:
    header = 'label'
    for i in range(1, 21):
        header += ',' + 'num-{0}'.format(i)

    fo.write(header + '\n')

    for t, row in enumerate(DictReader(open(path + 'train.csv', 'r')), start=1):
        features = []
        if 26 <= int(row['clickTime'][:2]) < 31:
            for k in fields:
                if len(k) > 0:
                    v = row[k]
                    key = k + '-' + v
                    if key in frequent_feats.keys():
                        # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                        ratio = frequent_feats[key]
                        features.append(str(round(float(ratio), 5)))
                    else:
                        features.append('')

            fo.write(row['label'] + ',' + ','.join(features) + '\n')
            if t % 100000 == 0:
                print('Line processed: {0}'.format(t))

with open(path + 'test-ccc.csv', 'w') as fo:
    header = 'label'
    for i in range(1, 21):
        header += ',' + 'num-{0}'.format(i)
    fo.write(header + '\n')
    for t, row in enumerate(DictReader(open(path + 'test.csv', 'r')), start=1):
        features = []
        for k in fields:
            if len(k) > 0:
                v = row[k]
                key = k + '-' + v
                if key in frequent_feats.keys():
                    # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                    ratio = frequent_feats[key]
                    features.append(str(round(float(ratio), 5)))
                else:
                    features.append('')

        fo.write(row['label'] + ',' + ','.join(features) + '\n')
        if t % 100000 == 0:
            print('Line processed: {0}'.format(t))
