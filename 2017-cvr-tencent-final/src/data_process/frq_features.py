# _*_ coding: utf-8 _*_

"""
统计频繁特征
"""

import collections
import csv
from datetime import datetime

field = ['creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'hour', 'cate_1', 'cate_2']

# 创建一个默认字典
counts = collections.defaultdict(lambda: [0, 0, 0])
print('start...')
for i, row in enumerate(csv.DictReader(open('../../data/train.csv')), start=1):
    label = row['label']
    if 17 <= int(row['date']) < 30:
        for feat in field:
            value = row[feat]
            if label == '0':
                counts[feat + ',' + value][0] += 1
            else:
                counts[feat + ',' + value][1] += 1
            counts[feat + ',' + value][2] += 1
        if i % 1000000 == 0:
            print(datetime.now(), 'Line read:', i)

output = open('../../data/fc.trav.csv', 'w')
output.write('Field,Value,Neg,Pos,Total,Ratio\n')

for key, (neg, pos, total) in sorted(counts.items(), key=lambda x: x[1][2]):
    if total < 0:
        continue
    ratio = round(float(pos) / total, 5)
    # print(key+','+str(neg)+','+str(pos)+','+str(total)+','+str(ratio))
    output.write(key + ',' + str(neg) + ',' + str(pos) + ',' + str(total) + ',' + str(ratio) + '\n')
