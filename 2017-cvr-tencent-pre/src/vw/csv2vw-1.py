# _*_ coding: utf-8 _*_

import collections
from csv import DictReader
from datetime import datetime

train_path = '../../data/validation/train_merged.csv'
test_path = '../../data/validation/test_merged.csv'
train_vw = '../../data/validation/vw/train.vw'
test_vw = '../../data/validation/vw/test.vw'
vali_path = '../../data/validation/xgb/validation.csv'

field = ['clickTime', 'creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory']

drop_list = ['appID', 'camgaignID', 'advertiserID', 'telecomsOperator']

table = collections.defaultdict(lambda: 0)


# 为特征名建立编号, filed
def field_index(x):
    index = field.index(x)
    return index


def getIndices(key):
    indices = table.get(key)
    if indices is None:
        indices = len(table)
        table[key] = indices
    return indices

feature_indices = set()
with open(train_vw, 'w') as outfile:
    for e, row in enumerate(DictReader(open(train_path))):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    kv = k + ':' + v
                    features.append('{0}:1'.format(getIndices(kv)))
                    feature_indices.add(kv + '\t' + str(getIndices(kv)))
            #if 'num' in k:
                #features.append('{0}:{1}'.format(getIndices(k), v))
        if row['label'] == '1':
            label = 1
        else:
            label = -1
        if e % 100000 == 0:
            print(datetime.now(), 'creating train.vw...', e)
        outfile.write('{0} \' |f {1}\n'.format(label, ' '.join('{0}'.format(val) for val in features)))


with open(test_vw, 'w') as f1, open(vali_path, 'w') as f2:
    f2.write('id,label' + '\n')
    for t, row in enumerate(DictReader(open(test_path))):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    idx = field_index(k)
                    kv = k + ':' + v
                    if kv + '\t' + str(getIndices(kv)) in feature_indices:
                        features.append('{0}:1'.format(getIndices(kv)))
            #if 'num' in k:
                #features.append('{0}:{1}'.format(getIndices(k), v))
        if row['label'] == '1':
            label = 1
        else:
            label = -1
        if t % 100000 == 0:
            print(datetime.now(), 'creating validation data and test.vw...', t)
        f1.write('{0} \' |f {1}\n'.format(label, ' '.join('{0}'.format(val) for val in features)))
        f2.write(str(t) + ',' + row['label'] + '\n')

f1.close()
f2.close()
