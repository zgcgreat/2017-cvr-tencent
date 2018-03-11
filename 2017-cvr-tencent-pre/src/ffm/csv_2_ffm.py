import collections
from csv import DictReader
from datetime import datetime

train_path = '../../data/train_merged.csv'
test_path = '../../data/test_merged.csv'
train_ffm = '../../output/ffm/train.ffm'
test_ffm = '../../output/ffm/test.ffm'
vali_path = '../../output/ffm/validation.csv'

field = ['clickTime', 'creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory']

ad_features = ['advertiserID', 'camgaignID', 'adID', 'creativeID', 'appID', 'appCategory', 'appPlatform']
user_features = ['userID', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence']
context_features = ['positionID', 'sitesetID', 'positionType', 'connectionType', 'telecomsOperator']

table = collections.defaultdict(lambda: 0)


# 为特征名建立编号, filed
def field_index(x):
    index = field.index(x)
    return index

# def field_index(x):
#     if x in ad_features:
#         return 1
#     if x in user_features:
#         return 2
#     if x in context_features:
#         return 3


def getIndices(key):
    indices = table.get(key)
    if indices is None:
        indices = len(table)
        table[key] = indices
    return indices


feature_indices = set()
with open(train_ffm, 'w') as outfile:
    for e, row in enumerate(DictReader(open(train_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    idx = field_index(k)
                    kv = k + ':' + v
                    features.append('{0}:{1}:1'.format(idx, getIndices(kv)))
                    feature_indices.add(kv + '\t' + str(getIndices(kv)))
            if 'ctr' in k:
                idx = 0
                if k.split('-')[0] in field:
                    idx = field_index(k.split('-')[0])
                features.append('{0}:{1}:{2}'.format(idx, getIndices(k), v))
        if e % 100000 == 0:
            print(datetime.now(), 'creating train.ffm...', e)
        outfile.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))

with open(test_ffm, 'w') as f1, open(vali_path, 'w') as f2:
    f2.write('id,label' + '\n')
    for t, row in enumerate(DictReader(open(test_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    idx = field_index(k)
                    kv = k + ':' + v
                    if kv + '\t' + str(getIndices(kv)) in feature_indices:
                        features.append('{0}:{1}:1'.format(idx, getIndices(kv)))
            if 'ctr' in k:
                idx = 0
                if k.split('-')[0] in field:
                    idx = field_index(k.split('-')[0])
                features.append('{0}:{1}:{2}'.format(idx, getIndices(k), v))
        if t % 100000 == 0:
            print(datetime.now(), 'creating validation data and test.ffm...', t)
        f1.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))
        f2.write(str(t) + ',' + row['label'] + '\n')

f1.close()
f2.close()
