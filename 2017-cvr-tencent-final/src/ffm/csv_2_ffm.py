import collections
from csv import DictReader
from datetime import datetime

train_path = '../../data/train.csv'
test_path = '../../data/test.csv'
train_ffm = '../../output/ffm/train.ffm'
test_ffm = '../../output/ffm/test.ffm'
valid_ffm = '../../output/ffm/valid.ffm'

# field = ['creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
#          'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
#          'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'hour', 'cate_1', ' cate_2']
#
# ad_features = ['advertiserID', 'camgaignID', 'adID', 'creativeID', 'appID', 'appCategory', 'appPlatform']
# user_features = ['userID', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence']
# context_features = ['positionID', 'sitesetID', 'positionType', 'connectionType', 'telecomsOperator']
field = []
fi = open('../../output/ffm/feats.txt')

for x in next(fi).replace('\n', '').split(',')[1:]:

    field.append(x)
fi.close()
print(field)

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

with open(train_ffm, 'w') as ftr,  open(valid_ffm, 'w') as fva:
    for e, row in enumerate(DictReader(open(train_path)), start=1):
        if 28 <= int(row['date']) <= 29:
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
            ftr.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))

        if int(row['date']) == 29:
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
                print(datetime.now(), 'creating valid.ffm...', e)
            fva.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))


with open(test_ffm, 'w') as fo:
    for t, row in enumerate(DictReader(open(test_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    idx = field_index(k)
                    kv = k + ':' + v
                    # if kv + '\t' + str(getIndices(kv)) in feature_indices:
                    #     features.append('{0}:{1}:1'.format(idx, getIndices(kv)))
                    features.append('{0}:{1}:1'.format(idx, getIndices(kv)))
            if 'ctr' in k:
                idx = 0
                if k.split('-')[0] in field:
                    idx = field_index(k.split('-')[0])
                features.append('{0}:{1}:{2}'.format(idx, getIndices(k), v))
        if t % 100000 == 0:
            print(datetime.now(), 'creating validation data and test.ffm...', t)
        fo.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))


fo.close()

