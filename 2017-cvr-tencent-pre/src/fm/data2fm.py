# _*_ coding: utf-8 _*_

import collections
from csv import DictReader
from datetime import datetime

train_path = '../../data/train-ctr.csv'
test_path = '../../data/test-ctr.csv'
train_fm = '../../output/fm/train.fm'
test_fm = '../../output/fm/test.fm'

field = ['clickTime', 'creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory']

table = collections.defaultdict(lambda: 0)


def getIndices(key):
    indices = table.get(key)
    if indices is None:
        indices = len(table)
        table[key] = indices
    return indices

with open(train_fm, 'w') as outfile:
    for e, row in enumerate(DictReader(open(train_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    k = k + '-ctr'
                    features.append('{0}:{1}'.format(getIndices(k), v))

        if e % 100000 == 0:
            print(datetime.now(), 'creating train.fm...', e)
        outfile.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))

with open(test_fm, 'w') as f1:
    for t, row in enumerate(DictReader(open(test_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    k = k + '-ctr'
                    features.append('{0}:{1}'.format(getIndices(k), v))
        if t % 100000 == 0:
            print(datetime.now(), 'creating validation data and test.fm...', t)
        f1.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))



