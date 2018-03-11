# -*- encoding:utf-8 -*-

import scipy as sp
from csv import DictReader
from sklearn.metrics import roc_auc_score
from datetime import datetime

path = '../data/validation/'


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


xgb = []
lgb = []
sub_xgb = DictReader(open(path + 'xgb/submission.csv', 'r'))
for row in sub_xgb:
    xgb.append(row['Predicted'])

sub_lgb = DictReader(open(path + 'LightGBM/submission.csv', 'r'))
for row in sub_lgb:
    lgb.append(row['Predicted'])


pred = []
for i in range(len(xgb)):
    pred.append((0.95 * float(xgb[i]) + 0.05 * float(lgb[i])))

true = []
vafile = DictReader(open(path + 'xgb/validation.csv', 'r'))
for row in vafile:
    true.append(float(row['label']))

ll = logloss(true, pred)
auc = roc_auc_score(true, pred)
print('auc:', auc)
print('ll:', ll)

fo = open('../result/result.txt', 'a+')
fo.write(str(datetime.now()) + '\n')
fo.write('auc: {0}\n'.format(auc))
fo.write('ll: {0}\n\n'.format(ll))
fo.close()
