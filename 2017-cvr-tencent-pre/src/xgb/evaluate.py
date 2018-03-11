# -*- encoding:utf-8 -*-

import scipy as sp
import sys
from csv import DictReader
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from datetime import datetime

path = '../../output/xgb/'


threshold = 0.05

label_path = path + 'validation.csv'
predict_path = path + 'submission.csv'


def logloss1(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll

label_reader = DictReader(open(label_path))
predict_reader = DictReader(open(predict_path))

count = 0
y_true = []
y_pred = []
y_scores = []
for t, row in enumerate(label_reader):
    predict = predict_reader.__next__()
    actual = float(row['label'])
    predicted = float(predict['prob'])
    y_true.append(actual)
    y_scores.append(predicted)

    # 大于阈值的即视为点击
    if predicted >= threshold:
        y_pred.append(1)
    else:
        y_pred.append(0)
    count += 1

# 计算性能指标
auc = roc_auc_score(y_true, y_scores)

ll = logloss1(y_true, y_scores)


print('auc: {0}\n'.format(auc))
print('ll:{0}\n'.format(ll))

result = open(path + 'details.txt', 'a+')
result.write('------------------------------------------------------------\n')
result.write(str(datetime.now()) + '\n')
result.write('Total instances: {count}\n\n\nValidation File: {vafile}\n\nPrediction file: {prefile}\n\n'
             .format(count=count, vafile=label_path, prefile=predict_path))

result.write('auc: {0}\n\nll:{1}\n\n'.format(auc, ll))
result.write('-------------------------------------------------------------\n\n')


file = open(path + 'validation.csv', 'r')
next(file)
clks = 0
ids = []
for line in file:
    id = line.split(',')[0]
    clk = line.split(',')[1]
    if int(clk) == 1:
        clks += 1
        ids.append(id)
file.close()
print('总点击数:', clks)


f2 = open(path + 'submission.csv', 'r')
next(f2)
pred = 0
true = 0
for line in f2:
    id = line.split(',')[0]
    ctr = line.split(',')[1]
    if float(ctr) >= threshold:
        pred += 1
        if id in ids:
            true += 1

f2.close()
print('预测为1的个数:', pred)
print('真实为1的个数:', true)
result.write('总点击数: {0}\n'.format(clks))
result.write('预测为1的个数: {0}\n'.format(pred))
result.write('真实为1的个数: {0}\n'.format(true))
result.close()


