# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression as LR
from sklearn.cross_validation import cross_val_score
import scipy as sp
from sklearn.cross_validation import train_test_split

train = pd.read_csv("../../data/tr-ctr-cnt.csv", nrows=1000)
train = train.fillna(0)
X = train.iloc[:, 1:]
Y = train.iloc[:, 0]
# print(train.head())
test = pd.read_csv("../../data/te-ctr-cnt.csv", nrows=10)
test = test.fillna(0)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
lr = LR()
model = lr.fit(x_train, y_train)
for i in range(5):
    model.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    # print(np.mean(scores), scores)
    pre = model.predict(x_test)
    # print(pre)
# 输出测试集的概率
pred = model.predict_proba(x_test)
# print(pred)
# print(pred.shape)
a = pred[:, 1]
# print(a)
# print(len(a))
# print(a.shape)
# print(type(pred))
# print(pred.shape)
# print(pre)

import scipy as sp


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = -sp.mean(act * sp.log(pred) + sp.subtract(1, act) * sp.log(1 - pred))
    return ll


loss = logloss(y_test, a)
print(loss)
# 求出结果
X_test = test.iloc[:, 1:]
pre = model.predict_proba(X_test)
# print(pre)
# print(sum(pre))
b = pre[:, 1]
# print(b)
f = open("submission.csv", 'w')
f.write('instanceID,prob\n')
for t, x in enumerate(b, start=1):
  f.write(str(t) + ',' + str(x) + '\n')
f.close()
