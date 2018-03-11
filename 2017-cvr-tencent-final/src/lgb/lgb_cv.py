# coding: utf-8

import json
import gc
import operator
import lightgbm as lgb
import pandas as pd
import scipy as sp
from numpy import *
from sklearn.metrics import roc_auc_score
from sklearn_pandas import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

field = ['clickTime', 'creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory']

path = '../../data/'

print('Load data...')

usecols = []
fi = open('../../output/xgb/feat_importance.csv', 'r')
next(fi)
for t, line in enumerate(fi):
    feat = line.split(',')[0]
    usecols.append(feat)
    if t == 15:
        break
fi.close()

tr_nrows = 9000000

train_data = pd.read_csv(path + 'traincp.csv', nrows=10000)
# train_data = train_data.tail(tr_nrows)

drop_list = []
for x in train_data.columns:
    if x not in usecols:
        drop_list.append(x)
drop_list = []

train_data = train_data.drop(drop_list, axis=1)

label_train = train_data['label']
train_data.drop(['label'], axis=1, inplace=True)

lgb_train = lgb.Dataset(train_data, label_train)


del train_data, label_train
gc.collect()

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'metric_freq': 1,
    'is_training_metric': 'true',
    'max_bin': 255,
    'num_leaves': 100,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'sample_for_bin': 50000,
    'sample_freq': 1,
    'colsample_bytree': 0.6,
    'reg_alpha': 1,
    'reg_lambda': 0,
    'min_split_gain': 0.5,
    'min_child_weight': 1,
    'min_child_samples': 10,
    'scale_pos_weight': 1,
    'tree_learner': 'serial',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 200,
    'max_depth': 8,
    'use_two_round_loading': 'false',
    'nthread': 4,
    'silent': 1
}

print('Start cv...')
# train

gbm = lgb.cv(params, lgb_train, num_boost_round=500, nfold=10, early_stopping_rounds=5, verbose_eval=None)

del lgb_train
gc.collect()

# print('Feature names:', gbm.feature_name())

# print('Calculate feature importances...')
# # feature importances
# print('Feature importances:', list(sorted(gbm.feature_importance(), key=operator.itemgetter(1), reverse=True)))


print('Start predicting...')
# predict
test_data = pd.read_csv(path + 'test-p.csv')
test_data = test_data.drop('label', axis=1)
test_data = test_data.drop(drop_list, axis=1)


y_pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)

fo = open('../../output/lgb/submission.csv', 'w')
fo.write('instanceID,prob\n')
for t, prob in enumerate(y_pred, start=1):
    fo.write(str(t) + ',' + str(prob) + '\n')
fo.close()

fo = open('../../output/lgb/feature_importance.csv', 'w')
for x in gbm.feature_name():
    fo.write(x + ',')
fo.write('\n')
for x in list(gbm.feature_importance()):
    fo.write(str(x) + ',')

fo.close()
