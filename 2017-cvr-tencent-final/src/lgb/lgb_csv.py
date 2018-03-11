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

field = ['creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'hour', 'cate_1', 'cate_2']

path = '../../data/train_data/'

print('Load data...')

usecols = ['label']
fi = open('../../output/xgb/feat_importances.csv', 'r')
next(fi)
for t, line in enumerate(fi):
    feat = line.split(',')[0]
    usecols.append(feat)
    if t == 70:
        break
fi.close()

tr_nrows = 8000000
valid_nrows = 3000000
train_data = pd.read_csv(path + 'train.csv')
train_data = train_data.sample(frac=0.7, random_state=1)
# train_data = train_data.tail(tr_nrows)
# train_data = train_data[train_data['appPlatform'] == '0']
print(len(train_data))

drop_list = []
for x in train_data.columns:
    if x not in usecols:
        drop_list.append(x)
drop_list = []
print(len(drop_list))
train_data = train_data.drop(drop_list, axis=1)

label_train = train_data['label']
train_data.drop(['label'], axis=1, inplace=True)
print(label_train.unique())
# # 用29天验证
# val_X = train_data.tail(valid_nrows)
# val_y = label_train.tail(valid_nrows)
#
# X_train = train_data.head(len(train_data) - valid_nrows)
# y_train = label_train.head(len(train_data) - valid_nrows)
# del train_data, label_train
# gc.collect()

X_train, val_X, y_train, val_y = train_test_split(train_data, label_train, test_size=0.1, random_state=2)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(val_X, val_y, reference=lgb_train)

del X_train, y_train, val_X, val_y
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
    'nthread': 4
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=[lgb_train, lgb_val],
                early_stopping_rounds=5)

# lgb.cv(params, lgb_train, num_boost_round=500, nfold=10, early_stopping_rounds=5)

del lgb_train, lgb_val
gc.collect()

print('Feature names:', gbm.feature_name())

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))

# train_data = pd.read_csv(path + 'traincp.csv')
# train_data = train_data.tail(tr_nrows)
#
# train_data = train_data.drop(drop_list, axis=1)
#
# label_train = train_data['label']
# train_data.drop(['label'], axis=1, inplace=True)
# lgb_train = lgb.Dataset(train_data, label_train)
#
# del train_data, label_train
# gc.collect()
#
# print('Start training...')
# # train
# g = lgb.train(params,
#               lgb_train,
#               num_boost_round=gbm.best_iteration,
#               valid_sets=lgb_train)
#
# del lgb_train
# gc.collect()

print('Start predicting...')
# predict
test_data = pd.read_csv(path + 'test.csv')
test_data = test_data.drop('label', axis=1)
test_data = test_data.drop(drop_list, axis=1)

print(gbm.best_iteration)
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
