# coding: utf-8

import gc
import sys
import numpy as np
import lightgbm as lgb
import pandas as pd
from numpy import *
from sklearn.model_selection import train_test_split

num = sys.argv[1]

path = '../../output/stack-data1/'

out_path = '../../output/results/lgb/'

print('Load data...')

# usecols = []
# fi = open('../../../output/xgb/feat_importance.csv', 'r')
# next(fi)
# for t, line in enumerate(fi):
#     feat = line.split(',')[0]
#     usecols.append(feat)
#     if t == 15:
#         break
# fi.close()

train_data = pd.read_csv(path + 'train{0}.csv'.format(num))

print(len(train_data))

# drop_list = []
# for x in train_data.columns:
#     if x not in usecols:
#         drop_list.append(x)
drop_list = []

train_data = train_data.drop(drop_list, axis=1)

label_train = train_data['label']
train_data.drop(['label'], axis=1, inplace=True)

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


del lgb_train, lgb_val
gc.collect()

print('Feature names:', gbm.feature_name())

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))

valid = pd.read_csv(path+'valid{0}.csv'.format(num))
valid = valid.drop(drop_list, axis=1)
label_valid = np.array(valid['label'])
valid.drop(['label'], axis=1, inplace=True)

val_pred = gbm.predict(valid, num_iteration=gbm.best_iteration)
del valid
gc.collect()

output = open(out_path + 'subval{0}.csv'.format(num), 'w')
output.write('label,lgb_prob\n')
for t, p in enumerate(val_pred, start=1):
    output.write('{0},{1}\n'.format(label_valid[t-1], p))
output.close()


print('Start predicting...')
# predict
test_data = pd.read_csv('../../../data/test1.csv')
test_data = test_data.drop('label', axis=1)
test_data = test_data.drop(drop_list, axis=1)

print(gbm.best_iteration)
y_pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)

fo = open('../../output/results/lgb/sub{0}.csv'.format(num), 'w')
fo.write('instanceID,lgb_prob\n')
for t, prob in enumerate(y_pred, start=1):
    fo.write(str(t) + ',' + str(prob) + '\n')
fo.close()

