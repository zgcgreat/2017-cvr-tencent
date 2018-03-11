# _*_ coding: utf-8 _*_

import zipfile
import pandas as pd
import xgboost as xgb
import gc
import numpy as np
from sklearn_pandas import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

path = '../../data/train_data/'
out_path = '../../output/xgb/'

usecols = ['label', 'userID_day_count', 'appCategory_day_count', 'creativeID_positionID_day_count',
           'adID_userID_day_count', 'exptv_combo_creativeID_positionType', 'exptv_combo_adID_residence',
           'exptv_combo_adID_connectionType', 'exptv_combo_camgaignID_gender',
           'exptv_combo_advertiserID_appPlatform', 'exptv_combo_advertiserID_marriageStatus',
           'exptv_combo_advertiserID_positionID', 'exptv_combo_appID_userID', 'exptv_combo_appID_age',
           'exptv_combo_appID_education', 'exptv_combo_appID_haveBaby', 'exptv_combo_appCategory_userID',
           'exptv_combo_appCategory_gender', 'exptv_combo_appCategory_connectionType',
           'exptv_combo_userID_positionID', 'exptv_combo_userID_sitesetID', 'exptv_combo_userID_positionType',
           'exptv_combo_userID_connectionType', 'exptv_combo_age_residence',
           'exptv_combo_gender_marriageStatus', 'exptv_combo_gender_haveBaby',
           'exptv_combo_gender_sitesetID', 'exptv_combo_education_residence',
           'exptv_combo_education_connectionType', 'exptv_combo_education_clickTime_hour',
           'exptv_combo_marriageStatus_positionID', 'exptv_combo_marriageStatus_clickTime_hour',
           'exptv_combo_haveBaby_hometown', 'exptv_combo_positionID_connectionType']

fi = open('../../output/xgb/feat_importance.csv', 'r')
next(fi)
for t, line in enumerate(fi):
    feat = line.split(',')[0]
    usecols.append(feat)
    if t == 70:
        break
fi.close()

train_data = pd.read_csv(path + 'train.csv')

drop_list = ['sitesetID-cnv', 'hometown_province-cnv']
for x in train_data.columns:
    if x not in usecols:
        drop_list.append(x)
# drop_list = []

train_data = train_data.drop(drop_list, axis=1)
# train_data.to_csv('../../data/train1.csv', index=False)


label_train = train_data['label']
train_data.drop(['label'], axis=1, inplace=True)

sign = pd.read_csv('../../data/chuan.csv')
sign.columns = ['ss']

train_data = pd.concat([train_data, sign], axis=1)

# pos = pd.read_csv('../../data/strain.csv', usecols=['num-1', 'num-2', 'num-3', 'num-5'])
# train_data = pd.concat([train_data, pos], axis=1)
# print(len(train_data), len(pos))
# del pos
# gc.collect()

# # 用29天验证
# val_X = train_data.tail(valid_nrows)
# val_y = label_train.tail(valid_nrows)

# X_train = train_data.head(len(train_data) - valid_nrows)
# y_train = label_train.head(len(train_data) - valid_nrows)
# print(len(val_X), len(X_train), len(val_X)+len(X_train))
X_train, val_X, y_train, val_y = train_test_split(train_data, label_train, test_size=0.3, random_state=2)
del train_data, label_train
gc.collect()

xgb_val = xgb.DMatrix(val_X, label=val_y)
xgb_train = xgb.DMatrix(X_train, label=y_train)
evallist = [(xgb_train, 'train'), (xgb_val, 'eval')]

del X_train, y_train, val_X, val_y
gc.collect()

params = {'booster': 'gbtree', 'learning_rate': 0.1, 'n_estimators': 500, 'bst:max_depth': 4,
          'bst:min_child_weight': 1, 'bst:eta': 0.05, 'silent': 1, 'objective': 'binary:logistic',
          'gamma': 0.1, 'subsample': 0.8, 'scale_pos_weight': 1, 'colsample_bytree': 0.8,
          'eval_metric': 'logloss', 'nthread': 4, 'sample_type': 'uniform',
          'normalize_type': 'forest', 'tree_method': 'approx'}

num_round = 1000

bst = xgb.train(params, xgb_train, num_round, evals=evallist, early_stopping_rounds=5)
del xgb_train, xgb_val
gc.collect()

#
# bst = xgb.cv(params=params, dtrain=xgb_train, nfold=5, metrics='logloss', verbose_eval=2, early_stopping_rounds=5)

bst.save_model(out_path + 'xgb.model')

print(bst.get_fscore())

test_data = pd.read_csv(path + 'test1.csv')
# test_data.to_csv('../../data/test1.csv', index=False)
test_data = test_data.drop(['label'], axis=1)
test_data = test_data.drop(drop_list, axis=1)
xgb_test = xgb.DMatrix(test_data)

del test_data
gc.collect()

y_pred = bst.predict(xgb_test)

output = open(out_path + 'submission.csv', 'w')
output.write('instanceID,prob\n')
for t, p in enumerate(y_pred, start=1):
    output.write('{0},{1}\n'.format(t, p))
output.close()
