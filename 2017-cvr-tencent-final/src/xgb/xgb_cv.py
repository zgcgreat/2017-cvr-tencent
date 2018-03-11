# _*_ coding: utf-8 _*_

import zipfile
import pandas as pd
import xgboost as xgb
import gc
import numpy as np
from sklearn_pandas import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

path = '../../data/'
out_path = '../../output/xgb/'
field = ['clickTime', 'creativeID', 'userID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory']

usecols = ['label']
fi = open('../../output/xgb/feat_importance.csv', 'r')
next(fi)
for t, line in enumerate(fi):
    feat = line.split(',')[0]
    usecols.append(feat)
    if t == 80:
        break
fi.close()

tr_nrows = 9000000
valid_nrows = 3000000
train_data = pd.read_csv(path + 'traincp.csv')
# train_data = train_data.tail(tr_nrows)

drop_list = []
for x in train_data.columns:
    if x not in usecols:
        drop_list.append(x)
drop_list = []

train_data = train_data.drop(drop_list, axis=1)

label_train = train_data['label']
train_data.drop(['label'], axis=1, inplace=True)
print(len(train_data))

xgb_train = xgb.DMatrix(train_data, label=label_train)

del train_data, label_train
gc.collect()

params = {'booster': 'gbtree', 'learning_rate': 0.05, 'n_estimators': 500, 'bst:max_depth': 4,
          'bst:min_child_weight': 1, 'bst:eta': 0.05, 'silent': 1, 'objective': 'binary:logistic',
          'gamma': 0.1, 'subsample': 0.8, 'scale_pos_weight': 1, 'colsample_bytree': 0.8,
          'eval_metric': 'logloss', 'nthread': 4, 'sample_type': 'uniform',
          'normalize_type': 'forest', 'tree_method': 'approx'}

num_round = 2000

#
bst = xgb.cv(params=params, dtrain=xgb_train, nfold=10, metrics='logloss', verbose_eval=2, early_stopping_rounds=5)
del xgb_train
gc.collect()

bst.save_model(out_path + 'xgb.model')

print(bst.get_fscore())

test_data = pd.read_csv(path + 'testcp.csv')

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

