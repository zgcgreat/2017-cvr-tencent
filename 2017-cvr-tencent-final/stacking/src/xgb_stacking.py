# _*_ coding: utf-8 _*_

import gc
import sys
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split


path = '../output/'

train_data = pd.read_csv(path + 'train.csv')

label_train = train_data['label']
train_data.drop(['label'], axis=1, inplace=True)
print(len(train_data))

X_train, val_X, y_train, val_y = train_test_split(train_data, label_train, test_size=0.1, random_state=2)
del train_data, label_train
gc.collect()

xgb_val = xgb.DMatrix(val_X, label=val_y)
xgb_train = xgb.DMatrix(X_train, label=y_train)
evallist = [(xgb_train, 'train'), (xgb_val, 'eval')]

del X_train, y_train, val_X, val_y
gc.collect()
params = {'booster': 'gbtree', 'learning_rate': 0.05, 'n_estimators': 50, 'bst:max_depth': 6,
          'bst:min_child_weight': 1, 'bst:eta': 0.05, 'silent': 1, 'objective': 'binary:logistic',
          'gamma': 0.5, 'subsample': 0.8, 'scale_pos_weight': 1, 'colsample_bytree': 1,
          'eval_metric': 'logloss', 'nthread': 4, 'sample_type': 'uniform',
          'normalize_type': 'forest', 'tree_method': 'approx'}

num_round = 505

bst = xgb.train(params, xgb_train, num_round, evals=evallist, early_stopping_rounds=5)
del xgb_train, xgb_val
gc.collect()


# bst.save_model(path + 'xgb.model'.format(num))

print(bst.get_fscore())


test_data = pd.read_csv(path+'test.csv')
print(len(test_data))
xgb_test = xgb.DMatrix(test_data)

del test_data
gc.collect()

y_pred = bst.predict(xgb_test)

output = open(path + 'submission.csv', 'w')
output.write('instanceID,prob\n')
for t, p in enumerate(y_pred, start=1):
    output.write('{0},{1}\n'.format(t, p))
output.close()

