# _*_ coding: utf-8 _*_

import gc
import sys
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

num = sys.argv[1]
# num = 1

path = '../../output/stack-data1/'

out_path = '../../output/results/xgb/'


usecols = ['label']
fi = open('../../../output/xgb/feat_importance.csv', 'r')
next(fi)
for t, line in enumerate(fi):
    feat = line.split(',')[0]
    usecols.append(feat)
    if t == 80:
        break
fi.close()
train_data = pd.read_csv(path + 'train{0}.csv'.format(num))

drop_list = []
for x in train_data.columns:
    if x not in usecols:
        drop_list.append(x)
drop_list = []

train_data = train_data.drop(drop_list, axis=1)

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
params = {'booster': 'gbtree', 'learning_rate': 0.05, 'n_estimators': 500, 'bst:max_depth': 4,
          'bst:min_child_weight': 1, 'bst:eta': 0.05, 'silent': 1, 'objective': 'binary:logistic',
          'gamma': 0.1, 'subsample': 0.8, 'scale_pos_weight': 1, 'colsample_bytree': 0.8,
          'eval_metric': 'logloss', 'nthread': 4, 'sample_type': 'uniform',
          'normalize_type': 'forest', 'tree_method': 'approx'}

num_round = 505

bst = xgb.train(params, xgb_train, num_round, evals=evallist, early_stopping_rounds=5)
del xgb_train, xgb_val
gc.collect()


bst.save_model(out_path + 'xgb{0}.model'.format(num))

print(bst.get_fscore())

valid = pd.read_csv(path+'valid{0}.csv'.format(num))
valid = valid.drop(drop_list, axis=1)
label_valid = np.array(valid['label'])
valid.drop(['label'], axis=1, inplace=True)

xgb_val = xgb.DMatrix(valid)

del valid
gc.collect()

val_pred = bst.predict(xgb_val)
output = open(out_path + 'subval{0}.csv'.format(num), 'w')
output.write('label,xgb_prob\n')
for t, p in enumerate(val_pred, start=1):
    output.write('{0},{1}\n'.format(label_valid[t-1], p))
output.close()


test_data = pd.read_csv('../../../data/test1.csv')

test_data = test_data.drop(['label'], axis=1)
test_data = test_data.drop(drop_list, axis=1)
xgb_test = xgb.DMatrix(test_data)

del test_data
gc.collect()

y_pred = bst.predict(xgb_test)

output = open(out_path + 'sub{0}.csv'.format(num), 'w')
output.write('instanceID,xgb_prob\n')
for t, p in enumerate(y_pred, start=1):
    output.write('{0},{1}\n'.format(t, p))
output.close()

