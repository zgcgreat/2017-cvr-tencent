# _*_ coding: utf-8 _*_

import sys
import subprocess
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

#
# if len(sys.argv) != 3:
#     print('wrong arg')
#     exit(1)
#
# data_path = sys.argv[1]
# result_path = sys.argv[2]

path = '../../data/'

train_data = pd.read_csv(path + 'train-ctr.csv')
test_data = pd.read_csv(path + 'test-ctr.csv')

drop_list = ['marriageStatus-ctr', 'appPlatform-ctr', 'haveBaby-ctr']

train_data = train_data.drop(drop_list, axis=1)
test_data = test_data.drop(drop_list, axis=1)

label_train = train_data['label']
train_data = train_data.drop('label', axis=1)

test_data = test_data.drop('label', axis=1)


X_train, val_X, y_train, val_y = train_test_split(train_data, label_train, test_size=0.1, random_state=1)

lgb_train = xgb.DMatrix(X_train, y_train)
lgb_eval = xgb.DMatrix(val_X, val_y)
lgb_test = xgb.DMatrix(test_data)
# params = {'silent': 1, 'objective': 'binary:logistic', 'booster': 'gblinear', 'lambda': 25,
#          'nthread': 8, 'eval_metric': 'logloss'}

params = {'objective': 'binary:logistic',
          'learning_rate': 0.2,
          'eta': 0.2,
          'max_depth': 10,
          'eval_metric': 'logloss',
          'silent': 1,
          'nthread': 8,
          'gamma': 0.8,
          'min_child_weight': 4,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.8,
          'booster': 'gblinear',
          'lambda': 25}

feval = [(lgb_eval, 'eval'), (lgb_train, 'train')]

epoch = 1
num_round = 2500


params['seed'] = 3015 + 10
model = xgb.train(
        params=params,
        dtrain=lgb_train,
        num_boost_round=num_round,
        evals=feval,
        early_stopping_rounds=5
    )
y_pred = model.predict(lgb_test)

# model.dump_model(path + 'dump.raw.txt')
output = open(path + 'submission.csv', 'w')
output.write('instanceID,prob\n')
for t, p in enumerate(y_pred, start=1):
    output.write('{0},{1}\n'.format(t, p))

output.close()


# cmd = 'rm {path}train.svm {path}test.svm'.format(path=path)
# subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
