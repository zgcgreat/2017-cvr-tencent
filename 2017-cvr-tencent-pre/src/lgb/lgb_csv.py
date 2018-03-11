# coding: utf-8

import json

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
# load or create your dataset
print('Load data...')


# drop_list = ['clickTime_age', 'creativeID_advertiserID', 'camgaignID_appCategory', 'advertiserID_residence'
#              ,'residence_age', 'age_appCategory', 'appID.1']

drop_list = []
train_data = pd.read_csv(path + 'train-ctr.csv')
train_data = train_data.drop_duplicates()
train_data = train_data.sample(frac=0.1, random_state=1)

train_data = train_data.drop(drop_list, axis=1)

test_data = pd.read_csv(path + 'test-ctr.csv')

test_data = test_data.drop('label', axis=1)
test_data = test_data.drop(drop_list, axis=1)

label_train = train_data['label']
train_data.drop(['label'], axis=1, inplace=True)
#
# sel = VarianceThreshold(threshold=0.0001)
# train_data = sel.fit_transform(train_data)

# train_data = SelectKBest(chi2, k=60).fit_transform(train_data, label_train)

X_train, val_X, y_train, val_y = train_test_split(train_data, label_train, test_size=0.1, random_state=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'binary_logloss'},
    'metric_freq': 1,
    'is_training_metric': 'true',
    'max_bin': 255,
    'num_leaves': 100,
    'learning_rate': 0.1,
    'tree_learner': 'serial',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 200,
    'max_depth': -1,
    'random_state': 1
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=[lgb_train, lgb_eval],
                early_stopping_rounds=5)


def update_pred(pred):
    pred = pred / (pred + (1 - pred) / 0.75)
    return pred


print('Start predicting...')
# predict
y_pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)


fo = open('../../output/lgb/submission.csv', 'w')
fo.write('instanceID,prob\n')
for t, prob in enumerate(y_pred, start=1):
    fo.write(str(t) + ',' + str(prob) + '\n')
fo.close()


print('Feature names:', gbm.feature_name())

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))

fo = open('../../output/lgb/feature_importance.csv', 'w')
for x in gbm.feature_name():
    fo.write(x + ',')
fo.write('\n')
for x in list(gbm.feature_importance()):
    fo.write(str(x) + ',')

fo.close()
