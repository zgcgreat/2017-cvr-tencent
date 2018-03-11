import json
from csv import DictReader

import lightgbm as lgb
import scipy as sp
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn_pandas import GridSearchCV

data_path = '../../data/'
path = '../../output/lgb/'

# y_true = []
# fi = open('{0}/validation.csv'.format(path), 'r')
# for line in DictReader(fi):
#     y_true.append(line['label'])
# fi.close()


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


print('Load data...')
train = '{0}/train.svm'.format(data_path)
test = '{0}/test.svm'.format(data_path)
lgb_train = lgb.Dataset(train)

lgb_eval = lgb_train.create_valid('eval.svm')
lgb_test = lgb.Dataset(test)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'binary_logloss', 'auc'},
    'metric_freq': 1,
    'is_training_metric': 'true',
    'max_bin': 255,
    'num_leaves': 31,
    'learning_rate': 0.2,
    'tree_learner': 'serial',
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 1e-3,
    'max_depth': 20,
    'thread': 4

}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=150,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Save model...')
# save model to file
gbm.save_model('{0}/model.txt'.format(path))

print('Start predicting...')
# predict
y_pred = gbm.predict(lgb_test, num_iteration=gbm.best_iteration)
pd.DataFrame(y_pred).to_csv('{0}/pred_svm.txt'.format(path), index=False)
# eval
# print('The auc of prediction is:', roc_auc_score(y_true, y_pred))
# print('The logloss of prediction is:', logloss(y_true, y_pred))


print('Feature names:', gbm.feature_name())

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))

# other scikit-learn modules
# estimator = lgb.LGBMRegressor(num_leaves=31)
#
# param_grid = {
#     'learning_rate': [0.01, 0.1, 1],
#     'n_estimators': [50, 100, 150]
# }
#
# gbm = GridSearchCV(estimator, param_grid)
#
# gbm.fit(train)
#
# print('Best parameters found by grid search are:', gbm.best_params_)
