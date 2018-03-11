# _*_ coding: utf-8 _*_

import zipfile
import pandas as pd
import xgboost as xgb
from sklearn_pandas import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

path = '../../data/'
out_path = '../../output/xgb/'
field = ['clickTime', 'creativeID', 'userID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory']

train_data = pd.read_csv(path + 'train-ctr.csv')
# train_data = train_data.sample(frac=0.8, random_state=0)
# drop_list = ['clickTime_age', 'creativeID_advertiserID', 'camgaignID_appCategory', 'advertiserID_residence'
#              , 'residence_age', 'age_appCategory', 'appID.1']
#
drop_list = []
train_data = train_data.drop(drop_list, axis=1)


test_data = pd.read_csv(path + 'test-ctr.csv')
test_data = test_data.drop(['label'], axis=1)
test_data = test_data.drop(drop_list, axis=1)

label_train = train_data['label']
train_data.drop(['label'], axis=1, inplace=True)

val_X = train_data[train_data.shape[0]-301769:, :]
val_y = label_train[train_data.shape[0]-301769:, :]
print(val_X.shape)
train_data = train_data[:train_data.shape[0]-301769, :]
label_train = label_train[:train_data.shape[0]-301769, :]

# X_train, val_X, y_train, val_y = train_test_split(train_data, label_train, test_size=0.1, random_state=1)


xgb_val = xgb.DMatrix(val_X, label=val_y)
xgb_train = xgb.DMatrix(train_data, label=label_train)

# xgb_train = xgb.DMatrix(train_data, label)
xgb_test = xgb.DMatrix(test_data)

params = {'booster': 'gbtree',
          'learning_rate': 0.05,
          'n_estimators': 100,
          'bst:max_depth': 5,
          'bst:min_child_weight': 1,
          'bst:eta': 0.05,
          'silent': 1,
          'objective': 'reg:logistic',
          'gamma': 0.1, 'subsample': 0.8,
          'scale_pos_weight': 1,
          'colsample_bytree': 0.8,
          'eval_metric': 'logloss',
          'nthread': 8,
          'sample_type': 'uniform',
          'normalize_type': 'forest',
          'random_state': 1}

plst = params.items()
evallist = [(xgb_train, 'train'), (xgb_val, 'eval')]
num_round = 500

bst = xgb.train(plst, xgb_train, num_round, evals=evallist, early_stopping_rounds=10)

#
# bst = xgb.cv(params=params, dtrain=xgb_train, nfold=5, metrics='logloss', verbose_eval=2, early_stopping_rounds=10)

# bst.save_model(out_path + 'xgb.model')

feat_importance = bst.get_fscore()
print(feat_importance)

with open(out_path + 'feat_importance.csv', 'w') as fo:
    for k in feat_importance.keys():
        fo.write(str(k) + ',')
    fo.write('\n')
    for k in feat_importance.keys():
        fo.write(str(feat_importance[k]) + ',')

y_pred = bst.predict(xgb_test)

output = open(out_path + 'submission.csv', 'w')
output.write('instanceID,prob\n')
for t, p in enumerate(y_pred, start=1):
    output.write('{0},{1}\n'.format(t, p))
output.close()

with zipfile.ZipFile(out_path + "submission.zip", "w") as fout:
    fout.write(out_path + "submission.csv", compress_type=zipfile.ZIP_DEFLATED)

# xgb.plot_importance(bst)


