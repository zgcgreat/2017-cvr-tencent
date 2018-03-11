# _*_ coding: utf-8 _*_
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np

path = '../../data/'

dtype = {'label': object, 'clickTime': object, 'creativeID': object, 'positionID': object, 'connectionType': object,
         'telecomsOperator': object, 'age': object, 'gender': object, 'education': object, 'marriageStatus': object,
         'haveBaby': object, 'hometown': object, 'residence': object, 'sitesetID': object, 'positionType': object,
         'adID': object, 'camgaignID': object, 'advertiserID': object, 'appID': object, 'appPlatform': object,
         'appCategory': object}

train = pd.read_csv('{0}/train.csv'.format(path), dtype=dtype)
test = pd.read_csv('{0}/test.csv'.format(path), dtype=dtype)

# train = train.fillna(0)
# test = test.fillna(0)

label_train = train['label']
label_test = test['label']

train.drop(['label'], axis=1, inplace=True)
test.drop(['label'], axis=1, inplace=True)

# train_test = pd.concat([train, test], ignore_index=True)

# train_test = pd.get_dummies(train_test, drop_first=True)

oneEnc = OneHotEncoder()
data_one = pd.concat([train, test])
data_one = oneEnc.fit_transform(data_one)
train_one = data_one[:train.shape[0]]
test_one = data_one[train.shape[0]:]
# train = hstack([train_one, user_installedapps_tfv[:train.shape[0]]])
# test = hstack([test_one, user_installedapps_tfv[train.shape[0]:]])

# train = train_test.loc[:train.shape[0]-1, :]
# test = train_test.loc[train.shape[0]:, :]

# GBDT作为基模型的特征选择
# print('training...')
# clf = GradientBoostingClassifier()
# clf = clf.fit(train, label_train)
# model = SelectFromModel(clf, prefit=True)
# print('complete training...')
#
# train = model.transform(train)
# test = model.transform(test)

pd.DataFrame(train).to_csv('{0}/tr_ont-hot.csv'.format(path), index=False)
pd.DataFrame(test).to_csv('{0}/te_one-hot.csv'.format(path), index=False)

print(pd.DataFrame(train).head())
print(pd.DataFrame(test).head())
