# _*_ coding: utf-8 _*_
import pandas as pd
from numpy import *
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler

path = '../../data/'


def time_transform(series):
    return str(series)[2:4]

train_ori = pd.read_csv('{0}/tr_new.csv'.format(path))
train_statical = pd.read_csv('{0}/train-ctr.csv'.format(path))

train_ori['clickTime'] = train_ori['clickTime'].apply(time_transform)


# user_installedapps = pd.read_csv('{0}/user_installedapps.csv'.format(path))
# user_installedapps['installedapps'] = user_installedapps.fillna(
#     mean(user_installedapps['installedapps']))
# mapper = DataFrameMapper([(['installedapps'], [MinMaxScaler()])])
# user_installedapps['installedapps'] = mapper.fit_transform(user_installedapps)
# print(user_installedapps[:5])

train_statical.drop('label', axis=1, inplace=True)
train = pd.concat([train_ori, train_statical], axis=1)


del train_ori
del train_statical
train.to_csv('{0}/train_merged.csv'.format(path), index=False)
del train
print('train data merged')

test_ori = pd.read_csv('{0}/test.csv'.format(path))
test_statical = pd.read_csv('{0}/test-ctr.csv'.format(path))

test_ori['clickTime'] = test_ori['clickTime'].apply(time_transform)

test_statical.drop('label', axis=1, inplace=True)
test = pd.concat([test_ori, test_statical], axis=1)
# test = pd.merge(test, user_installedapps, how='left', on='userID')
del test_ori
del test_statical
# del user_installedapps
test.to_csv('{0}/test_merged.csv'.format(path), index=False)
del test
print('test data merged')
