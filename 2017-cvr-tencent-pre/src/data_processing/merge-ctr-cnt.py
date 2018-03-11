# _*_ coding: utf-8 _*_
import pandas as pd
from numpy import *
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler

path = '../../data/'


train_ori = pd.read_csv('{0}/train-ctr.csv'.format(path))
train_cnt = pd.read_csv('{0}/train-cnt.csv'.format(path))
train_cnt.drop('label', axis=1, inplace=True)
train_cnt = pd.DataFrame(MinMaxScaler().fit_transform(train_cnt))
print(train_cnt.head())

train = pd.concat([train_ori, train_cnt], axis=1)


del train_ori
del train_cnt
train.to_csv('{0}/tr-ctr-cnt.csv'.format(path), index=False)
del train
print('train data merged')

test_ori = pd.read_csv('{0}/test-ctr.csv'.format(path))
test_cnt = pd.read_csv('{0}/test-cnt.csv'.format(path))
test_cnt.drop('label', axis=1, inplace=True)
test_cnt = pd.DataFrame(MinMaxScaler().fit_transform(test_cnt))


test = pd.concat([test_ori, test_cnt], axis=1)

del test_ori
del test_cnt

test.to_csv('{0}/te-ctr-cnt.csv'.format(path), index=False)
del test
print('test data merged')
