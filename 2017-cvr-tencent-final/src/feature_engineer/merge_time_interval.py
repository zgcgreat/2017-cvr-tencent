import pandas as pd
import gc

path = '../../output/feature_data/'

tr_user_time_interval = pd.read_csv(path + 'tr_time_interval.csv')
tr_clk_cnv = pd.read_csv(path + 'tr_clk-cnv.csv', usecols=['clk-cnv'])
tr_cnv_clk = pd.read_csv(path + 'tr_cnv-clk.csv', usecols=['cnv-clk'])
tr_sign = pd.read_csv(path + 'train_sign.csv', usecols=['appear_times', 'sign'])
print(len(tr_user_time_interval), len(tr_clk_cnv), len(tr_cnv_clk), len(tr_sign))
train = pd.concat([tr_user_time_interval, tr_clk_cnv, tr_cnv_clk, tr_sign], axis=1)
print(train.head())

train.to_csv('../../output/tr_uccs.csv', index=False)
del train
gc.collect()

te_user_time_interval = pd.read_csv(path + 'te_time_interval.csv')
te_clk_cnv = pd.read_csv(path + 'te_clk-cnv.csv', usecols=['clk-cnv'])
te_cnv_clk = pd.read_csv(path + 'te_cnv-clk.csv', usecols=['cnv-clk'])
te_sign = pd.read_csv(path + 'test_sign.csv', usecols=['appear_times', 'sign'])

test = pd.concat([te_user_time_interval, te_clk_cnv, te_cnv_clk, te_sign], axis=1)
print(len(test))

test.to_csv('../../output/te_uccs.csv', index=False)
del test
gc.collect()
