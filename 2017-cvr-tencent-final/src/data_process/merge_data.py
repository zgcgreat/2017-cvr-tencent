import pandas as pd
import gc

# 合并cvr和分布概率
# tr_cvr = pd.read_csv('../../data/strain.csv')
# trp = pd.read_csv('../../data/train-p.csv')
# trp = trp.drop('label', axis=1)
# df = pd.concat([tr_cvr, trp], axis=1)
# df.to_csv('../../data/traincp.csv', index=False)
#
# tr_cvr = pd.read_csv('../../data/stest.csv')
# trp = pd.read_csv('../../data/test-p.csv')
# trp = trp.drop('label', axis=1)
# df = pd.concat([tr_cvr, trp], axis=1)
# df.to_csv('../../data/testcp.csv', index=False)

# 加入重复数据标记特征
# tr_cvr = pd.read_csv('../../output/feature_data/tr_clk_cnv.csv')
# trp = pd.read_csv('../../output/feature_data/train_sign.csv', usecols=['appear_times', 'sign'])
#
# df = pd.concat([tr_cvr, trp], axis=1)
# print(df.head())
# df.to_csv('../../data/train_data/train.csv', index=False)
#
# tr_cvr = pd.read_csv('../../output/feature_data/te_clk_cnv.csv')
# trp = pd.read_csv('../../output/feature_data/test_sign.csv', usecols=['appear_times', 'sign'])
# # trp = trp.drop(['label', 'date'], axis=1)
# df = pd.concat([tr_cvr, trp], axis=1)
# df.to_csv('../../data/train_data/test.csv', index=False)


# 加入平滑后的positionID特征
tr = pd.read_csv('../../data/train.csv', usecols=['positionID', 'date'])
tr = tr[tr['date'] >= 28]
tr = tr[tr['date'] <= 29]
cov_pos = pd.read_csv('../../output/feature_data/cov-pos.csv', usecols=['positionID','positionID_Tr','positionID_Sum','0'])
tr = pd.merge(tr, cov_pos, how='left', on='positionID')
tr = tr.drop(['positionID', 'date'], axis=1)
print(tr.head())

train = pd.read_csv('../../data/str.csv')
print(len(train), len(tr))
train = pd.concat([train, tr], axis=1)
train.to_csv('../../data/train_data/train1.csv', index=False)
# #
tr = pd.read_csv('../../data/test.csv', usecols=['positionID'])
tr = pd.merge(tr, cov_pos, how='left', on='positionID')
tr = tr.drop(['positionID'], axis=1)
train = pd.read_csv('../../data/ste.csv')
train = pd.concat([train, tr], axis=1)
train.to_csv('../../data/train_data/test1.csv', index=False)
#
#
# # 加入时间间隔特征
usecols = ['user_time_interval', 'userID_creativeID_time_interval', 'userID_appID_time_interval',
           'creativeID_appID_time_interval', 'clk-cnv', 'appear_times', 'sign']
tr_df = pd.read_csv('../../output/tr_uccs.csv', usecols=usecols)
tr = pd.read_csv('../../data/train_data/train1.csv')
print(len(tr_df), len(tr))
tr = pd.concat([tr, tr_df], axis=1)
tr.to_csv('../../data/train_data/train2.csv', index=False)

tr_df = pd.read_csv('../../output/te_uccs.csv', usecols=usecols)
tr = pd.read_csv('../../data/train_data/test1.csv')
tr = pd.concat([tr, tr_df], axis=1)
tr.to_csv('../../data/train_data/test2.csv', index=False)


# # 加入user当天点击量的特征
df1 = pd.read_csv('../../data/daily_feature/day_count_feature_28.csv')
df2 = pd.read_csv('../../data/daily_feature/day_count_feature_29.csv')
df = pd.concat([df1, df2], axis=0).reset_index()
df = df.drop('index', axis=1)
print(df.tail())
del df1, df2
gc.collect()
tr = pd.read_csv('../../data/train_data/train2.csv')
print(tr.tail())
tr = pd.concat([tr, df], axis=1)
tr.to_csv('../../data/train_data/train3.csv', index=False)

print(len(df))

df = pd.read_csv('../../data/daily_feature/day_count_feature_31.csv')
tr = pd.read_csv('../../data/train_data/test2.csv')
tr = pd.concat([tr, df], axis=1)
del df
gc.collect()
tr.to_csv('../../data/train_data/test3.csv', index=False)

# 加入组合转化率特征
df1 = pd.read_csv('../../data/combo_exp_feature/filter_combo_exp7_nosmooth_feature_28.csv')
df2 = pd.read_csv('../../data/combo_exp_feature/filter_combo_exp7_nosmooth_feature_29.csv')
df1 = df1.drop('label', axis=1)
df2 = df2.drop('label', axis=1)
df = pd.concat([df1, df2], axis=0).reset_index()
df = df.drop('index', axis=1)
print(df.tail())
del df1, df2
gc.collect()
tr = pd.read_csv('../../data/train_data/train3.csv')
print(tr.tail())
tr = pd.concat([tr, df], axis=1)
tr.to_csv('../../data/train_data/train1.csv', index=False)

print(len(df))

df = pd.read_csv('../../data/combo_exp_feature/filter_combo_exp7_nosmooth_feature_31.csv')
df = df.drop('label', axis=1)
tr = pd.read_csv('../../data/train_data/test3.csv')
tr = pd.concat([tr, df], axis=1)
del df
gc.collect()
tr.to_csv('../../data/train_data/test1.csv', index=False)
