# _*_ coding: utf-8 _*_

import gc
import sys
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

# num = sys.argv[1]
num = 1

path = '../../output/stack-data/'

out_path = '../../output/results/xgb/'

bst = xgb.Booster(model_file='xgb{0}.model'.format(num))  # load model


# valid = pd.read_csv(path + 'valid{0}.csv'.format(num))
# label_valid = np.array(valid['label'])
# valid.drop(['label'], axis=1, inplace=True)
#
# xgb_val = xgb.DMatrix(valid)
#
# del valid
# gc.collect()
#
# val_pred = bst.predict(xgb_val)
# output = open(out_path + 'subval{0}.csv'.format(num), 'w')
# output.write('label,xgb_prob\n')
# for t, p in enumerate(val_pred, start=1):
#     output.write('{0},{1}\n'.format(label_valid[t - 1], p))
# output.close()

test_data = pd.read_csv('../../../data/train_data/test1.csv')

test_data = test_data.drop(['label'], axis=1)
xgb_test = xgb.DMatrix(test_data)

del test_data
gc.collect()

y_pred = bst.predict(xgb_test)

output = open(out_path + 'sub{0}.csv'.format(num), 'w')
output.write('instanceID,xgb_prob\n')
for t, p in enumerate(y_pred, start=1):
    output.write('{0},{1}\n'.format(t, p))
output.close()
