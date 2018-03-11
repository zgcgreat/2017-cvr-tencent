import pandas as pd
import numpy as np

train = pd.read_csv('../../data/train-ctr.csv')
train = train.sample(frac=0.8, random_state=1)
# df = train.groupby(['label', 'clickTime-ctr'], as_index=False)
# print(df)
print(np.var(train))
del train

# test = pd.read_csv('../../data/test-ctr.csv')
#
# print(np.var(test))
