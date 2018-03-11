import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('../../data/train_merged.csv')

Y = train['label']
X = train.drop('label', axis=1)
names = train.columns

X = np.array(X)
Y = np.array(Y)
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
    print(i)
    score = cross_val_score(rf, X[:, i:i + 1], Y, scoring="r2",
                            cv=ShuffleSplit(len(X), 3, .3))
    scores.append((round(np.mean(score), 3), names[i]))
print(sorted(scores, reverse=True))
