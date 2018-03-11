import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
import scipy as sp
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


path = '../../data/'
train = pd.read_csv(path + 'train-ctr.csv')
test = pd.read_csv(path + 'test-ctr.csv')

y_train = train['label']
train.drop('label', axis=1, inplace=True)
y_test = test['label']
test.drop('label', axis=1, inplace=True)

# train = MinMaxScaler().fit_transform(train)
# test = MinMaxScaler().fit_transform(test)
# print(train)

X_train, val_X, y_train, val_y = train_test_split(train, y_train, test_size=0.1, random_state=1)
# y_train = train_xy.label
# X_train = train_xy.drop(['label'], axis=1)
# val_y = val.label
# val_X = val.drop(['label'], axis=1)

model = LassoCV()
model.fit(X_train, y_train)

pred = model.predict(val_X)

ll = logloss(val_y, pred)
print(ll)
