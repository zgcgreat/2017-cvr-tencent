import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy as sp

path = '../../data/cvr_data/'


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll

drop_list = ['userID', 'date', 'minute']
print('Loading data...')
train = pd.read_csv(path+'tr_cvr.csv')
test = pd.read_csv(path+'te_cvr.csv')

label_train = train['label']
label_test = test['label']

train = train.drop('label', axis=1)
test = test.drop('label', axis=1)

# train = train.drop(drop_list, axis=1)
# test = test.drop(drop_list, axis=1)

# print('One-Hot Encoding...')
# oneEnc = OneHotEncoder()
# data = pd.concat([train, test])
# data = oneEnc.fit_transform(data)
# train = data[:train.shape[0]]
# test = data[train.shape[0]:]

print('train test splitting...')
X_train, val_X, y_train, val_y = train_test_split(train, label_train, test_size=0.1, random_state=2)

print('training...')
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(val_X)

ll = logloss(val_y, pred)
print(ll)
