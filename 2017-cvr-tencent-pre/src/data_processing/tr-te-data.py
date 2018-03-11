# -*- encoding:utf-8 -*-

from csv import DictReader
import pandas as pd
import gc

input_path = '../../data_ori'
output_path = '../../data'

tr = '{0}/train.csv'.format(output_path)
te = '{0}/test.csv'.format(output_path)

train = '{0}/train.csv'.format(input_path)
test = '{0}/test.csv'.format(input_path)
user = '{0}/user.csv'.format(input_path)
user_app_action = '{0}/user_app_actions.csv'.format(input_path)
user_installedapp = '{0}/user_installedapps.csv'.format(input_path)
app_categorie = '{0}/app_categories.csv'.format(input_path)
position = '{0}/position.csv'.format(input_path)
ad = '{0}/ad.csv'.format(input_path)


def hour(series):
    return str(series)[2:4]


def minute(series):
    return str(series)[4:]


def time_transform(series):
    return str(series)[:4]


def convert_age(age):
    return int(int(age) / 5)


def date(clickTime):
    return str(clickTime)[:2]


def ad_cam(advertiserID, camgaignID):
    return advertiserID + camgaignID


def make_data(data, user, positions, ads, app_categories):
    # user['hometown_city'] = user['hometown'] % 100
    # user['hometown_province'] = (user['hometown'] / 100).astype('int')
    # user['residence_city'] = user['residence'] % 100
    # user['residence_province'] = (user['residence'] / 100).astype('int')
    data = pd.merge(data, user, how='left', on='userID')
    # data['date'] = data['clickTime'].apply(date)
    # data['hour'] = data['clickTime'].apply(hour)
    # data['minute'] = data['clickTime'].apply(minute)

    data = pd.merge(data, positions, how='left', on='positionID')
    data = pd.merge(data, ads, how='left', on='creativeID')
    data = pd.merge(data, app_categories, how='left', on='appID')
    # data = data.fillna(0)
    # data['clickTime'] = data['clickTime'].apply(time_transform)
    # data['age'] = data['age'].replace(0., data['age'].mean())  # 将未知年龄均化
    data['age'] = data['age'].apply(convert_age)

    del data['userID']
    gc.collect()
    return data


if __name__ == '__main__':
    users = pd.read_csv(user)
    positions = pd.read_csv(position)
    ads = pd.read_csv(ad)
    app_categories = pd.read_csv(app_categorie)

    train = pd.read_csv(train)
    # train = train[train['clickTime'] < 310000]
    train = train.drop_duplicates()
    train = make_data(train, users, positions, ads, app_categories)

    # train.insert(1, 'weekday', 0)
    # train['weekday'] = train['clickTime'].apply(week_day)

    del train['conversionTime']
    # train = train[train['clickTime'] < 300000]
    train.to_csv(tr, index=False)
    print(len(train))
    del train
    gc.collect()
    print('train data completed !!!')

    test = pd.read_csv(test, dtype={'clickTime': object})
    test = make_data(test, users, positions, ads, app_categories)
    test.drop(['instanceID'], axis=1, inplace=True)
    # test.insert(1, 'weekday', 0)
    # test['weekday'] = test['clickTime'].apply(week_day)
    test.to_csv(te, index=False)
    print(len(test))
    del test
    gc.collect()
    print('test data completed !!!')
