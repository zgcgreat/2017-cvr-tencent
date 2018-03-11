import numpy as np
import pandas as pd
import xgboost as xgb
import operator
import matplotlib.pyplot as plt


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


if __name__ == '__main__':
    train = pd.read_csv("../../data/train-ctr.csv")
    train = train.sample(frac=0.1, random_state=1)
    # cat_sel = [n for n in train.columns if n.startswith('cat')]  # 类别特征数值化
    # for column in cat_sel:
    #     train[column] = pd.factorize(train[column].values, sort=True)[0] + 1

    params = {'booster': 'gbtree', 'learning_rate': 0.05, 'n_estimators': 50, 'bst:max_depth': 5,
              'bst:min_child_weight': 1, 'bst:eta': 0.05,
              'silent': 1, 'objective': 'reg:logistic', 'gamma': 0.1, 'subsample': 0.8, 'scale_pos_weight': 1,
              'colsample_bytree': 0.8, 'eval_metric': 'logloss', 'nthread': 8, 'sample_type': 'uniform',
              'normalize_type': 'forest'}
    rounds = 150
    y = train['label']
    X = train.drop(['label'], axis=1)

    xgtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(params, xgtrain, evals=xgtrain, num_boost_round=rounds, early_stopping_rounds=10)

    features = [x for x in train.columns if x not in ['label']]
    ceate_feature_map(features)

    importance = bst.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.to_csv("../../output/xgb/feat_importance.csv", index=False)

    plt.figure()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.savefig('feats_importance.png', dpi=100)
    plt.show()

