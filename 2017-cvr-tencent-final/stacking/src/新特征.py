import pandas as pd

path = '../output/results/'

'''构造新的训练集特征'''
def tr_feat(name):
    va1 = pd.read_csv(path + '{0}/subval1.csv'.format(name))
    va2 = pd.read_csv(path + '{0}/subval2.csv'.format(name))
    va3 = pd.read_csv(path + '{0}/subval3.csv'.format(name))
    va4 = pd.read_csv(path + '{0}/subval4.csv'.format(name))
    va5 = pd.read_csv(path + '{0}/subval5.csv'.format(name))

    df = pd.concat([va1, va2, va3, va4, va5], axis=0)
    return df


def tr_concat():
    lgb_df = tr_feat('lgb')
    xgb_df = tr_feat('xgb')
    lgb_df = lgb_df.drop('label', axis=1)

    df = pd.concat([xgb_df,lgb_df], axis=1)
    return df


def te_feat(name):
    sub1 = pd.read_csv(path + '{0}/sub1.csv'.format(name), usecols=['{0}_prob'.format(name)])
    sub2 = pd.read_csv(path + '{0}/sub2.csv'.format(name), usecols=['{0}_prob'.format(name)])
    sub3 = pd.read_csv(path + '{0}/sub3.csv'.format(name), usecols=['{0}_prob'.format(name)])
    sub4 = pd.read_csv(path + '{0}/sub4.csv'.format(name), usecols=['{0}_prob'.format(name)])
    sub5 = pd.read_csv(path + '{0}/sub5.csv'.format(name), usecols=['{0}_prob'.format(name)])
    df = (sub1['{0}_prob'.format(name)]+sub2['{0}_prob'.format(name)]+sub3['{0}_prob'.format(name)]
         + sub4['{0}_prob'.format(name)] + sub5['{0}_prob'.format(name)]) / 5
    return df


def te_concat():
    xgb_df = te_feat('xgb')
    lgb_df = te_feat('lgb')
    df = pd.concat([xgb_df, lgb_df], axis=1)

    return df


if __name__ == '__main__':
    tr_df = tr_concat()
    tr_df.to_csv('../output/train.csv', index=False)
    te_df = te_concat()
    te_df.to_csv('../output/test.csv', index=False)
    print(len(te_df))


