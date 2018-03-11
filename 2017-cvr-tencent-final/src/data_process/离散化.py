import pandas as pd

path = '../../data/train_data/'


def bin(filename):
    df = pd.read_csv(path+'{0}.csv'.format(filename))
    for feat in df.columns:
        if feat not in ['label', 'appear_times', 'sign']:
            df[feat] = pd.cut(df[feat], bins=20, labels=False)
    return df

if __name__ == '__main__':
    print('train')
    df = bin('train1')
    df.to_csv('../../output/ffm/train.csv', index=False)
    print('test')
    df = bin('test1')
    df.to_csv('../../output/ffm/test.csv', index=False)