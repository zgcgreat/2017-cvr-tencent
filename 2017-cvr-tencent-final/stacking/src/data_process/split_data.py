import pandas as pd
import gc

path = '../../../../output/ffm/'


def split_data(i):
    valid = df.iloc[(i - 1) * int(len(df) / 5):i * int(len(df) / 5), :]
    train = df.ix[df.index.difference(valid.index)]
    valid.to_csv('../../output/stack-data/ffm/valid{0}.ffm'.format(i), index=False)
    train.to_csv('../../output/stack-data/ffm/train{0}.ffm'.format(i), index=False)

    print(len(train), len(valid), len(train) + len(valid))
    print((i - 1) * int(len(df) / 5), i * int(len(df) / 5))

    del train, valid
    gc.collect()

    
if __name__ == '__main__':
    df = pd.read_csv(path + 'train.ffm')
    print(len(df))

    for i in range(1, 2):
        split_data(i)
