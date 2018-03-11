# _*_ coding: utf-8 _*_

import sys
import csv
from common import *

# csv_path = sys.argv[1]
# dense_path = sys.argv[2]
# sparse_path = sys.argv[3]
tr_sp_path = '../../data/train.csv'
tr_ds_path = '../../data/train_data/train.csv'
tr_dense_path = '../../output/ffm_gbdt/tr.gbdt.dense'
tr_sparse_path = '../../output/ffm_gbdt/tr.gbdt.sparse'

te_sp_path = '../../data/test.csv'
te_ds_path = '../../data/train_data/test.csv'
te_dense_path = '../../output/ffm_gbdt/te.gbdt.dense'
te_sparse_path = '../../output/ffm_gbdt/te.gbdt.sparse'


fields = ['creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'hour', 'cate_1', 'cate_2']

# 类别属性中出现频率最高的26个特征
target_cat_feats = []
fi = csv.DictReader(open('../../data/fc.31.csv', 'r'))
for t, row in enumerate(fi, start=1):
    target_cat_feats.append(row['Field'] + '-' + row['Value'])
    if t == 20:
        break
print(target_cat_feats)


# 读取频繁特征
def read_frequent_feats(threshold=10):
    frequent_feats = {}
    # fc.trav.t10.txt为出现频率超过10的表
    for row in csv.DictReader(open('../../data/fc.31.csv')):
        if int(row['Total']) < threshold:
            continue
        frequent_feats[row['Field'] + '-' + row['Value']] = row['Ratio']
    return frequent_feats


threshold = 10

# 出现频率超过10次的特征集合
frequent_feats = read_frequent_feats(threshold)


def tr_sparse():
    with open(tr_sparse_path, 'w') as f_s:
        for t, row in enumerate(csv.DictReader(open(tr_sp_path)), start=1):
            if 28 <= int(row['date']) <= 29:
                cat_feats = set()
                for feat in fields:
                    key = feat + '-' + row[feat]
                    cat_feats.add(key)

                feats = []
                for j, feat in enumerate(target_cat_feats, start=1):
                    if feat in cat_feats:
                        feats.append(str(j))

                # 将最常见的26个类别属性编码放入sparse文件中
                f_s.write(row['label'] + ' ' + ' '.join(feats) + '\n')
                if int(row['date']) > 29:
                    break
                if t % 1000000 == 0:
                    print('Line processed: {0}'.format(t))
    f_s.close()


def te_sparse():
    with open(te_sparse_path, 'w') as f_s:
        for t, row in enumerate(csv.DictReader(open(te_sp_path)), start=1):
            cat_feats = set()
            for feat in fields:
                key = feat + '-' + row[feat]
                cat_feats.add(key)

            feats = []
            for j, feat in enumerate(target_cat_feats, start=1):
                if feat in cat_feats:
                    feats.append(str(j))

            # 将最常见的26个类别属性编码放入sparse文件中
            f_s.write(row['label'] + ' ' + ' '.join(feats) + '\n')

            if t % 1000000 == 0:
                print('Line processed: {0}'.format(t))
    f_s.close()


field = []
fi = open('../../output/ffm_gbdt/feats.txt')

for x in next(fi).replace('\n', '').split(',')[1:]:

    field.append(x)
fi.close()


def dense(data_path, out_path):
    with open(out_path, 'w') as f_d:
        for t, row in enumerate(csv.DictReader(open(data_path)), start=1):
            feats = []
            for feat in field:
                val = row[feat]
                # 若数值缺失, 则为-10
                if val == '':
                    val = -10
                feats.append('{0}'.format(val))
            # 将数值属性放入.dense文件中
            f_d.write(row['label'] + ' ' + ' '.join(feats) + '\n')

            if t % 1000000 == 0:
                print('Line processed: {0}'.format(t))
    f_d.close()


if __name__=='__main__':
    tr_sparse()
    # dense(tr_ds_path, tr_dense_path)
    te_sparse()
    # dense(te_ds_path, te_dense_path)
