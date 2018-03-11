# _*_ coding: utf-8 _*_
import collections
from csv import DictReader

path = '../../data/'

field = ['creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'hour', 'cate_1', 'cate_2']

# THRESHOLD = 10

# [positive][total][index]
# table = collections.defaultdict(lambda: [0, 0, 0])

# for i, row in enumerate(DictReader(open('../../data/validation/train.csv'))):
#     label = row['label']
#     for field in fields:
#         value = row[field]
#         if label == '1':
#             table[field + '-' + value][0] += 1  # 0 特征值被点击的次数
#         table[field + '-' + value][1] += 1  # 特征值总的出现次数
#         table[field + '-' + value][2] = len(table)
#     if i % 100000 == 0:
#         print('Line processed: {0}'.format(i))


def fo_row(label, ctr_feats, clk_feats, cnv_feats):
    row = label
    for k in ctr_feats.keys():
        row += ',' + str(ctr_feats[k])
    for k in clk_feats.keys():
        row += ',' + str(clk_feats[k])
    for k in cnv_feats.keys():
        row += ',' + str(cnv_feats[k])
    return row


# 读取频繁特征
def read_frequent_feats(threshold, date):
    frequent_feats = collections.defaultdict(lambda: [0, 0, 0])
    # fc.trav.t10.txt为出现频率超过10的表
    for row in DictReader(open('../../data/fc.{0}.csv'.format(date))):
        if int(row['Total']) < threshold:
            continue
        frequent_feats[row['Field'] + '-' + row['Value']][0] = row['Ratio']
        frequent_feats[row['Field'] + '-' + row['Value']][1] = row['Total']
        frequent_feats[row['Field'] + '-' + row['Value']][2] = row['Pos']
    return frequent_feats


threshold = 10


def tr_feat(fo):
    # 出现频率超过10次的特征集合
    frequent_feats_28 = read_frequent_feats(threshold, 28)
    frequent_feats_29 = read_frequent_feats(threshold, 29)

    for t, row in enumerate(DictReader(open(path + 'train.csv', 'r')), start=1):
        if int(row['date']) == 28:
            cvr_feats = {}
            clk_feats = {}
            cnv_feats = {}
            for k in field:
                v = row[k]
                key = k + '-' + v
                if key in frequent_feats_28.keys():
                    # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                    ratio = frequent_feats_28[key][0]
                    cvr_feats[k] = (str(round(float(ratio), 5)))
                    clk_feats[k] = frequent_feats_28[key][1]
                    cnv_feats[k] = frequent_feats_28[key][2]
                else:
                    cvr_feats[k] = (str(0))
                    clk_feats[k] = (str(0))
                    cnv_feats[k] = (str(0))
            outrow = fo_row(row['label'], cvr_feats, clk_feats, cnv_feats)
            fo.write(outrow + '\n')

        if int(row['date']) == 29:
            cvr_feats = {}
            clk_feats = {}
            cnv_feats = {}
            for k in field:
                v = row[k]
                key = k + '-' + v
                if key in frequent_feats_29.keys():
                    # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                    ratio = frequent_feats_29[key][0]
                    cvr_feats[k] = (str(round(float(ratio), 5)))
                    clk_feats[k] = frequent_feats_29[key][1]
                    cnv_feats[k] = frequent_feats_29[key][2]
                else:
                    cvr_feats[k] = (str(0))
                    clk_feats[k] = (str(0))
                    cnv_feats[k] = (str(0))
            outrow = fo_row(row['label'], cvr_feats, clk_feats, cnv_feats)
            fo.write(outrow + '\n')

        if t % 1000000 == 0:
            print('Line processed: {0}'.format(t))


def te_feat(fo):
    frequent_feats_31 = read_frequent_feats(threshold, 31)
    for t, row in enumerate(DictReader(open(path + 'test.csv', 'r')), start=1):
        cvr_feats = {}
        clk_feats = {}
        cnv_feats = {}
        for k in field:
            v = row[k]
            key = k + '-' + v
            if key in frequent_feats_31.keys():
                # 计算某个特征值的点击率(出现该特征值被点击的次数 / 该特征值出现的总次数)
                ratio = frequent_feats_31[key][0]
                cvr_feats[k] = (str(round(float(ratio), 5)))
                clk_feats[k] = frequent_feats_31[key][1]
                cnv_feats[k] = frequent_feats_31[key][2]
            else:
                cvr_feats[k] = (str(0))
                clk_feats[k] = (str(0))
                cnv_feats[k] = (str(0))
        outrow = fo_row(row['label'], cvr_feats, clk_feats, cnv_feats)
        fo.write(outrow + '\n')
        if t % 1000000 == 0:
            print('Line processed: {0}'.format(t))


def get_header():
    header = 'label'
    for feat in field:
        header += ',' + feat + '-cvr'

    for feat in field:
        header += ',' + feat + '-clk'
    for feat in field:
        header += ',' + feat + '-cnv'
    return header


if __name__ == '__main__':
    from datetime import datetime

    start = datetime.now()
    header = get_header()
    print(header)
    ftrain = open('../../data/str.csv', 'w')
    ftrain.write(header + '\n')
    tr_feat(ftrain)

    print('31')
    ftest = open('../../data/ste.csv', 'w')
    ftest.write(header + '\n')
    te_feat(ftest)
    print(datetime.now() - start)
