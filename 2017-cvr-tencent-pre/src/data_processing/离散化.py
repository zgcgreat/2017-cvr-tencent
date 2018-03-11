
path = '../../data/'


def sep(feat):
    return int(float(feat) / 0.0001)

fi_tr = open(path + 'train-ctr.csv', 'r')
fo_tr = open(path + 'train-c.csv', 'w')

header = next(fi_tr)
with fo_tr as fo:
    fo.write(header)
    for t, line in enumerate(fi_tr, start=1):
        feats = []
        s = line.replace('\n', '').split(',')
        label = s[0]
        for feat in s[1:]:
            feats.append(sep(feat))
        fo.write(str(label) + ',' + ','.join(str(feat) for feat in feats) + '\n')

        if t % 100000 == 0:
            print('Line processed:', t)
fi_tr.close()
fo_tr.close()


fi_te = open(path + 'test-ctr.csv', 'r')
fo_te = open(path + 'test-c.csv', 'w')

header = next(fi_te)
with fo_te as fo:
    fo.write(header)
    for t, line in enumerate(fi_te, start=1):
        feats = []
        s = line.replace('\n', '').split(',')
        label = s[0]
        for feat in s[1:]:
            feats.append(sep(feat))
        fo.write(str(label) + ',' + ','.join(str(feat) for feat in feats) + '\n')

        if t % 100000 == 0:
            print('Line processed:', t)
fi_te.close()
fo_te.close()
