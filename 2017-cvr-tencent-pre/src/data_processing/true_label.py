fi = open('../../output/xgb/validation.csv', 'r')
fo = open('../../output/xgb/validation.csv', 'w')
fo.write('id,label\n')
next(fi)
for t, row in enumerate(fi, start=0):
    fo.write(str(t) + ',' + row.replace('\n', '').split(',')[1] + '\n')

fi.close()
fo.close()
