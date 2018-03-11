import csv

""

""

fo = csv.writer(open('../../data/tr_new.csv', 'w'))
fi = csv.reader(open('../../data/train.csv', 'r'))
header = next(fi)
fo.writerow(header)
# print(header)
for t, line in enumerate(fi, start=1):
    if 22 <= int(line[1][:2]) < 31:
        fo.writerow(line)
    if t % 1000000 == 0:
        print('Line read:', t)
