from csv import DictReader
from datetime import datetime

out_path = '../../data/daily_data/'


# records = 100000  # 总的记录条数
# oneday = 20000  # 一天的记录条数


def split_data(head):
    out_path = '../../data/daily_data/'
    if head == True:
        out_path = '../../data/daily_data_header/'

    last = 16
    fo = open(out_path+'/train_16.csv', 'w')

    fi = open('../../data/train.csv', 'r')
    header = next(fi)
    if head == True:
        fo.write(header)
    for t, row in enumerate(fi):
        # this = int(t / oneday)
        this = int(row.split(',')[1][:2]) - 1
        if last != this:
            # print('got! this={0} last={1}'.format(this, last))
            fo.close()
            last = this
            fo = open(out_path+'train_{0}.csv'.format(this + 1), 'w')
            if head == True:
                fo.write(header)

            print('day{0} completed !'.format(this))
        fo.write(row)


def split_test(head):
    out_path = '../../data/daily_data/'
    if head == True:
        out_path = '../../data/daily_data_header/'
    with open(out_path + 'test.csv', 'w') as fo:
        fi = open('../../data/test.csv', 'r')
        header = next(fi)
        if head == True:
            fo.write(header)
        for line in fi:
            fo.write(line)
        fi.close()


if __name__ == '__main__':
    start = datetime.now()
    head = True
    split_data(head)
    split_test(head)
    print(datetime.now() - start)
