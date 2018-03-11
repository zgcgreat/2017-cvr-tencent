from csv import DictReader

path = '../../data/'


def fun(data):
    with open(path+'data_no_header/{0}.csv'.format(data), 'w')as fo:
        fi = open(path+'{0}.csv'.format(data), 'r')
        next(fi)
        for line in fi:
            # print(int(line.split(',')[19]))
            if 24 <= int(line.split(',')[19]) <= 29:
                fo.write(line)

if __name__ == '__main__':
    fun('train')
    print('train completed !')
    fun('test')
    print('test completed !')
