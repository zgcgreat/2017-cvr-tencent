import operator
import collections
import subprocess
'''
标记除了label不同的重复数据,
输出：
    第几次出现 appear_times
    对于出现两次以上的数据，第一次出现标记为1,再次出现如果label为0,标记为2, 如果label为1, 标记为3  sign
    
'''

appear_time = collections.defaultdict(lambda: 0)  # 记录数据出现的次数
dup = set()  # 记录重复数据
last_time = {}


def appear_times(filename):
    with open('../../output/feature_data/{0}_sign-tmp.csv'.format(filename), 'w') as fo:
        fi = open('../../data/{0}.csv'.format(filename), 'r')
        header = next(fi)
        fo.write(header.replace('\n', '') + ',appear_times\n')
        # fo.write('appear_times\n')
        for t, row in enumerate(fi, start=1):
            cur_row = row.replace('\n', '')
            s = row.replace('\n', '').split(',')
            date = s[20]
            hour = s[21]
            minute = s[22]
            time = int(date) * 24 * 60 + int(hour) * 60 + int(minute)

            ss = s[1]+s[2]+s[18]  # creativeID+userID+appID

            if ss in appear_time.keys():
                if time - last_time[ss] <= 2:  # 如果两次点击的时间差小于2分钟，视为重复点击
                    appear_time[ss] += 1
                    dup.add(ss)
            else:
                appear_time[ss] = 1

            sig = appear_time[ss]
            last_time[ss] = time

            fo.write(cur_row + ',' + str(sig) + '\n')
            # fo.write(str(sig) + '\n')
            # if t % 1000000 == 0:
            #     # print(cur_row)
            #     print('Line processed:', t)
    fi.close()


def sign_tr(filename):
    with open('../../output/feature_data/{0}_sign.csv'.format(filename), 'w') as fo:
        fi = open('../../output/feature_data/{0}_sign-tmp.csv'.format(filename), 'r')
        header = next(fi)
        # fo.write(header.replace('\n', '') + ',sign\n')
        fo.write('label,date,appear_times,sign\n')
        for t, row in enumerate(fi, start=1):
            date = row.split(',')[20]
            if 28 <= int(date) <= 29:

                s = row.replace('\n', '').split(',')
                date = s[20]

                ss = s[1] + s[2] + s[18]  # creativeID+userID+appID

                appear_times = int(row.replace(',', '').replace('\n', '')[-1:])
                label = int(row.split(',')[0])

                if ss in dup:
                    if appear_times == 1:
                        sig = 1
                    if appear_times > 1:
                        sig = 2
                    # if appear_times > 1:
                    #     sig = 3
                else:
                    sig = -1

                fo.write(str(label) + ',' + date + ',' + str(appear_times) + ',' + str(sig) + '\n')

        fi.close()


def sign_te(filename):
    with open('../../output/feature_data/{0}_sign.csv'.format(filename), 'w') as fo:
        fi = open('../../output/feature_data/{0}_sign-tmp.csv'.format(filename), 'r')
        header = next(fi)
        # fo.write(header.replace('\n', '') + ',sign\n')
        fo.write('label,date,appear_times,sign\n')
        for t, row in enumerate(fi, start=1):
            s = row.replace('\n', '').split(',')
            date = s[20]

            ss = s[1] + s[2] + s[18]  # creativeID+userID+appID

            appear_times = int(row.replace(',', '').replace('\n', '')[-1:])
            label = int(row.split(',')[0])

            if ss in dup:
                if appear_times == 1:
                    sig = 1
                if appear_times > 1:
                    sig = 2
                # if appear_times > 2:
                #     sig = 3
            else:
                sig = -1

            # fo.write(cur_row + ',' + str(sig) + '\n')
            fo.write(str(label) + ',' + date + ',' + str(appear_times) + ',' + str(sig) + '\n')

        fi.close()


if __name__ == '__main__':
    print('train...')
    appear_times('train')

    print('test...')
    appear_times('test')

    sign_tr('train')
    print('test')
    sign_te('test')

    cmd = 'rm {path}train_sign-tmp.csv {path}test_sign-tmp.csv'.format(path='../../output/feature_data/')
    subprocess.call(cmd,shell=True)