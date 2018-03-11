import numpy as np
import matplotlib.pyplot as plt

data_path = '../../data/'
out_path = '../../output/data_analysis/'

cnt = []
for i in range(24):
    cnt.append(0)


fi_te = open(data_path + 'test.csv', 'r')
next(fi_te)
for line in fi_te:
    s = line.replace('\n', '').split(',')
    label = s[0]
    date = int(s[1][:2])
    time = int(s[1][2:4])
    cnt[time] += 1

fi_te.close()
print(cnt)


def figure(list):
    x = np.arange(0, 24)
    y = list
    plt.plot(x, y, label='{0}'.format(i+17))
    # plt.plot(x, y, 'o')

    plt.legend(loc='best')

    plt.xticks(x)  # 设置x轴刻度
    # plt.title('{0}'.format())
    plt.xlabel('Time')
    plt.ylabel('Traffic')
    plt.grid(True)  # 添加网格
    plt.savefig('{0}test-cnt.png'.format(out_path), dpi=100)
    # plt.show()

figure(cnt)


def write_result(list):
    fo = open(out_path + 'test-cnt.csv', 'w')
    for i in range(len(list)):
        fo.write(str(list[i]) + ',')
    fo.write('\n')
    fo.close()
write_result(cnt)

