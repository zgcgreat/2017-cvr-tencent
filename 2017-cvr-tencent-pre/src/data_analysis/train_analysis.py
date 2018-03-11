import numpy as np
import matplotlib.pyplot as plt

data_path = '../../data/'
out_path = '../../output/data_analysis/'


def init():
    cnt = []
    for i in range(14):
        cnt.append([])
        for j in range(24):
            cnt[i].append(0)

    cv = []
    for i in range(14):
        cv.append([])
        for k in range(24):
            cv[i].append(0)

    cvr = []
    for i in range(14):
        cvr.append([])
        for l in range(24):
            cvr[i].append(0)
    return cnt, cv, cvr

cnt, cv, cvr = init()

fi_tr = open(data_path + 'train.csv', 'r')
next(fi_tr)
for line in fi_tr:
    s = line.replace('\n', '').split(',')
    label = s[0]
    date = int(s[1][:2])
    time = int(s[1][2:4])
    cnt[date-17][time] += 1
    if label == '1':
        cv[date - 17][time] += 1
fi_tr.close()
print(cnt)
print(cv)

for i in range(14):
    for j in range(24):
        cvr[i][j] = cv[i][j] / cnt[i][j]
print(cvr)


def figure(list):
    fo_name = ''
    if list == cnt:
        fo_name = 'cnt'
    if list == cv:
        fo_name = 'cv'
    if list == cvr:
        fo_name = 'cvr'
    x = np.arange(0, 24)
    for i in range(len(list)):
        y = list[i]
        plt.plot(x, y, label='{0}'.format(i+17))
        # plt.plot(x, y, 'o')

    plt.legend(loc='best')
    # pl.legend([plot1, plot2], ('budget', 'spend'), loc = 1)

    plt.xticks(x)  # 设置x轴刻度
    # plt.title('{0}'.format())
    plt.xlabel('Time')
    plt.ylabel('Traffic')
    plt.grid(True)  # 添加网格
    plt.savefig('{0}train-{1}.png'.format(out_path, fo_name), dpi=100)
    # plt.show()

figure(cnt)
figure(cv)
figure(cvr)


def write_result(list):
    fo_name = ''
    if list == cnt:
        fo_name = 'cnt'
    if list == cv:
        fo_name = 'cv'
    if list == cvr:
        fo_name = 'cvr'
    fo = open(out_path + 'train-{0}.csv'.format(fo_name), 'w')
    for i in range(len(list)):
        for j in range(len(list[i])):
            fo.write(str(list[i][j]) + ',')
        fo.write('\n')
    fo.close()

write_result(cnt)
write_result(cv)
write_result(cvr)
