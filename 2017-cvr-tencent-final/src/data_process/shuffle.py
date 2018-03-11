# _*_ coding: utf-8 _*_

'''
打乱数据集顺序
'''
import random
import time

start = time.time()

print('shuffling dataset...')

input = open('../../data/traincp.csv', 'r')
output = open('../../data/train_data/train.csv', 'w')


lines = input.readlines()

outlines = []
output.write(lines.pop(0))  # pop()方法, 传递的是待删除元素的index
while lines:
    line = lines.pop(random.randrange(len(lines)))
    output.write(line)

input.close()
output.close()

print('dataset shuffled !')

print('Time spent: {0:.2f}s'.format(time.time() - start))
