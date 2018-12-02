# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 10:33:28 2018

@author: zh
"""

import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt

data = pd.read_csv('../input/train.csv')
#test_data = pd.read_csv('../input/test.csv')

train_data = data[data.columns[1:]][:40000]
train_label = data[data.columns[0]][:40000]

test_data = data[data.columns[1:]][40000:42000]
test_label = data[data.columns[0]][40000:42000]

#pic = test_data[26:27].values.reshape(28,28)
#plt.imshow(pic, cmap=plt.cm.gray)

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

counter = 0
for i in range(100):
    count = classify0(test_data[i:i+1], train_data, train_label, 3)
    print('i:{},predict:{},true:{}'.format(i,count, test_label[i:i+1].item()))
    if (count == test_label[i:i+1].item()):
        counter += 1

print('{:.2%}'.format(counter/100))