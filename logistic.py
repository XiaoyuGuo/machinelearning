from numpy import *
import random


def sigmoid(inX):
    return 1/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weight = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weight)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

dataSet = []
labels = []
k = 2
for i in range(100):
    dataSet.append([1.0, random.uniform(15,20),random.uniform(15,20)])
    labels.append(1)

for i in range(100):
    dataSet.append([1.0, random.uniform(5,10),random.uniform(5,10)])
    labels.append(2)


print(dataSet)
