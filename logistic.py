from numpy import *
import time
  
def sigmoid(inX):
    '''Sigmoid method'''
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '''Use gradient ascent function'''
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()

    m, n = shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500
    weights = ones((n, 1))

    print(dataMatrix * weights)
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def init():
    dataMat = []; labelMat = []
    for i in range(1000):
        dataMat.append([1.0, float(random.uniform(0,5)), float(random.uniform(0,5))])
        labelMat.append(1)
    for i in range(1000):
        dataMat.append([1.0, float(random.uniform(15,20)), float(random.uniform(15,25))])
        labelMat.append(0)

    weights = gradAscent(dataMat, labelMat)
    print(weights)
    print(sigmoid(mat([1.0, 3, 3]) * weights))

init()