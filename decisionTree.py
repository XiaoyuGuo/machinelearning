from numpy import *
from math import log

def createDataSet():
    dataSet = [
        [1,1,"yes"],
        [1,1,"yes"],
        [1,0,"no"],
        [0,1,"no"],
        [0,1,"no"]
        ]
    labels = ["no surfacing", "flippers"]
    return dataSet, labels


# ID3中，主要采用InformationGain对属性进行衡量
# C4.5中，主要采用InformationGainRatio对属性进行衡量
# 否则，属性选择则会倾向于类别较多的结点
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    
    #对于数据集里的每一个特征向量
    for featVec in dataSet:
        #最后一项为标签
        cuurentLabel = featVec[-1]  
        if cuurentLabel not in labelCounts:
            labelCounts[cuurentLabel] = 0
        labelCounts[cuurentLabel] += 1
    
    shannonEnt = 0.0
    
    #计算信息熵
    for key in labelCounts:
        #统计概率
        prob = labelCounts[key]/numEntries
        #累加
        shannonEnt -= prob * log(prob, 2)
    
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #以axis为轴分离
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 #去掉末尾结果
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    #对于每一个特征分量
    for i in range(numFeatures):
        feetList = [example[i] for example in dataSet]
        uniqueVals = set(feetList)
        newEntropy = 0.0
        #分离出子集并计算子集的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        #计算信息增益，找到信息增益最大的分离方式
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    #返回划分时信息增益最大的特征分量的下标
    return bestFeature
     
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount,key=lambda x:x[0], reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    #获取当前数据集的所有类别
    classList = [example[-1] for example in dataSet]

    #如果都是一个类别，则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1: #如果没数据了
        return majorityCnt(classList) #返回出现次数最多的类别
    
    #找到最优分隔特征和其对应的属性名称
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    #创建一个根节点
    myTree = {bestFeatLabel:{}}
    
    #删除标签中的该节点
    del(labels[bestFeat])
    
    #对于该特征，获取所有样本该特征的值，并得到其所有不同的值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    
    #对于该特征每一个不同的值
    for value in uniqueVals:
        subLabels = labels[:]
        #通过删除该特征值的向量组dataSet，递归求该值下的子树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree    

