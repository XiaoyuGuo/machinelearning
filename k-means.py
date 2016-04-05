from numpy import *
import time
import matplotlib.pyplot as plt
import random

#初始化并返回k个随机点
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    
    return centroids

#计算两个向量之间的距离
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))
    
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    clusterAssment = mat(zeros((numSamples,2)))
    clusterChanged = True
    #初始化了两个点
    centroids = initCentroids(dataSet, k)
    
    #当聚类质心点改变的时候
    while clusterChanged:
        clusterChanged = False
        
        #range不生成数组，适应极大量的数据，更快一些
        for i in range(numSamples):
            minDist = 100000
            minIndex = 0
            
            #分别找到两质心和该点的距离，然后记录该点和质心的最短距离和质心索引
            for j in range(k):
                distance = euclDistance(centroids[j,:], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            
            #如果在评估矩阵中该点所对应的质心不等于新的质心
            #则将该点对应的质心索引（如：A或B）和其最短距离的平方记录到评估矩阵
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        
        #对于每一个质心
        for j in range(k):
            #返回评估矩阵中的非零且和质心有关的元素
            pointsInCluster = dataSet[nonzero(clusterAssment[:,0].A == j)[0]]
            #求新的质心
            centroids[j, :] = mean(pointsInCluster, axis = 0)
    
    #当质心不在改变，则说明函数收敛
    return centroids, clusterAssment
    
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    mark = ["or", "ob"]
    for i in range(numSamples):
        markIndex = int(clusterAssment[i,0])
        plt.plot(dataSet[i,0], dataSet[i,1], mark[markIndex])
    plt.show()


dataSet = []
k = 2
for i in range(100):
    dataSet.append([random.uniform(15,20),random.uniform(15,20)])

for i in range(100):
    dataSet.append([random.uniform(5,10),random.uniform(5,10)])
    
for i in range(100):
    dataSet.append([random.uniform(0,20),random.uniform(0,20)])
dataSet = mat(dataSet)
centroids, clusterAssment = kmeans(dataSet, k)
showCluster(dataSet, k, centroids, clusterAssment)