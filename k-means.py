from numpy import *
import time
import matplotlib.pyplot as plt
import random

def initCentroids(dataSet, k):
    '''随机选取数据集中的k个点'''
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    
    return centroids

def euclDistance(vector1, vector2):
    '''返回两个特征向量之间的欧几里得距离'''
    return sqrt(sum(power(vector2 - vector1, 2)))
    
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    clusterAssment = mat(zeros((numSamples,2)))
    clusterChanged = True
    #初始化了两个点
    centroids = initCentroids(dataSet, k)
    
    #当聚类中心改变的时候
    while clusterChanged:
        clusterChanged = False
        
        for i in range(numSamples):
            minDist = 100000
            minIndex = 0
            
            #分别找到两质心和该点的距离，然后记录该点和质心的最短距离和质心索引
            for j in range(k):
                distance = euclDistance(centroids[j,:], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            
            #如果在评估矩阵中该点所对应的中心不等于新的中心
            #则将该点对应的中心索引和其最短距离的平方记录到评估矩阵
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        
        #对于每一个聚类中心
        for j in range(k):
            #返回评估矩阵中的非零且和质心有关的元素
            pointsInCluster = dataSet[nonzero(clusterAssment[:,0].A == j)[0]]
            #求新的聚类中心
            centroids[j, :] = mean(pointsInCluster, axis = 0)
    
    #当中心不再改变，则说明函数收敛
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