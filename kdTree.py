#    KD Tree Algorithm
#    An optimize method to apply kNN
#    Author  : Guo Xiaoyu
#    Date    : 2017/3/4
#    Version : Python3.6.0

import random
import math

class KDNode():
    '''KD tree's node'''
    def __init__(self, data=None, split=None, lchild=None, rchild=None):
        '''
        data  : (feature1, feature2)
        split : split plane
        lchild: left child
        rchild: right child
        '''
        self.data = data
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

class KDTree():
    '''KD tree'''

    def calcVariance(self, int_list):
        '''Calc Variance of a list'''
        # Functional programming
        int_list = list(map(lambda x:x**2, int_list))
        # Return variance
        return sum(int_list)/len(int_list)

    def init(self, root, data):
        '''
        root: root of KD tree
        data: data set
        return root of KD tree
        '''
        size = len(data)
        if size == 0:
            return
        
        # Calc dimension of sample
        dimension = len(data[0])

        # Max variance of feature
        max_variance = 0

        split = 0

        # Find the split according to the variance
        for i in range(dimension):
            temp_list = []
            for sample in data:
                temp_list.append(sample[i])
            variance = self.calcVariance(temp_list)
            if variance > max_variance:
                max_variance = variance
                split = i
        
        data.sort(key=lambda x:x[split])
        sample = data[int(size/2)]

        # Construct a KDNode
        root = KDNode(sample, split)


        # Recursion

        # Samples before current sample
        root.lchild = self.init(root.lchild, data[0:int(size/2)])
        # Samples after current sample
        root.rchild = self.init(root.rchild, data[int(size/2+1):size])

        return root

    def findNN(self, kd_tree, query):
        '''
        Find nearest neiborhood
        '''
        nn = kd_tree.data
        min_distance = self.calcDistance(query, kd_tree.data)
        node_list = []
        temp_root = kd_tree

        while temp_root:
            # Record path
            node_list.append(temp_root)
            distance = self.calcDistance(query, temp_root.data)
            if min_distance > distance:
                nn = temp_root.data
                min_distance = distance
            split = temp_root.split
            if query[split] <= temp_root.data[split]:
                temp_root = temp_root.lchild
            else:
                temp_root = temp_root.rchild
        
        while node_list:
            back_point = node_list.pop()
            split = back_point.split

            if abs(query[split] - back_point.data[split]) < min_distance:
                if query[split] <= back_point.data[split]:
                    temp_root = back_point.right
                else:
                    temp_root = back_point.left
                
                if temp_root:
                    node_list.append(temp_root)
                    curDistance = self.calcDistance(query, temp_root.data)
                    if min_distance > curDistance:
                        min_distance = curDistance
                        nn = temp_root.data
        return nn, min_distance
    
    def calcDistance(self, pt1, pt2):
        '''
        Calc distance between two sample
        '''
        sum = 0
        for i in range(len(pt1)):
            sum += (pt1[i] - pt2[i]) ** 2
        return math.sqrt(sum)

kd_tree = KDTree()
kd_tree_root = kd_tree.init(None, [(1, 2), (3, 4), (5, 6), (7,8)])
print(kd_tree.findNN(kd_tree_root, (3, 3)))