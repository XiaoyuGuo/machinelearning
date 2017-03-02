# Support Vector Machine
# Author  : Guoxiaoyu
# Date    : 2017/3/2
# Version : Python3.6.0

class LabeledPoint():
    '''Labeled point'''
    def __init__(self, point=None, label=None):
        '''Construct func'''
        self.point = point
        self.label = label

class SVMModel():
    '''SVMModel'''
    def train(self):
        pass
    def predict(self):
        pass