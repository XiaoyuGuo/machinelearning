'''
BP NetWork
Single hidden layer
Error Back Propagation Algorithm
'''

import math
import random
from activation import ReLu

class BPNetwork():
    '''BPNetwork'''

    training_set = []
    rate = 0.20
    θ = [0.1] # Output Layer Threshold
    w = [0.1,0.1] # w1 w2
    γ = [0.1, 0.1] # γ1 γ2 Hidden Layer Threshold
    v = [0.1, 0.1, 0.1, 0.1] # v11, v21, v12, v22

    def train(self, data):
        '''Train network'''
        x1 = data[0]
        x2 = data[1]
        y = data[2]
        b1, b2 = self.calHide(x1, x2) #Input to Hidden
        cal_y = self.calOut(b1, b2) #Hidden to Output
        #Back Propagation
        self.bp(x1, x2, b1, b2, y, cal_y)

    def bp(self, x1, x2, b1, b2, y, cal_y):
        '''Error Back Propagation'''
        # Total goal: Minimize Mean Square error

        # Between output layer and hidden layer
        g = cal_y * (1 - cal_y) * (y - cal_y)
        Δw1 = self.rate * g * b1 # Through negative gradient and according to study rate, change weight
        Δw2 = self.rate * g * b2 # Through negative gradient and according to study rate, change weight
        self.w[0] += Δw1 # Change weight
        self.w[1] += Δw2 # Change weight
        Δθ = -self.rate * g # According to study rate, change threshold
        self.θ[0] += Δθ # Change threshold

        # Betwwen hidden layer and output layer
        e1 = b1 * (1 - b1) * self.w[0] * g
        e2 = b2 * (1 - b2) * self.w[1] * g
        Δv11 = self.rate * e1 * x1
        Δv21 = self.rate * e1 * x2
        Δv12 = self.rate * e2 * x1
        Δv22 = self.rate * e2 * x2
        self.v[0] += Δv11
        self.v[1] += Δv21
        self.v[2] += Δv12
        self.v[3] += Δv22
        Δγ1 = -self.rate * e1
        Δγ2 = -self.rate * e2
        self.γ[0] += Δγ1
        self.γ[1] += Δγ2
    
    def calHide(self, x1, x2):
        '''Calc hidden layer'''
        # Calc output of hidden layer
        b1 = ReLu(x1 * self.v[0] + x2 * self.v[1] - self.γ[0])
        b2 = ReLu(x1 * self.v[2] + x2 * self.v[3] - self.γ[1])
        # Return output of hidden layer
        return b1, b2
    
    def calOut(self, b1, b2):
        '''Calc out layer'''
        # Return output of output layer
        return ReLu(b1 * self.w[0] + b2 * self.w[1] - self.θ[0])

    def init(self):
        '''Initialize this bpnetwork'''
        for i in range(1, 1000001):
            if i % 4 == 0:
                self.training_set.append([0, 0, 0])
            elif i % 4 == 1: 
                self.training_set.append([1, 1, 0])
            elif i % 4 == 2:
                self.training_set.append([1, 0, 1])
            else:
                self.training_set.append([0, 1, 1])
        for data in self.training_set:
            self.train(data)

    def cal(self, x1, x2, y):
        b1, b2 = self.calHide(x1, x2)
        return ((self.calOut(b1, b2) - y) ** 2) / 2
    
    def predict(self, x1, x2):
        b1, b2 = self.calHide(x1, x2)
        return self.calOut(b1, b2)

bp = BPNetwork()
bp.init()
print(abs(bp.predict(1, 1) - 0))
print(abs(bp.predict(0, 0) - 0))
print(abs(bp.predict(1, 0) - 1))
print(abs(bp.predict(0, 1) - 1))