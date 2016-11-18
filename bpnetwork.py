'''BP NetWork'''
import math
import random

class BPNetwork():
    '''BPNetwork'''
    
    training_set = []
    rate = 0.7
    θ = [0.1] # 输出层阈值
    w = [0.1,0.1] # w1 w2
    γ = [0.1, 0.1] # γ1 γ2 隐含层阈值
    v = [0.1, 0.1, 0.1, 0.1] # v11, v21, v12, v22
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def train(self, data):
        '''Train network'''
        x1 = data[0]
        x2 = data[1]
        y = data[2]
        b1, b2 = self.calHide(x1, x2) #更新隐含层结果
        cal_y = self.calOut(b1, b2)
        # 反向传播，修改权值
        self.bp(x1, x2, b1, b2, y, cal_y)

    def bp(self, x1, x2, b1, b2, y, cal_y):
        '''误差反向传播'''
        # 调整w和θ
        # g * b1和g * b2为相应的负梯度，减少均方误差
        g = cal_y * (1 - cal_y) * (y - cal_y)
        Δw1 = self.rate * g * b1
        Δw2 = self.rate * g * b2
        self.w[0] += Δw1
        self.w[1] += Δw2
        Δθ = -self.rate * g
        self.θ[0] += Δθ

        # 调整v和γ
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
        '''Cal hidden layer'''
        # v11 * x1 + v21 * x2 - γ1 对于隐含层第一个结点
        b1 = self.sigmoid(x1 * self.v[0] + x2 * self.v[1] - self.γ[0])
        # v12 * x1 + v22 * x2 - γ2 对于隐含层的第二个结点
        b2 = self.sigmoid(x1 * self.v[2] + x2 * self.v[3] - self.γ[1])
        # 返回隐含层的结果
        return b1, b2
    
    def calOut(self, b1, b2):
        '''Cal out layer'''
        # 根据已有的隐含层结果作为输入
        # 返回输出层的结果
        return self.sigmoid(b1 * self.w[0] + b2 * self.w[1] - self.θ[0])

    def init(self):
        '''init'''
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

bp = BPNetwork()
bp.init()
e = []
print(bp.w)
print(bp.v)
for data in bp.training_set:
    e.append(bp.cal(data[0], data[1], data[2]))
print(sum(e)/len(e))
b1, b2 = bp.calHide(1, 0)
print(bp.calOut(b1, b2))