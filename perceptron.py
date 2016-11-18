import random

w = [random.uniform(-1, 1), random.uniform(-1, 1)]  # w1, w2
b = random.uniform(-1, 1)  # sita 阈值
rate = 0.001
data = []
for i in range(0, 1000000, 1):
    x1 = random.uniform(0, 1)
    x2 = random.uniform(0, 1)
    if x1 + x2 >= 0.75:
        data.append([(x1, x2), 1])
    else:
        data.append([(x1, x2), -1])


def learn(vec, flag):
    global b
    w[0] = w[0] + rate * (flag - sign(vec[0], vec[1])) * \
        vec[0]  # （真实值减去计算值） * 学习率 * x1
    w[1] = w[1] + rate * (flag - sign(vec[0], vec[1])) * \
        vec[1]  # （真实值减去计算值） * 学习率 * x2
    b = b + rate * (flag - sign(vec[0], vec[1])) * -1


def sign(x1, x2):
    global b
    if x1 * w[0] + x2 * w[1] - b > 0:
        return 1
    else:
        return -1

for d in data:
    learn(d[0], d[1])


def check(x1, x2):
    if sign(x1, x2) == 1:
        return True
    else:
        return False
        
print(-w[1]/w[0])
print(b/w[0])
