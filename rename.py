# 利用Pytorch解决XOR问题
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from bitvector.readdata1 import read_csv_data

# [1,0,0] 与  [0,1,0] 异或       [0,0,1] 或
data = np.array([[1, 0, 1, 0, 0,0],
                 [0, 1, 1, 0, 0,0],
                 [1, 1, 1, 0, 0,0],
                 [0, 0, 1, 0, 0,0],

                 [1, 0, 0, 1, 0,0],
                 [0, 1, 0, 1, 0,0],
                 [1, 1, 0, 1, 0,0],
                 [ 0, 0, 0, 1, 0,0],

                 [1, 0, 0, 0, 1,0],
                 [0, 1, 0, 0, 1,0],
                 [1, 1, 0, 0, 1,0],
                 [0, 0, 0, 0, 1,0],

                 [1, 0, 0, 0, 0, 1],
                 [0, 1, 0, 0, 0, 1],
                 [1, 1, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]

                 ],dtype='float32')

y = np.array([[0,0],
              [0,0],
              [1,0],
              [0,0],

              [1,0],
              [1,0],
              [0,0],
              [0,0],

              [1,0],
              [1,0],
              [1,0],
              [0,0],

              [1, 0],
              [1, 0],
              [0, 1],
              [0, 0],
              ], dtype='float32')
# x = data[:, :5]
csv_file = 'bitwise_operations.csv'  # Replace with your CSV file path

data, y = read_csv_data(csv_file)
x = data

# y = data[:, 2]

print(x)

# 初始化权重变量
def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.)
        m.bias.data.fill_(0.)


class Bit(nn.Module):
    def __init__(self):
        super(Bit, self).__init__()
        # self.fc1 = nn.Linear(5, 8)   # 一个隐藏层 3个神经元
        # self.fc2 = nn.Linear(8, 7)  # 一个隐藏层 3个神经元
        # self.fc3 = nn.Linear(7, 4)  # 一个隐藏层 3个神经元
        # self.fc4 = nn.Linear(4, 1)   # 输出层 1个神经元

        self.fc1 = nn.Linear(199, 200)  # 一个隐藏层 3个神经元
        self.fc2 = nn.Linear(200, 100)  # 一个隐藏层 3个神经元
        self.fc3 = nn.Linear(100, 64)  # 一个隐藏层 3个神经元
        # self.fc4 = nn.Linear(4, 1)  # 输出层 1个神经元

    def forward(self, x):
        h1 = F.sigmoid(self.fc1(x))  # 之前也尝试过用ReLU作为激活函数, 太容易死亡ReLU了.
        h2 = F.sigmoid(self.fc2(h1))
        h3 = F.sigmoid(self.fc3(h2))
        # h4 = F.sigmoid(self.fc4(h3))

        return h3


net = Bit()
# net.apply(weight_init_normal)
x = torch.Tensor(x.reshape(-1, 199))
print(x)
y = torch.Tensor(y.reshape(-1,64))

# 定义loss function
criterion = nn.BCELoss()  # MSE
# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)  # SGD
# 训练
for epoch in range(10000):
    optimizer.zero_grad()   # 清零梯度缓存区
    out = net(x)
    loss = criterion(out, y)
    print(loss)
    loss.backward()
    optimizer.step()  # 更新

# 测试
test = net(x)
print(test)
print("input is {}".format(x.detach().numpy()))
print('out is {}'.format(test.detach().numpy()))

for item in test.detach().numpy():
    for i in item:
        if(i > 0.5):
            # print("1  ")
            print('1 ',  end='')
        else:
            print("0 " ,end='')
    print(" ")