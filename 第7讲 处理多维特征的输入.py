"""
基于 PyTorch 搭建多层全连接神经网络，做糖尿病二分类任务的代码：
    输入 8 个特征，预测是否患糖尿病（0/1 二分类），使用BCELoss 二分类损失 + SGD 梯度下降训练，
    最后画出 loss 变化曲线。
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# prepare dataset
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
"""
np.loadtxt：读取文本格式 csv，逗号delimiter=','分隔，数据转成 32 位浮点数
"""

x_data = torch.from_numpy(xy[:, :-1])  # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
"""
第一个: → 取所有样本行
:-1 → 列从 0 到倒数第二列，排除最后一列标签，也就是特征 X，一共 8 维特征

"""

y_data = torch.from_numpy(xy[:, [-1]])  # [-1] 最后得到的是个矩阵
"""
[-1] 是列表形式，输出 shape 是[样本数,1]二维矩阵
如果写成xy[:, -1]会变成一维向量[样本数]，BCELoss 要求预测值和标签维度匹配，所以这里用[-1]保持二维
torch.from_numpy()：把 numpy 数组转为 pytorch 张量 tensor，可以参与 GPU / 自动求导计算

xy[:,-1] → shape (N,) 一维数组
xy[:,[-1]] → shape (N,1) 二维矩阵 ✅ 适配 BCELoss
"""

# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 输入数据x的特征是8维，x有8个特征
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()  # 将其看作是网络的一层，而不是简单的函数使用

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))  # y hat
        return x

# 实例化网络
model = Model()

# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss(reduction='mean')
"""
reduction='mean'：对所有样本 loss 求平均值（旧版本参数叫size_average，已经废弃）
"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []
# training cycle forward, backward, update
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()