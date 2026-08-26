"""
基于 PyTorch 的二分类任务完整代码：
    用糖尿病数据集（8 个特征，预测是否患病，0/1 标签）搭建了一个 3 层全连接神经网络，
    使用 DataLoader 批量加载数据、BCELoss 二分类损失、SGD 梯度下降优化。
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
"""
Dataset：自定义数据集抽象类，用来封装自己的数据
DataLoader：负责分批次、打乱、多线程取数据，训练时用 mini-batch
"""

# prepare dataset


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        # xy[:, :-1]：所有行，除最后一列 → 特征矩阵
        # xy[:, [-1]]：所有行，只取最后一列 → 标签

    def __getitem__(self, index):       # 根据索引取单条样本,目的是为支持下标(索引)操作
        return self.x_data[index], self.y_data[index]

    def __len__(self):      # 返回数据集总样本数量，DataLoader 需要知道一共有多少数据，才能计算一共要迭代多少 batch
        return self.len


dataset = DiabetesDataset('diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)  # num_workers 多线程
"""
dataset=dataset：传入我们自定义的数据集对象
batch_size=32：一个批次 32 个样本，一次喂给模型 32 条数据
shuffle=True：每个 epoch 开始前打乱数据顺序，防止模型学到样本顺序偏见
num_workers=0：加载数据的子进程数量。Windows 下一般写 0，避免多进程报错；Linux 可以改成多线程加速
"""

# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
"""
前向流程：
    输入[batch,8] → linear1 → sigmoid → linear2 → sigmoid → linear3 → sigmoid → 输出[batch,1]
"""

model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')      # reduction='mean'：默认，把一个 batch 里所有样本的 loss 求平均值
"""
    如果用 BCEWithLogitsLoss，则不需要最后一层 sigmoid，这个代码里是 BCELoss，所以最后必须 sigmoid
"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
"""
    model.parameters()：拿到网络里所有 w 和 b（权重、偏置）
"""

# training cycle forward, backward, update
# if __name__ == '__main__':
#     for epoch in range(100):
#         for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
#             inputs, labels = data
#             y_pred = model(inputs)
#             loss = criterion(y_pred, labels)
#             print(epoch, i, loss.item())
#
#             optimizer.zero_grad()
#             loss.backward()
#
#             optimizer.step()
"""
外层循环：for epoch in range(100)
    epoch：一轮完整遍历全部数据集
    一共训练 100 轮；每一轮开头 DataLoader 会自动 shuffle 数据
内层循环：for i, data in enumerate(train_loader, 0)
    enumerate 遍历 train_loader，i是 batch 序号，data是当前批次的(inputs, labels)
    每次取出一个 batch（32 条样本）
    每次循环拿到两个变量：
        1、i → 当前 batch 的索引编号，从 0,1,2,3…… 依次递增
        2、data → 当前这一批的数据，就是(inputs, labels)

enumerate(train_loader, 0)
    enumerate()：自带序号计数器的遍历工具
    train_loader：可迭代对象，每次迭代吐出一个 batch 的数据（inputs, labels）
    0：计数器从 0 开始（这个 0 其实是默认值，写不写都一样）
    等价：一边取出每个 batch，一边给这个 batch 编上号 i
"""
for epoch in range(100):
    for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()