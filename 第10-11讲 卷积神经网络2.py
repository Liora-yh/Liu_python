"""
基于 Inception（GoogLeNet 里的 InceptionA 模块）的 MNIST 手写数字分类模型，用纯 PyTorch 实现。
    核心亮点：Inception 多分支并行卷积，同时用 1×1、3×3、5×5、池化分支提取不同感受野的特征，
        最后拼接特征，相比单一卷积能捕捉更丰富信息。
    数据集：MNIST（28×28 单通道灰度手写数字，0~9 十分类）

整体流程：
    数据集加载 → 自定义 InceptionA 模块 → 搭建整体网络 Net → 交叉熵损失 + SGD 优化 → 循环训练 + 测试
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# 核心：InceptionA 模块
# design model using class
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 分支1：1×1卷积
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # 分支2：5×5卷积（先用1×1降维，减少参数量）
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # 分支3：两层3×3卷积（先用1×1降维）
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        # 分支4：平均池化 + 1×1卷积
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        # 分支1前向
        branch1x1 = self.branch1x1(x)

        # 分支2前向
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        # 分支3前向
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        # 分支4前向：平均池化，尺寸不变（padding=1 stride=1 kernel=3）
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 4个分支在通道维度拼接
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)  # b,c,w,h  c对应的是dim=1
        # shape: [batch, channel, w, h]，dim=1是通道维
"""
nception 设计思想
    多感受野并行提取特征：
        1×1：捕捉局部细粒度特征、降通道、减少计算量
        3×3：中等感受野
        5×5：更大感受野
        池化分支：保留原始全局信息
通道计算：16 + 24 + 24 + 24 = 88 通道，一共输出88个通道
padding 设置保证输出特征图宽高和输入完全一致，才能拼接
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入单通道灰度图，输出10通道
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 输入88通道，就是inceptionA输出的通道数
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)  # 88 = 24x3 + 16

        # 接收conv1输出的10通道特征
        self.incep1 = InceptionA(in_channels=10)  # 与conv1 中的10对应
        # 接收conv2输出的20通道特征
        self.incep2 = InceptionA(in_channels=20)  # 与conv2 中的20对应

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)         # batch大小
        # 第一层卷积+池化+inception
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        # 第二层卷积+池化+inception
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        # 展平：把 [B,C,W,H] → [B, C*W*H]，送入全连接
        x = x.view(in_size, -1)
        x = self.fc(x)

        return x

"""
维度推演，解释为什么 Linear 是 1408

输入：[64,1,28,28]
    1. conv1(1→10, k=5) → [64,10,24,24] → maxpool2d → [64,10,12,12]
    2. incep1 (输入 10 通道，输出 88 通道) → [64,88,12,12]
    3. conv2(88→20, k=5) → [64,20,8,8] → maxpool2d → [64,20,4,4]
    4. incep2 (输入 20 通道，输出 88 通道) → [64,88,4,4]
    5. 展平：88 * 4 * 4 = **1408** → 所以 fc 输入是 1408，输出 10

网络结构顺序：Conv1 → MaxPool → InceptionA1 → Conv2 → MaxPool → InceptionA2 → Flatten → Linear
"""

model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():       # 测试阶段不计算梯度，节省显存加速
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))

# torch.max(outputs, dim=1)：返回 (最大值，对应下标)，下标就是数字类别
# with torch.no_grad()：非常重要，关闭计算图，不保存梯度信息

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()