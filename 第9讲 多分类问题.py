"""
PyTorch 实现 MNIST 手写数字识别的全连接神经网络（MLP，多层感知机） 代码。
    任务：输入 28×28 手写数字灰度图片，输出 0~9 的分类结果。
    整体流程：准备数据集 → 搭建网络模型 → 定义损失函数 & 优化器 → 循环训练 + 测试
MNIST：28×28 像素灰度手写数字，共 60000 训练集，10000 测试集。单张图片像素总数 28*28=784。
"""
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
"""
torchvision：计算机视觉常用数据集、图像预处理工具
transforms：图像预处理（转张量、归一化）
datasets：内置数据集（MNIST/CIFAR 等）
"""

# prepare dataset

batch_size = 64     # 一次送入网络 64 张图片做批量训练
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])  # 归一化,均值和方差
"""
transforms.Compose：把多个预处理操作打包串行执行
ToTensor()：
    把 PIL 图片 /numpy 数组 [H,W,C]（0~255 整数） → PyTorch 张量 [C,H,W]，数值缩放到 [0,1]
    MNIST 是单通道灰度图，shape 变成 (1,28,28)
Normalize(mean, std)：标准化 x = (x - mean)/std
    (0.1307,), (0.3081,) 是MNIST 数据集全局均值和标准差，是提前统计好的固定值，加速收敛。
"""

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)     # -1其实就是自动获取mini_batch
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)       # 最后一层不做激活，不进行非线性变换
"""
x.view(-1,784)：把输入张量变形
    输入原始 shape：[batch, 1, 28, 28]
    view 之后：[batch,784]，把图片拉成一维向量，适配全连接层
    -1 代表自动推断这个维度的值（就是 batch_size）
F.relu()：激活函数，引入非线性；没有 relu 多层网络等价于单层，拟合能力会很差
最后一层不加 relu、不加 softmax！
后面 CrossEntropyLoss 内部自带 softmax+log，直接接收原始 logits 输出，是标准写法
"""

model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# momentum=0.5：动量，加速收敛、缓解震荡
# 梯度下降有冲量，可以避免下降过程中停留在鞍点，没取到最优点

# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 获得一个批次的数据和标签
        inputs, target = data
        optimizer.zero_grad()
        # 获得模型预测结果(64, 10)
        outputs = model(inputs)
        # 交叉熵代价函数outputs(64,10),target（64）
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
        # 每 300 个 batch 打印一次平均 loss【每300轮输出一次】

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100 * correct / total))
"""
with torch.no_grad()：关闭计算图、不保存梯度，测试阶段不需要反向传播，节省显存、提速
torch.max(outputs.data, dim=1)
    outputs shape [64,10]，每一行是一张图片对应 0~9 的得分
    dim=1：在第 1 维（10 个类别维度）取最大值
    返回两个值：最大值、最大值对应的索引（也就是预测类别）
_ 表示丢弃最大值，只取 predicted 预测标签
predicted == labels：逐元素对比，得到布尔张量，sum()统计预测正确数量
最后输出测试集准确率
"""

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
# 一共训练 10 轮，每跑完一轮完整训练集，就跑一次测试集评估准确率