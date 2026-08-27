"""
基于 PyTorch、带残差块 (Residual Block) 的 CNN 手写数字 MNIST 分类代码。
    核心亮点：手动实现简易残差结构，解决深层网络梯度消失问题，用来识别 0~9 手写数字。
MNIST：单通道 28×28 灰度手写数字数据集
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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])  # 归一化,均值和方差

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# 残差块 ResidualBlock
# design model using class
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)    # 先求和后激活
"""
残差核心公式：H(x) = F(x)+x
    x：原始输入（shortcut 短连接）
    F(x)：两层卷积学到的残差映射
    直接把输入加到卷积输出上，缓解深层网络梯度消失

参数细节:
    nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        输入通道 = 输出通道，3×3 卷积，padding=1 → **特征图宽高不变**
        28→28，12→12 这种，不会缩小尺寸，所以可以直接x+y相加（要求 shape 完全一致）
    流程：conv1 → relu → conv2 → 输入x + 卷积结果y → relu输出

这是基础版残差块，没有 BN 层，简化教学版本，不是原版 ResNet
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # 88 = 24x3 + 16

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10)  # 暂时不知道1408咋能自动出来的

    def forward(self, x):
        in_size = x.size(0)

        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)

        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
"""
conv1：输入通道 1（灰度图）→ 输出 16 通道，5×5 卷积
conv2：16 通道 → 32 通道，5×5 卷积
rblock1：16 通道残差块；rblock2：32 通道残差块
mp = MaxPool2d(2)：最大池化，窗口 2×2，宽高缩小一半
fc = Linear(512,10)：全连接层，512 维特征 →10 个类别（0~9）

输入：[batch, 1, 28, 28]             
    1、conv1(1→16, k=5)：输出尺寸 28-5+1=24 → [B,16,24,24]
    2、MaxPool2d(2)：24/2=12 → [B,16,12,12]
    3、rblock1(16通道)：padding=1，尺寸不变 → [B,16,12,12]
    4、conv2(16→32, k=5)：12-5+1=8 → [B,32,8,8]
    5、MaxPool2d(2)：8/2=4 → [B,32,4,4]
    6、rblock2(32通道)：尺寸不变 → [B,32,4,4] 

展平：32×4×4=512，正好对应 nn.Linear(512,10)

x.view(in_size, -1)：in_size=batch，-1自动计算 512，把 4 维卷积特征[B,32,4,4]转为 2 维[B,512]，送入全连接分类。
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
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()