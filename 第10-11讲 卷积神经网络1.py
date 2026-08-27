"""
**基于 PyTorch 搭建简易 CNN 卷积神经网络，训练 MNIST 手写数字识别数据集，最后画出测试集准确率随轮次变化曲线**
    MNIST：28×28 灰度手写数字图片，类别 0~9，输入通道 = 1
"""
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),                                  # 1. 转张量：像素0~255 → 0~1，维度(H,W,C)→(C,H,W)
    transforms.Normalize((0.1307,), (0.3081,))    # 2. 标准化：(x-均值)/标准差，MNIST预先统计好的均值方差
])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class

# 搭建 CNN 网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层1：输入通道1(灰度图)，输出通道10，卷积核5×5
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        # 卷积层2：输入通道10，输出通道20，卷积核5×5
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # 最大池化层 2×2，步长默认2
        self.pooling = torch.nn.MaxPool2d(2)
        # 全连接层：输入320个神经元，输出10个（对应0~9十分类）
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # x 输入shape: [batch_size, 1, 28, 28]
        # flatten data from (n,1,28,28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))     # conv1 → 池化 → relu激活
        x = F.relu(self.pooling(self.conv2(x)))     # conv2 → 池化 → relu激活
        x = x.view(batch_size, -1)  # 展平20×4×4=320：把4维特征图转为2维 [N, 320]; -1 此处自动算出的
        x = self.fc(x)

        return x


model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data   # inputs:[64,1,28,28], target:[64] 真实标签
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
            _, predicted = torch.max(outputs.data, dim=1)   # 取概率最大的索引作为预测数字
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print('accuracy on test set: %d %% ' % (100 * correct / total))
    acc = 100 * correct / total
    print('accuracy on test set: %d %% ' % acc)
    return acc

if __name__ == '__main__':
    # for epoch in range(10):
    #     train(epoch)
    #     test()
    epoch_list = []
    acc_list = []

    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.show()