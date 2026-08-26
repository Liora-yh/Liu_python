"""
PyTorch 实现逻辑回归（二分类） 的完整示例：
    用 1 个特征（Hours）预测二分类概率（是否通过，0/1 标签）。

逻辑回归本质：线性层 + Sigmoid，输出 0~1 之间的概率值
数据集：x=[1,2,3]，对应标签 y=[0,0,1]

"""
import torch
import numpy as np
import matplotlib.pyplot as plt

# import torch.nn.functional as F

# prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])    # 维度：2 维，shape = torch.Size([3, 1])
"""
双层括号，所以是二维张量
外层[]：代表样本集合，一共 3 个样本
内层每个[]：代表单个样本的特征，每个样本 1 个特征
形状：[3, 1] → [样本数量, 特征数量]，二维（2 个维度）

一维写法：[1.0, 2.0, 3.0]
维度：1 维，shape = torch.Size([3])
"""

y_data = torch.Tensor([[0], [0], [1]])      # y 是二分类标签：0 / 1


# design model using class 搭建逻辑回归模型（继承 nn.Module）
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # y_pred = F.sigmoid(self.linear(x))
        # nn.Sigmoid()：是类（Module，网络层）
        # F.sigmoid()：是普通函数（⚠️ 新版 PyTorch 已经废弃，推荐改用torch.sigmoid()）
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
"""
nn.Linear(1,1)：线性变换 z = w*x + b
torch.sigmoid(z)：\(\sigma(z)=\frac{1}{1+e^{-z}}\)，把线性输出映射到(0,1)，代表属于类别 1 的概率
forward：前向传播逻辑，不用自己写反向，pytorch 自动微分
"""

model = LogisticRegressionModel()

# construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
# 现在由于Python版本原因criterion = torch.nn.BCELoss(size_average = False）也会引发warning，括号里面改成reduction='sum'
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
"""
BCELoss：二分类交叉熵损失，专门配合 sigmoid 输出使用
    reduction='sum'：把所有样本损失直接累加（不做均值），老版本size_average=False等价，现在已经废弃
SGD：随机梯度下降，lr=0.01学习率，优化w,b
"""

# training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)              # forward: predict ①前向传播：算预测概率
    loss = criterion(y_pred, y_data)    # ②计算损失
    print(epoch, loss.item())

    optimizer.zero_grad()               # ③梯度清零！非常关键，pytorch梯度会累加
    loss.backward()                     # 反向传播，自动求w,b的梯度
    optimizer.step()                    # ④根据梯度更新参数 w = w - lr*grad

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
"""
model里的self.linear = torch.nn.Linear(1,1)，这个线性层内部存着权重 w和偏置 b，
它们本质都是PyTorch 张量 (tensor)，不是普通数字
如果直接 print (model.linear.weight)，输出会长这样：
    tensor([[0.8624]], requires_grad=True)
里面包含数值，还附带梯度信息requires_grad=True，我们只想拿到纯数字，所以用.item()
"""
"""
.item() 的作用: 把只包含单个数值的张量，提取成普通 Python 数值（float）【第4讲】
限制：只有张量里只有 1 个元素的时候才能用.item()，多元素张量不能用
"""

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1)) #reshape, 200行1列
"""
拆成两部分理解:
    torch.Tensor(x)：
        把上面numpy 数组 x 转换成 PyTorch 张量 tensor
        模型model只能接收 tensor 输入，不能直接接收 numpy 数组
    .view((200, 1))
        view()是 tensor 的形状重塑函数，等价 reshape
        (200,1) → 新形状：200 行，1 列
含义：200 个样本，每个样本 1 个特征，正好符合前面说的[样本数, 特征数]二维格式！
如果不view，tensor 形状是[200]一维，直接丢进 Linear 层会报错
变量x_t含义：t 代表 tensor，用来送入模型的输入张量
"""

y_t = model(x_t)
y = y_t.data.numpy()    # y = y_t.detach().numpy()  # y_t是tensor，y是numpy数组
"""
链式调用，两步操作：
    .data：老版 pytorch 写法，剥离 tensor 的梯度、计算图信息，只提取纯数据（不参与求导）
        新版本推荐.detach()，即y_t.detach().numpy()
    .numpy()：把 PyTorch 张量 → 转回 numpy 数组
        matplotlib 绘图函数plt.plot只认 numpy 数组 /python 列表，不认 tensor
"""

plt.plot(x, y)      # matplotlib 绘图核心函数，画曲线
"""
第一个参数x：横坐标（0~10 的小时）
第二个参数y：纵坐标（对应的预测概率 0~1）
程序会一一匹配 (x [0],y [0])、(x [1],y [1])…… 一共 200 个点，连成平滑的 Sigmoid 曲线
"""

plt.plot([0, 10], [0.5, 0.5], c ='r') #color = red
"""
也是plt.plot，画一条红色水平参考线
[0,10]：x 坐标起点 0，终点 10
[0.5,0.5]：y 坐标全程固定 0.5
c='r'：color='red'，红色
意义：分类阈值线
    逻辑回归判定规则：
        概率＞0.5 → 判定类别 1（通过）
        概率＜0.5 → 判定类别 0（不通过）
这条红线一眼就能看出：x 大于多少的时候，预测结果会从 0 变成 1
"""

plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()      # 给图像加上网格线，方便看坐标数值，读图更清晰
plt.show()