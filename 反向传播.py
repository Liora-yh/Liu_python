import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# torch.tensor()：工厂函数（官方推荐）；torch.Tensor()：类构造器，旧 API，不推荐日常写代码。
# w = torch.Tensor([1.0])
w = torch.tensor([1.0])     # w的初值为1.0
w.requires_grad = True      # 需要计算梯度


def forward(x):
    return x * w  # w是一个Tensor
# x是普通 python 浮点数，w是 Tensor；x*w会自动把 x 转为张量，构建计算图。
# 返回结果是 Tensor，携带梯度信息

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2
# 返回 Tensor，参与构建计算图

print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)          # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward()            # backward,compute grad for Tensor whose requires_grad set to True
        print('\tgrad:', x, y, w.grad.item())
        
        w.data = w.data - 0.01 * w.grad.data        # 权重更新时，注意grad也是一个tensor
        # w是带梯度跟踪(requires_grad=True)的张量。如果直接w = ...，会生成新张量，破坏计算图。
        # .data：访问张量里面原始数据，脱离梯度计算图，只修改数值，不记录运算。

        w.grad.data.zero_()         # after update, remember set the grad to zero

    print('progress:', epoch, l.item())         # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

print("predict (after training)", 4, forward(4).item())

"""
forward(4) 和 forward(4).item() 区别（PyTorch）
前提：forward(4) 返回的是 PyTorch 标量 Tensor（只有 1 个数字的张量对象）。

1. forward(4)
    返回：PyTorch Tensor 张量对象，不是普通 Python 数字。
    打印输出示例：
        predict (after training) 4 tensor(7.9998, grad_fn=<MulBackward0>)
    会显示 tensor(xxx)，还附带 grad_fn（自动求导的梯度函数标记），代表这个数还在计算图里面，支持反向传播求梯度。
    类型：torch.Tensor，不能直接拿来做普通 python 数学运算、存入列表、写文件。

2. forward(4).item()
    .item()：把只有单个元素的 Tensor，提取出来变成 Python 原生浮点数 float。
    打印输出示例：
        predict (after training) 4 7.999777758621207
    输出干干净净，只有纯数字，没有tensor()、没有grad_fn。
    类型：python 内置float，脱离 PyTorch 计算图，不再参与反向传播。

.item() 只能用于只有 1 个元素的 tensor，如果 tensor 里面有多个数字，调用.item()直接报错：only one element tensors can be converted to Python scalars。

什么时候用哪个？
    打印、绘图、保存数值、统计 loss  →  用 .item()
    训练阶段、反向传播之前          →  千万不要调用.item()
"""