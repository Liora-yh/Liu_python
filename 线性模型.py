# Pytorch入门经典：用穷举法找线性回归权重 w
#数据集：y = 2x，样本(1,2),(2,4),(3,6)。
# 模型：最简单一元线性模型 \(\hat y = w\cdot x\)，【y = w * x】没有偏置 b。
# 损失：均方误差 MSE。
# 思路：把权重 w 从 0 到 4，以 0.1 步长全部试一遍，计算每个 w 对应的 MSE 损失，画出损失曲线，观察 w 取多少时损失最小。
import numpy as np
import matplotlib.pyplot as plt
# numpy：用来生成连续的权重 w 序列
# matplotlib.pyplot：绘图，画出 w‑loss 损失曲线

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# 输入 x，真实标签 y。真实关系：y = 2x，最优权重 w 理论等于 2

def forward(x):
    return x * w
# forward：前向计算，模型预测函数。
# 输入 x，输出预测值 y(尖) = w * x
# 注意：w 是全局变量，函数内部直接使用外部循环的 w。

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2
# 单个样本的损失：平方损失
# loss=(y(尖)-y)^2
# 预测值和真实值差的平方，衡量单个样本预测有多不准。

#===================穷举法====================

w_list = []     # w_list：保存每一次尝试的权重 w
mse_list = []   # mse_list：保存每个 w 对应的均方误差 MSE

# np.arange(0.0,4.1,0.1)：生成数组：0.0, 0.1, 0.2 ..., 4.0
# 为什么写 4.1：arange 左闭右开，终止写 4.1 才会包含 4.0。
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):    # zip(x_data,y_data)把 x、y 配对取出：(1,2)、(2,4)、(3,6)
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)

    # 内层循环：遍历全部 3 个样本：
        # y_pred_val = forward(x_val)：当前 w 下模型预测输出
        # loss_val：当前样本的平方损失
        # l_sum += loss_val：累加全部样本损失总和
        # 打印输出：制表符\t做缩进，打印每个样本：x，真实 y，预测 y，单个样本 loss。

    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)
    # w_list是空列表，专门存每一轮试出来的权重w
        # .append(w)：把当前循环里的w，追加放到w_list的尾巴上
    # mse_list是空列表，专门存这个w算出来的MSE（平均损失）
        # .append(l_sum / 3)：把当前w对应的均方误差，追加放到mse_list的尾巴上
    # MSE = (loss_1+loss_2+loss_3)/3
    # 均方误差：所有样本损失求平均。

plt.plot(w_list, mse_list)
plt.ylabel('Loss')      # 设置y轴的文字标签：纵坐标含义是损失Loss
plt.xlabel('w')         # 设置x轴的文字标签：横坐标含义是权重w
plt.show()
# plt.plot(X, Y)作用：画折线图，把一组 X 坐标、一组 Y 坐标的点依次连起来
# 画曲线图：
    # x 轴：权重 w
    # y 轴：MSE 损失 Loss
    # 运行后得到一条开口向上抛物线。
    # 抛物线最低点横坐标就是最优权重，大约在 w=2 附近，MSE 最小。