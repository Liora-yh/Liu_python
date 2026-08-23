"""
    Try to use the model, and draw the cost graph
    Linear Model: y(尖) = x * w + b

    画出线性模型y(尖) = x * w + b的损失3D曲面图
    任务：给定线性回归，使用MSE均方误差损失，用np.meshgrid生成w、b网格，
    向量化计算损失，绘制 3D cost 曲面。完整可直接运行Python代码：

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 模拟训练数据集
x = np.array([1.0, 2.0, 3.0])
y_true = np.array([2.0, 4.0, 6.0])  # 真实关系 y=2*x，理想w=2,b=0

# 2. 生成w、b网格 (meshgrid)
w_range = np.linspace(-2, 4, 100)
b_range = np.linspace(-2, 2, 100)
W, B = np.meshgrid(w_range, b_range)

# 3. 向量化计算MSE损失
# 广播机制：把每个(w,b)组合对全部样本算预测，再求均方误差
cost = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        w = W[i,j]
        b = B[i,j]
        y_hat = x * w + b
        loss = np.mean((y_hat - y_true)**2)
        cost[i,j] = loss

# 也可以完全向量化版本，去掉双重循环，速度更快：
# W_flat = W[..., np.newaxis]
# B_flat = B[..., np.newaxis]
# y_hat_all = x * W_flat + B_flat
# cost = np.mean((y_hat_all - y_true)**2, axis=-1)

# 4. 绘制3D损失曲面
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, B, cost, cmap='coolwarm', edgecolor='none')

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Cost Value')
ax.set_title('Cost Surface for Linear Model $\hat y = x\omega + b$')
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()