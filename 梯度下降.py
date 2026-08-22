import matplotlib.pyplot as plt

# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial guess of weight
w = 1.0

# define the model linear model y = w*x
def forward(x):
    return x * w


# define the cost function MSE
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)
# len(xs) = 样本数量，这里等于 3（3 组数据）

# define the gradient function  gd
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


epoch_list = []     # 保存迭代轮次（0，1，2 …99）
cost_list = []      # 保存每一轮对应的损失 MSE
print('predict (before training)', 4, forward(4))
# 训练之前做一次预测。此时 w=1.0，forward(4)=4*1 =4
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val  # 0.01 learning rate
    # print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    print(f'epoch: {epoch}, w={w:.2f}, loss={cost_val:.2f}')
    epoch_list.append(epoch)
    cost_list.append(cost_val)
# print('predict (after training)', 4, forward(4))
# 训练全部 100 轮结束，拿 x=4 做预测。理想结果 8.0。
# 因为只有 100 轮迭代，浮点数逼近，实际输出 7.99977...，非常接近 8。
print(f'predict (after training) 4 {forward(4):.2f}')

plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()