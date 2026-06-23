import torch
import torch.nn as nn

# 1,定义数据
x = torch.tensor([10.0])
y = torch.tensor([3.0])

# 2.定义参数和偏执
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 4.定义损失函数
loss_fn = nn.MSELoss()

# 5.正向传播，计算损失
z = w * x + b
loss = loss_fn(z, y)

# 6.反向传播：必须从损失开始（是一个标量）
loss.retain_grad() #非叶子节点，不清梯度
z.retain_grad()
loss.backward()

# 7.验证各个节点梯度
print(loss.grad)
print(z.grad)
print(b.grad)
print(w.grad)
print(x.grad)

print("-------------------")
z = w * x + b
loss = loss_fn(z, y)
loss.backward()
print(b.grad)
print(w.grad)
