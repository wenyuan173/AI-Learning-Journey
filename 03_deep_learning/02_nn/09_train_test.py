import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1.准备数据：y=wx+b
x = torch.randn(100,1)
y = x*2.5 + 5.0 + torch.randn(100,1)*0.2

# 2.构造数据集和数据加载器
ds = TensorDataset(x,y)
# loader = DataLoader(ds, batch_size=batch_size,shuffle=True,drop_last=False)

# 3.定义一个模型：只有一个线形层
model = nn.Linear(1,1)

# 4.定义损失函数
loss_fn = nn.MSELoss()

# 5.训练模型
# for epoch in range(epochs):
#     for  input,target loader:
        # 5.1 一次前向传播
        # y_pred = model(input)

        #5.2 计算损失
        # loss = loss_fn(y_pred,target)

        #5.3 反向传播计算梯度
        # loss.backward()

        # 5.4 更新参数
