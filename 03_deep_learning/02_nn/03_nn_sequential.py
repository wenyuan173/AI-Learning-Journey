import torch
from torch import nn
from torchsummary import summary

# 1.定义输入数据
x = torch.randn(10,3)

# 2.创建模型
model = nn.Sequential(
    nn.Linear(3,4),
    nn.Tanh(),
    nn.Linear(4,4),
    nn.ReLU(),
    nn.Linear(4,2),
    nn.Softmax(dim=-1),
)

# 保存模型参数
torch.save(model.state_dict(), 'model.pth')

state_dict = torch.load('model.pth')
model.load_state_dict(state_dict) # 加载参数

# 3.向前传播
y = model(x)
print(y)

summary(model, input_size=(3,),device='cpu',batch_size=20)
