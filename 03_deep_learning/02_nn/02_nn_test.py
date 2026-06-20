import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义三个全连接层
        self.linear1= nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 4)
        self.out= nn.Linear(4, 2)
        pass

    def forward(self, x):
        # 逐层计算向前传播
        x= self.linear1 (x)
        x =torch.tanh(x)

        x = self.linear2 (x)
        x = torch.relu(x)

        x = self.out(x)
        y=torch.softmax(x,dim=1)
        return y

if __name__ == '__main__':
    # 定义输入数据
    x = torch.randn(10,3) #10条数据，每条数据3特征

    # 创建模型
    model = MyModule()

    # 向前传播计算
    y = model(x)
    print(y)

    print(model.linear1.weight)
    print(model.linear1.bias)

    # 遍历所有参数（没有名字）
    for para in model.parameters():
        print(para)

    # 遍历所有参数（有名字）
    for name,para in model.named_parameters():
        print(name,para)

    # 查看状态字典
    dict = model.state_dict()
    print(dict)

    # 模型结构：torchsummary
    from torchsummary import summary
    summary(model, input_size=(3,),batch_size=10, device='cpu')






