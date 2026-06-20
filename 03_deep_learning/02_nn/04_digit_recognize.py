import torch
import torch.nn as nn
from common.load_data import get_digital_data
# 1.读取数据
x_train,x_test,y_train,y_test = get_digital_data()

# 2.定义模型
model= nn.Sequential(
    nn.Linear(784,50),
    nn.ReLU(),

    nn.Linear(50,100),
    nn.ReLU(),

    nn.Linear(100,10),
    # nn.softmax(dim=-1)   不需要加，它和损失整合在一起
)

# 3,加载状态字典
state_dict= torch.load("./data/nn_example.pt")
model.load_state_dict(state_dict)

# 4.预测
y_pred = model(x_test)

# 5.计算准确率
# 5.1 预测的类别
y_class = torch.argmax(y_pred,dim=-1)
# 5.2 计算预测正确的个数
acc_cnt =(y_class == y_test).sum().item()

acc= acc_cnt/len(y_test)

print(f"Accuracy: {acc:.4f}%")