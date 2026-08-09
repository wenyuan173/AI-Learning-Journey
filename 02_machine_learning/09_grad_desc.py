import numpy as np

x =np.array([[5],[8],[10],[12],[15],[3],[7],[9],[11],[14]])
# 自变量，每周学习时长
x = np.hstack((np.ones((10,1)),x)) # ones表示添加一列1，表示截距项
print(x)

y = np.array([[50],[60],[65],[70],[80],[40],[55],[62],[68],[75]])
# 因变量，数学考试成绩

n=x.shape[0] # 样本数量
# 定义损失函数(均方误差)
def loss(beta):
    return np.sum(x@beta-y)**2/n

def gradient(beta):
    return (x.T@(x@beta-y))/n*2

# 初始参数和学习率
alpha = 0.0097
beta = np.array([[1], [1]]) # 初始参数（随便定的起点）
iter = 1000 # 迭代次数

beta0 = []
beta1=[]
for i in range(iter):
    beta0.append(beta[0][0])
    beta1.append(beta[1][0])
    # 计算梯度和更新参数
    beta = beta - alpha * gradient(beta)
    print(f"迭代 {i}: beta= {beta}")

import matplotlib.pyplot as plt
plt.plot(beta0,beta1)
plt.show()