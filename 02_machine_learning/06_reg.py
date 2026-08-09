import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge
# 线性回归模型
from sklearn.preprocessing import PolynomialFeatures
# 多项式特征
from sklearn.model_selection import train_test_split
# 测试集和训练集的划分
from sklearn.metrics import mean_squared_error
# 均方误差损失（自己看的，其实是看平方误差）

# 1.读取数据（生成数据）均匀间隔的数值
x = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)
y = np.sin(x)+ np.random.uniform(-0.6, 0.6, 300).reshape(-1, 1)

fig, axs = plt.subplots(2,3,figsize=(15,4))
axs[0,0].scatter(x, y)
axs[0,1].scatter(x, y)
axs[0,2].scatter(x, y)

# 2.数据清洗

# 3.特征工程 + 划分数据集
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# 4.选择模型
model = LinearRegression() # 线性回归模型
# 5.训练模型

poly20 = PolynomialFeatures(degree=20)
train_x20 = poly20.fit_transform(train_x)
test_x20 = poly20.transform(test_x)
# 5.训练模型
model.fit(train_x20, train_y)
print(model.coef_, model.intercept_) # 模型的参数和截距

# 6.评估
train_lose = mean_squared_error(train_y, model.predict(train_x20))
test_lose = mean_squared_error(test_y, model.predict(test_x20))

# 7.绘制曲线
axs[0,0].plot(x,model.predict(poly20.transform(x)),color='red')
axs[0,0].text(-3,1.3,f"Train Lose:{train_lose:.4f}")
axs[0,0].text(-3,1,f"Test Lose:{test_lose:.4f}")
axs[1,0].bar(np.arange(len(model.coef_.reshape(-1))),model.coef_.reshape(-1))

# L1正则化：alpha正则化强度
model = Lasso(alpha=0.01)
model.fit(train_x20, train_y)
train_lose = mean_squared_error(train_y, model.predict(train_x20))
test_lose = mean_squared_error(test_y, model.predict(test_x20))
print("Lasso:",train_lose, test_lose) # 模型的参数和截距
axs[0,1].plot(x,model.predict(poly20.transform(x)),color='red')
axs[1,1].bar(np.arange(len(model.coef_.reshape(-1))),model.coef_.reshape(-1))

# L2正则化：alpha正则化强度
model = Ridge(alpha=1)
model.fit(train_x20, train_y)
train_lose = mean_squared_error(train_y, model.predict(train_x20))
test_lose = mean_squared_error(test_y, model.predict(test_x20))
print("Ridge:",train_lose, test_lose) # 模型的参数和截距
axs[0,2].plot(x,model.predict(poly20.transform(x)),color='red')
axs[1,2].bar(np.arange(len(model.coef_.reshape(-1))),model.coef_.reshape(-1))

plt.show()