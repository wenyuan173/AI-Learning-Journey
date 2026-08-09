import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
data = pd.read_csv('02_machine_learning/data/advertising.csv')

# 2. 数据预处理
data.drop(data.columns[0],axis=1, inplace=True) 
# 删除第一列，inplace表示直接改原始变量，不加就是重新创建一个
data.dropna(inplace=True) # 删除缺失值

# 3.提取特征和标签 + 划分数据集
x= data.drop('Sales',axis=1) # 自变量（抛去sales列作为输入）
y= data['Sales'] # 因变量

train_x,test_x,train_y,test_y = train_test_split(data[['TV','Radio','Newspaper']],data['Sales'],test_size=0.2,random_state=1)

# 4. 特征工程
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x) # 训练集标准化（先取得均值和方差，再进行标准化）
test_x = scaler.transform(test_x) # 不加fit，因为测试集不参与训练，直接使用训练集的均值和方差进行标准化

# 5.选择模型
model_lr = LinearRegression() # 线性回归模型
model_sgd = SGDRegressor() # 随机梯度下降模型

# 6.训练模型
model_lr.fit(train_x,train_y) # 训练线性回归模型
model_sgd.fit(train_x,train_y) # 训练随机梯度下降模型

# 7.查看参数
print("线性回归模型参数：",model_lr.coef_,model_lr.intercept_)
print("随机梯度下降模型参数：",model_sgd.coef_,model_sgd.intercept_)

# 8.查看损失
print("线性回归模型损失：",mean_squared_error(test_y, model_lr.predict(test_x)))
print("随机梯度下降模型损失：",mean_squared_error(test_y, model_sgd.predict(test_x)))

# 9.R²评价指标
r2_score_lr = model_lr.score(test_x,test_y) # 线性回归模型R²
r2_score_sgd = model_sgd.score(test_x,test_y) # 随机梯度下降模型R²
print("线性回归模型R²：",r2_score_lr)
print("随机梯度下降模型R²：",r2_score_sgd)

# 10.预测
print("线性回归模型预测结果：",model_lr.predict(test_x[5].reshape(1,-1)),test_y.values[5]) # 预测第6个样本的结果
print("随机梯度下降模型预测结果：",model_sgd.predict(test_x[5].reshape(1,-1)),test_y.values[5]) # 预测第6个样本的结果

