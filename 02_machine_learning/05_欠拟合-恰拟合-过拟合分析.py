# 导入必备库
import numpy as np  # 生成数据、矩阵运算
import matplotlib.pyplot as plt  # 画图（可视化拟合效果）
from sklearn.model_selection import train_test_split  # 划分训练/测试集
from sklearn.linear_model import LinearRegression   # 线性回归模型（拟合多项式的核心）
from sklearn.metrics import mean_squared_error  # 均方误差（衡量拟合好坏）
from sklearn.preprocessing import PolynomialFeatures    # 多项式特征转换（把x变成x²、x³...）

# 设置画图的中文/负号显示（避免中文乱码，类似java程序员的utf-8处理）
plt.rcParams['font.sans-serif'] = ['KaiTi'] # windows 系统
#plt.rcParams["font.sans-serif"] = ["SongTi SC"]  # macos 系统
plt.rcParams['axes.unicode_minus'] = False

'''
代码作用，解读前给同学们说明背景和大纲：
用多项式拟合 sin (x)，对比欠拟合 / 恰好拟合 / 过拟合；
泰勒展开是 “理论指导”：sin (x) 在 [-3,3] 用 5~7 次多项式就够，阶数太少欠拟合，太多过拟合；
关键结论：
欠拟合：模型 “学不会” 规律（阶数太低）；
过拟合：模型 “学太细”，把噪声当规律（阶数太高）；
恰好拟合：阶数匹配泰勒展开的有效阶数，既学规律又忽略噪声。
一句话记住：泰勒展开定阶数，不多不少才合适；少了欠拟合，多了过拟合！
'''

# ===================== 1. 生成数据（模拟带噪声的sin(x)）=====================
# 生成300个样本点，模拟从其它系统对接获得
n = 300
# 生成x：在[-3,3]区间取300个均匀的点，reshape(-1,1)把一维数组变成二维
# （sklearn 要求输入必须是 “样本数 × 特征数” 的二维数组）
X = np.linspace(-3, 3, n).reshape(-1, 1)
# 基于X生成y：sin(x) + 随机噪声（模拟真实数据的误差），reshape保证和X维度一致
# 为什么加噪声？真实世界的数据不会完美贴合sin(x)，噪声更贴近实际
y = np.sin(X) + np.random.uniform(-0.5, 0.5, n).reshape(-1, 1)
# 打印维度，验证是300行1列（300个样本，1个特征）
print(X.shape)  # 输出 (300, 1)
print(y.shape)  # 输出 (300, 1)

# 创建1行3列的散点图，用来对比3种拟合效果（欠拟合/恰拟合/过拟合）
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
# 三个子图都先画出原始数据的散点（黄色y=yellow），方便对比拟合曲线
ax[0].scatter(X, y, color='y')
ax[1].scatter(X, y, color='y')
ax[2].scatter(X, y, color='y')

# ===================== 2. 划分数据集（训练集80%，测试集20%）=====================
# 训练集是模型 “见过的题”，测试集是 “新题”，只有测试误差小，模型才是真的好。
# test_size=0.2设定20%的测试集，剩下80%训练集
# random_state=42：固定 “随机拆分” 的结果，让每次运行代码拆分的数据集都一样（42 是行业常用的 “幸运数字”，设其他数比如 10、20 也可以）。
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== 3. 定义线性回归模型（所有拟合都用这个模型）=====================
# 注意：LinearRegression不仅能拟合直线，配合多项式特征能拟合任意多项式！
model = LinearRegression()

# ===================== 4. 模型训练和测试（对比3种拟合情况）=====================
# 4.1 欠拟合——用直线（1次多项式）拟合sin(x)
# 本质：只用到泰勒展开的一阶项（P1(x)=x），阶数太低，拟合不了曲线
# 训练：用训练集的x（直线特征）拟合y
model.fit(x_train, y_train)
# 测试：用训练好的模型预测测试集的y
y_pred = model.predict(x_test)

# 计算误差：均方误差（MSE）越小，拟合越好
# 训练误差：模型在训练集上的表现，测试误差：模型在新数据上的表现
test_loss = mean_squared_error(y_test,y_pred)
train_loss = mean_squared_error(y_train,model.predict(x_train))

# 画出拟合直线（红色），对比黄色散点
ax[0].plot(X, model.predict(X), color='r')
# 在图上标注误差值，直观看到拟合好坏
ax[0].text(-3, 1, f"测试误差：{test_loss:.4f}")
ax[0].text(-3, 1.3, f"训练误差：{train_loss:.4f}")
ax[0].set_title("欠拟合（1次多项式/直线）")  # 加标题，方便识别

# 4.2 恰好拟合——用5次多项式拟合sin(x)
# 本质：对应泰勒展开的5阶项（P5(x)=x - x³/6 + x⁵/120），阶数刚好匹配sin(x)的曲线规律
# 特征工程：把x转换成5次多项式特征（x, x², x³, x⁴, x⁵）
poly = PolynomialFeatures(degree=5)  # degree=5：5次多项式
x_train2 = poly.fit_transform(x_train)  # 训练集(fit_transform)特征转换
x_test2 = poly.transform(x_test)        # 测试集(transform)特征转换（只用fit过的poly，避免数据泄露）
# 训练：用5次多项式特征拟合y
model.fit(x_train2, y_train)
# 测试：预测测试集
y_pred = model.predict(x_test2)
# 计算误差
test_loss = mean_squared_error(y_test, y_pred)
train_loss = mean_squared_error(y_train, model.predict(x_train2))
# 画出5次多项式拟合曲线（红色）
ax[1].plot(X, model.predict( poly.transform(X) ), color='r')
# 标注误差
ax[1].text(-3, 1, f"测试误差：{test_loss:.4f}")
ax[1].text(-3, 1.3, f"训练误差：{train_loss:.4f}")
ax[1].set_title("恰好拟合（5次多项式）")


# 4.3 过拟合——用20次多项式拟合sin(x)
# 本质：多项式阶数远高于泰勒展开的有效阶数，模型不仅学了sin(x)的规律，还学了噪声的规律
# 特征工程：把x转换成20次多项式特征（x到x²⁰）
poly = PolynomialFeatures(degree=20)
x_train3 = poly.fit_transform(x_train)
x_test3 = poly.transform(x_test)

# 训练：用20次多项式特征拟合y
model.fit(x_train3, y_train)

# 测试：预测测试集
y_pred = model.predict(x_test3)

# 计算误差：训练误差极小（几乎为0），但测试误差很大（模型记了噪声，没学规律）
test_loss = mean_squared_error(y_test, y_pred)
train_loss = mean_squared_error(y_train, model.predict(x_train3))

# 画出20次多项式拟合曲线（红色，会疯狂波动）
ax[2].plot(X, model.predict( poly.transform(X) ), color='r')
# 标注误差
ax[2].text(-3, 1, f"测试误差：{test_loss:.4f}")
ax[2].text(-3, 1.3, f"训练误差：{train_loss:.4f}")
ax[2].set_title("过拟合（20次多项式）")

# 显示所有图
plt.show()