# 导入numpy库：用于数值计算（生成数据、矩阵操作、数组维度调整等核心功能）
import numpy as np
# 导入matplotlib.pyplot库：用于绘制数据可视化图表（散点图、拟合曲线、参数直方图）
import matplotlib.pyplot as plt
# 从sklearn模型选择模块导入train_test_split：用于划分训练集和测试集
from sklearn.model_selection import train_test_split
# 从sklearn线性模型模块导入三种回归模型：
# LinearRegression（普通线性回归）、Lasso（L1正则化回归）、Ridge（L2正则化回归）
from sklearn.linear_model import LinearRegression, Lasso, Ridge
# 从sklearn评估指标模块导入均方误差：用于量化模型预测值与真实值的误差
from sklearn.metrics import mean_squared_error
# 从sklearn预处理模块导入多项式特征转换：将一维特征转为高次多项式特征，实现非线性拟合
from sklearn.preprocessing import PolynomialFeatures

'''
程序核心功能说明：
以在x∈[-3,3]区间用20次多项式拟合sin(x)曲线为例，
通过 20 次多项式拟合 sin (x)，对比「无正则化」「L1 正则化」「L2 正则化」的效果，
验证正则化对过拟合的抑制作用；对比三种回归方式的效果：
① 不使用正则化的普通线性回归（易过拟合,训练误差小、测试误差大）
② 使用L1正则化的Lasso回归,L1 正则化通过参数稀疏化实现特征选择，缓解过拟合（参数稀疏化，特征选择）
③ 使用L2正则化的Ridge回归,L2 正则化通过参数缩小实现模型平滑，提升泛化能力（参数整体缩小，防止过拟合）
通过可视化拟合曲线和参数分布，直观展示正则化的作用
完整展示了 “数据生成→特征工程→模型训练→评估可视化” 的机器学习流程，是理解正则化作用的经典案例
'''

# 设置matplotlib绘图参数：解决中文显示乱码和负号异常问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定中文显示字体为楷体
plt.rcParams['axes.unicode_minus'] = False   # 解决坐标轴负号显示为方块的问题

# ===================== 步骤1：生成模拟数据集 =====================
n = 300  # 定义样本数量为300个
# 生成x轴数据：在[-3, 3]区间生成300个均匀分布的数值，reshape(-1,1)转为列向量（sklearn要求输入为二维数组）
X = np.linspace(-3, 3, n).reshape(-1, 1)
# 生成y轴数据：sin(X)基础上添加[-0.5, 0.5]的随机噪声，模拟真实场景的带噪数据
y = np.sin(X) + np.random.uniform(-0.5, 0.5, n).reshape(-1, 1)

# 调试用代码（注释掉，上课给同学们提一嘴即可）：查看X和y的维度，确认数据格式符合sklearn要求
# print(X.shape)  # 输出(300, 1)，300行1列
# print(y.shape)  # 输出(300, 1)，300行1列

# ===================== 步骤2：创建画布并绘制原始数据散点图 =====================
# 创建2行3列的共计6个子图布局，画布尺寸为15*8英寸（宽*高）
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
# 第一行第一个子图：绘制原始数据散点图，颜色为黄色
ax[0,0].scatter(X, y, color='y')
# 第一行第二个子图：绘制原始数据散点图，颜色为黄色（用于L1正则化拟合）
ax[0,1].scatter(X, y, color='y')
# 第一行第三个子图：绘制原始数据散点图，颜色为黄色（用于L2正则化拟合）
ax[0,2].scatter(X, y, color='y')

# ===================== 步骤3：划分训练集和测试集 =====================
# 训练集是模型 “见过的题”，测试集是 “新题”，只有测试误差小，模型才是真的好。
# test_size=0.2设定20%的测试集，剩下80%训练集
# random_state=42：固定 “随机拆分” 的结果，
# 让每次运行代码拆分的数据集都一样（42 是行业常用的 “幸运数字”，设其他数比如 10、20 也可以）。
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== 步骤4：特征工程 - 多项式特征转换 =====================
# 过拟合——用20次多项式拟合sin(x)，设置多项式次数为20（高次多项式易过拟合，便于展示正则化效果）
# 等价于：y=w0+w1x+w2x^2+w3x^3+...+w20x^20
poly = PolynomialFeatures(degree=20)
# 对训练集特征进行多项式转换（fit_transform：先拟合数据分布，再转换，避免数据泄露）
x_train = poly.fit_transform(x_train)
# 对测试集特征进行多项式转换（仅transform：使用训练集的拟合规则）
x_test = poly.transform(x_test)

# ===================== 步骤5：模型训练与评估 - 无正则化线性回归 =====================
# 初始化普通线性回归模型,无正则化约束：系数波动极大，甚至出现超大值（过拟合根源）；
model = LinearRegression()
# 模型训练：用训练集数据拟合模型，学习多项式特征与y的映射关系
model.fit(x_train, y_train)
# 打印模型系数：20次多项式对应21个系数（x^0到x^20），查看系数分布
print(model.coef_)
# 模型测试：用训练好的模型预测测试集的y值
y_pred = model.predict(x_test)
# 计算误差：评估模型拟合效果
test_loss = mean_squared_error(y_test, y_pred)  # 测试集均方误差（反映泛化能力）
train_loss = mean_squared_error(y_train, model.predict(x_train))  # 训练集均方误差（反映拟合能力）
# 绘制拟合曲线：在第一行第一个子图绘制模型对完整X的预测曲线（红色）
ax[0,0].plot(X, model.predict( poly.transform(X) ), color='r')
# 在子图上标注测试误差（保留4位小数） 输出：测试误差：0.1235（保留4位小数，第5位四舍五入）
ax[0,0].text(-3, 1, f"测试误差：{test_loss:.4f}")
# 在子图上标注训练误差（保留4位小数）
ax[0,0].text(-3, 1.3, f"训练误差：{train_loss:.4f}")
# 绘制参数直方图：在第二行第一个子图展示21个系数的分布
ax[1,0].bar( np.arange(21), model.coef_.reshape(-1) )


# ===================== 步骤6：模型训练与评估 - L1正则化（Lasso回归） =====================
# L1 正则化：大量系数被压缩为 0（稀疏性），仅保留关键特征的参数；
# 初始化Lasso回归模型，设置正则化强度alpha=0.01（alpha越大，正则化约束越强）
# 这个就是超参数，属于人为干预，设置着看
model = Lasso(alpha=0.01)
# 模型训练：用训练集拟合Lasso模型（自动惩罚参数绝对值，实现稀疏化）
model.fit(x_train, y_train)
# 模型测试：预测测试集y值
y_pred = model.predict(x_test)
# 计算误差：测试集和训练集均方误差
test_loss = mean_squared_error(y_test, y_pred)
train_loss = mean_squared_error(y_train, model.predict(x_train))
# 绘制拟合曲线：第一行第二个子图（红色）
ax[0,1].plot(X, model.predict( poly.transform(X) ), color='r')
# 标注误差
ax[0,1].text(-3, 1, f"测试误差：{test_loss:.4f}")
ax[0,1].text(-3, 1.3, f"训练误差：{train_loss:.4f}")
# 绘制参数直方图：第二行第二个子图（可看到大量系数为0，体现稀疏性）
ax[1,1].bar( np.arange(21), model.coef_.reshape(-1) )


# ===================== 步骤7：模型训练与评估 - L2正则化（Ridge回归） =====================
# L2 正则化：所有系数均不为 0，但整体值显著缩小（平滑性）。
# 初始化Ridge回归模型，设置正则化强度alpha=0.5
model = Ridge(alpha=0.5)
# 模型训练：用训练集拟合Ridge模型（惩罚参数平方，让参数整体缩小）
model.fit(x_train, y_train)
# 打印Ridge模型系数
print(model.coef_)
# 模型测试：预测测试集y值
y_pred = model.predict(x_test)
# 计算误差：测试集和训练集均方误差
test_loss = mean_squared_error(y_test, y_pred)
train_loss = mean_squared_error(y_train, model.predict(x_train))
# 绘制拟合曲线：第一行第三个子图（红色）
ax[0,2].plot(X, model.predict( poly.transform(X) ), color='r')
# 标注误差
ax[0,2].text(-3, 1, f"测试误差：{test_loss:.4f}")
ax[0,2].text(-3, 1.3, f"训练误差：{train_loss:.4f}")
# 绘制参数直方图：第二行第三个子图（系数均不为0，但整体值更小）
ax[1,2].bar( np.arange(21), model.coef_.reshape(-1) )


# 显示所有绘制的图表
plt.show()



# 后台警告概述：
# D:\devSoft\PyCharm2025.2.5\workspace01\MachineLearningV1\.venv\Lib\site-packages
# \sklearn\linear_model\_coordinate_descent.py:697:

#  ConvergenceWarning: Objective did not converge.
#  You might want to increase the number of iterations,
#  check the scale of the features or consider increasing regularisation.
#  Duality gap: 1.178e+01, tolerance: 1.405e-02
#   model = cd_fast.enet_coordinate_descent(
# 这个 ConvergenceWarning（收敛警告）是Lasso 回归训练时抛出的，
# 故意的给同学们看到，不影响代码运行，但模型可能没找到最优解，Lasso 的稀疏性效果会打折扣
#
# 核心意思是：模型在默认的迭代次数内，还没找到最优解就停止了，核心原因是高次多项式特征尺度差异大
# 1 代码里用了20 次多项式特征拟合 sin (x)，20 次多项式的特征值范围天差地别
# （比如 x=3 时，x^1=3，x^20=3^20≈35 亿），特征尺度不统一会让 Lasso 的优化过程 “走得很慢”，默认迭代次数不够用，
# 2 设置的alpha=0.01太小，正则化约束弱，模型想拟合所有特征，参数优化过程难以稳定；