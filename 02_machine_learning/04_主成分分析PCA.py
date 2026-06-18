 # 1. 导入必备库
import numpy as np  # 生成数据、矩阵运算的核心库
import matplotlib.pyplot as plt  # 画图（支持3D/2D可视化）
from sklearn.decomposition import PCA  # 导入PCA降维工具

'''
PCA 降维的经典模板：构造有规律的高维数据 → PCA 降维 → 3D/2D 可视化对比
PCA 通过找 “方差最大的方向”，把高维数据投影到低维，既减少特征数，又保留核心信息；
3D 转 2D 的对比图，一眼看懂 “PCA 降维没丢关键信息”
'''


# 2. 设定样本数量：生成1000个样本
n = 1000

# 3.故意构造 “有规律的 3D 数据”，让 PCA 能精准找到主成分
# 3.1 PC1：主成分1，标准正态分布（均值0，标准差1），方差=1 → 信息量最大,是数据的 “核心信息”（比如房价预测里的 “面积”）
pc1 = np.random.normal(0, 1, n)
# 3.2 PC2：主成分2，正态分布（均值0，标准差0.2），方差=0.04 → 是次要信息（比如 “房间数”）
pc2 = np.random.normal(0, 0.2, n)
# 3.3 生成噪声：标准差0.1，方差=0.01 → 几乎无信息量，是无用的噪声（比如 “房主年龄”）。
noise = np.random.normal(0, 0.1, n)

# 4. 构造3维特征（把主成分和噪声组合成“带冗余/噪声的高维数据”）
# Feature1 = PC1 - PC2 → 由核心主成分构成，含有效信息
feature1 = pc1 - pc2
# Feature2 = PC1 + PC2 → 由核心主成分构成，和Feature1高度相关（冗余）
feature2 = pc1 + pc2
# Feature3 = PC2 + 噪声 → 含少量有效信息+噪声，信息量最少
feature3 = pc2 + noise
# 合并成3维特征矩阵：vstack垂直堆叠后转置，得到 (1000,3)（1000样本，3特征）
X = np.vstack([feature1, feature2, feature3]).T
# 打印原始数据形状，验证是3维
print(X.shape)  # 输出 (1000, 3)

# 5. 初始化PCA模型：n_components=2 → 把3维数据降到2维
pca = PCA(n_components=2)

# 6. 核心操作：训练PCA+降维
# fit：让PCA学习3维数据的主成分方向（找方差最大的2个方向）
# transform：把3维数据投影到这2个主成分方向，得到2维数据
X_pca = pca.fit_transform(X)
# 打印降维后形状，验证是2维
print(X_pca.shape)  # 输出 (1000, 2)

# 7. 可视化：画“降维前3D”和“降维后2D”对比图
# 7.1 创建画布：尺寸12×4，足够放下两个子图
fig = plt.figure(figsize=(12, 4))

# 7.2 第一个子图：降维前的3D散点图
ax1 = fig.add_subplot(121, projection='3d')  # 121=1行2列第1个图，projection='3d'启用3D
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c="g")  # 画3D散点，c="g"设为绿色
ax1.set_title('Before PCA(3D)')  # 标题
ax1.set_xlabel('Feature1')  # x轴：原始特征1
ax1.set_ylabel('Feature2')  # y轴：原始特征2
ax1.set_zlabel('Feature3')  # z轴：原始特征3

# 7.3 第二个子图：降维后的2D散点图
ax2 = fig.add_subplot(122)  # 122=1行2列第2个图，默认2D
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c="g")  # 画2D散点，颜色和3D一致
ax2.set_title('After PCA(2D)')  # 标题
ax2.set_xlabel('PC1')  # x轴：主成分1（方差最大的方向）
ax2.set_ylabel('PC2')  # y轴：主成分2（方差次大的方向）

# 7.4 显示图表
plt.show()