# pip install numpy pandas matplotlib
# 1. 导入必备库
import pandas as pd  # 处理表格数据的核心库（读取CSV、数据清洗、计算相关系数）
import seaborn as sns  # 基于matplotlib的可视化库，画热力图超方便
import matplotlib.pyplot as plt  # 基础可视化库，控制图表显示

# 2. 读取广告投放数据集（CSV格式）
# 数据集说明：advertising.csv通常包含「不同渠道广告投放费用」和「对应销售额」，
# 比如TV、Radio、Newspaper列是广告费用，Sales列是销售额
df = pd.read_csv("data/advertising.csv")
# 打印数据形状（行数=样本数，列数=特征数）
print("第1次打印数据形状: "+str(df.shape))


# 3. 数据清洗：处理冗余列和缺失值（实战中第一步必做）
# 3.1 删除第一列（通常是无意义的索引列/序号列，对分析没用，有点类似我们的主键OID）
# axis=1表示删除列，inplace=True表示直接修改原DataFrame，不生成新数据
df.drop(df.columns[0], axis=1, inplace=True)
# 3.2 删除包含缺失值的行（缺失值会影响相关系数计算，简单清洗直接删）
df.dropna(inplace=True)
# 再次打印形状，对比清洗前后的样本数/列数变化
print("第2次打印数据形状: "+str(df.shape))

# 4. 提取特征（X）和目标变量（y）—— 机器学习标准操作
# X：特征矩阵（广告投放费用，比如TV、Radio、Newspaper），删除目标列Sales
X = df.drop("Sales", axis=1)
# y：目标变量（销售额），是我们要分析/预测的核心
y = df["Sales"]
# 打印特征矩阵和目标变量的形状，验证维度是否正确
print("预期：(样本数, 特征数)，比如(200,3)（200个样本，3个广告渠道）: ",X.shape)  # 预期：(样本数, 特征数)，比如(200,3)（200个样本，3个广告渠道）
print("预期：(样本数,)，比如(200,)（每个样本对应一个销售额）: ",y.shape)  # 预期：(样本数,)，比如(200,)（每个样本对应一个销售额）
print()

# 5. 计算「每个特征与目标变量（销售额）的皮尔逊相关系数」
# corrwith(y)：专门计算DataFrame中每一列与指定Series（y）的相关系数，结果是一维Series
# 核心目的：精准计算 “每个广告渠道费用” 和 “销售额” 的相关系数；
corr = X.corrwith(y) # 上课时候给同学们查看下corrwith方法源码13297行
# 打印特征-销售额的相关系数，比如TV和Sales的r值可能≈0.8（强正相关），Newspaper可能≈0.3（几乎无关）
print("每个特征与目标变量（销售额）的皮尔逊相关系数： 投电视广告最有效，广播还行，报纸广告几乎没用。\n",corr)
print()

# 6. 计算「全量相关系数矩阵」（包含所有列，包括Sales）
# corr()：默认计算皮尔逊相关系数，生成对称矩阵（行和列都是所有特征+目标变量）
# 核心目的：不仅看特征和销售额的相关性，还看特征之间的相关性（比如Radio和TV是否冗余）
corr_matrix = df.corr()
# 打印相关系数矩阵，直观查看所有变量间的线性关系
print("打印相关系数矩阵，直观查看所有变量间的线性关系\n")
print("每一个数字 = 两个变量之间的 “相关程度”,范围永远在 -1 ~ 1 之间\n")
print(corr_matrix)


# # 7. 可视化相关系数矩阵（热力图）
# # 7.1 画热力图：annot=True显示数值，fmt='.2f'保留2位小数，cmap='coolwarm'红蓝配色（红:正相关，蓝:负相关）
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
# 7.2 设置图表标题
plt.title("Feature Correlation Matrix")
# 7.3 显示图表
plt.show()