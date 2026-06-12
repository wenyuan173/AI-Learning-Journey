# =============================================================================
# 机器学习 · 过滤法特征选择 完整实战案例
# =============================================================================
# 【核心原理】
#   过滤法（Filter Method）是一种独立于机器学习模型的特征选择方法。
#   它通过统计指标来评估每个特征的"质量"，再根据阈值过滤掉低质量特征。
#   优点：计算速度快，不依赖具体模型，通用性强。
#
# 【本文覆盖的四种过滤方法】
#   1. 方差过滤（VarianceThreshold）  → 删除变化不大的特征（近似常数）
#   2. 卡方检验过滤（chi2）            → 适用于分类任务 + 非负特征
#   3. F检验过滤（f_classif）          → 适用于分类任务，检验线性关系
#   4. 互信息过滤（mutual_info_classif）→ 能捕捉非线性关系，更强大
# =============================================================================

# ─────────────────────────── 导入所需库 ────────────────────────────────────
import numpy as np                               # 数值计算库，处理数组和矩阵
import matplotlib.pyplot as plt                  # 绘图库，用于可视化结果
import matplotlib                                # matplotlib 主模块，用于全局配置
import warnings                                  # 警告控制模块

from sklearn.datasets import load_breast_cancer  # 乳腺癌数据集（经典二分类）
from sklearn.feature_selection import (
    VarianceThreshold,                           # 方差过滤器
    SelectKBest,                                 # 选取得分最高的 K 个特征
    chi2,                                        # 卡方检验统计量
    f_classif,                                   # F 检验统计量（方差分析）
    mutual_info_classif,                         # 互信息统计量
)
from sklearn.preprocessing import MinMaxScaler   # 最小-最大归一化，将数据缩放到 [0,1]
from sklearn.linear_model import LogisticRegression  # 逻辑回归分类器
from sklearn.model_selection import cross_val_score  # K折交叉验证

# ─────────────────────── 全局绘图配置 ──────────────────────────────────────
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# ↑ 设置中文字体，依次尝试，保证中文标签不乱码
matplotlib.rcParams['axes.unicode_minus'] = False
# ↑ 解决坐标轴负号"-"显示为方块的问题
warnings.filterwarnings('ignore')               # 忽略版本兼容等无关警告


# =============================================================================
# 第一步：数据加载与预处理
# =============================================================================
def load_and_preprocess():
    """
    【函数作用】加载乳腺癌数据集并完成基础预处理。

    【数据集说明】
    - 乳腺癌数据集：569 个样本，30 个特征，2 个类别（良性/恶性）
    - 特征均为连续数值，来源于细胞核的数字化图像测量值
    - 目标：通过细胞特征判断肿瘤是否为恶性

    【返回值】
    - X_raw   : 原始特征矩阵（未归一化）
    - X_scaled: 归一化后的特征矩阵 [0,1]（卡方检验要求非负）
    - y       : 标签数组
    - feature_names: 特征名称列表
    """
    print("=" * 60)
    print("  第一步：数据加载与预处理")
    print("=" * 60)

    # ── 加载数据集 ──────────────────────────────────────────────────────────
    cancer = load_breast_cancer()                # 从 sklearn 内置数据集加载乳腺癌数据
    X_raw = cancer.data                          # 特征矩阵，形状 (569, 30)
    y = cancer.target                            # 标签向量，0=恶性, 1=良性，形状 (569,)
    feature_names = cancer.feature_names         # 30 个特征的英文名称列表

    # ── 打印数据基本信息 ────────────────────────────────────────────────────
    print(f"[OK] 数据集加载完成")
    print(f"   特征矩阵形状  X : {X_raw.shape}")        # 行=样本数，列=特征数
    print(f"   标签向量形状  y : {y.shape}")
    print(f"   特征数量      : {X_raw.shape[1]} 个")
    print(f"   样本数量      : {X_raw.shape[0]} 个")
    print(f"   类别分布      : 恶性(0)={np.sum(y==0)}, 良性(1)={np.sum(y==1)}")

    # ── 归一化处理 ──────────────────────────────────────────────────────────
    # 【原理】MinMaxScaler 将每列特征缩放到 [0,1]：
    #         X_scaled = (X - X_min) / (X_max - X_min)
    # 【目的】卡方检验要求输入为非负数；同时消除量纲差异，使各特征可比
    scaler = MinMaxScaler()                      # 创建归一化器对象
    X_scaled = scaler.fit_transform(X_raw)       # 拟合并变换：计算 min/max 后缩放

    print(f"\n   原始特征范围  : [{X_raw.min():.2f}, {X_raw.max():.2f}]")
    print(f"   归一化后范围  : [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    print(f"\n   前5个特征名称 : {list(feature_names[:5])}\n")

    return X_raw, X_scaled, y, feature_names    # 返回原始、归一化数据及标签、特征名


# =============================================================================
# 第二步：方差过滤（VarianceThreshold）
# =============================================================================
def variance_filter(X, feature_names, threshold=0.01):
    """
    【函数作用】使用方差阈值过滤掉变化极小的特征。

    【核心原理】
    - 一个特征的方差越小，说明它的值几乎不变（接近常数）。
    - 近似常数的特征对模型没有区分度，可以安全删除。
    - 公式：Var(X) = E[(X - μ)²]，μ 为特征均值
    - 若 Var(X) < threshold，则删除该特征。

    【参数】
    - X            : 特征矩阵（建议用原始数据，保留原始方差大小）
    - feature_names: 特征名称列表
    - threshold    : 方差阈值，低于此值的特征被删除（默认 0.01）

    【返回值】
    - X_filtered : 过滤后的特征矩阵
    - selector   : 拟合好的 VarianceThreshold 对象（可复用）
    - kept_names : 保留下来的特征名称列表
    """
    print("=" * 60)
    print("  第二步：方差过滤（VarianceThreshold）")
    print("=" * 60)
    print(f"  【原理】删除方差低于 {threshold} 的特征（变化太小 = 无区分度）\n")

    # ── 计算每个特征的方差（仅用于展示） ───────────────────────────────────
    variances = np.var(X, axis=0)               # axis=0 对每列（每个特征）计算方差
    print(f"  各特征方差（前10个）：")
    for i in range(10):                          # 打印前10个特征的方差
        tag = " ← 会被删除" if variances[i] < threshold else ""
        print(f"    {feature_names[i]:<35s}: {variances[i]:.4f}{tag}")

    # ── 创建并拟合方差过滤器 ────────────────────────────────────────────────
    selector = VarianceThreshold(threshold=threshold)   # 创建过滤器，设置阈值
    X_filtered = selector.fit_transform(X)              # 拟合+变换：自动删除低方差列

    # ── 获取被保留的特征名 ──────────────────────────────────────────────────
    kept_mask = selector.get_support()           # 返回布尔数组：True=保留, False=删除
    kept_names = feature_names[kept_mask]        # 用布尔索引取出保留的特征名

    # ── 打印过滤结果 ────────────────────────────────────────────────────────
    removed_count = X.shape[1] - X_filtered.shape[1]   # 计算被删除的特征数
    print(f"\n  过滤前特征数 : {X.shape[1]}")
    print(f"  过滤后特征数 : {X_filtered.shape[1]}")
    print(f"  删除特征数   : {removed_count}")
    print(f"  保留特征     : {list(kept_names)}\n")

    return X_filtered, selector, kept_names      # 返回过滤后数据、选择器、保留特征名


# =============================================================================
# 第三步：卡方检验过滤（chi2）
# =============================================================================
def chi2_filter(X_scaled, y, feature_names, k=20):
    """
    【函数作用】用卡方检验选出与标签相关性最强的 K 个特征。

    【核心原理】
    - 卡方检验（χ²检验）衡量特征与标签之间的"独立性"。
    - 若特征与标签完全独立（无关），则卡方值 ≈ 0。
    - 卡方值越大 → 特征与标签关联越强 → 越应该保留。
    - 公式：χ² = Σ (观测值 - 期望值)² / 期望值
    - 【限制】要求特征值必须非负（所以需要先归一化到 [0,1]）

    【参数】
    - X_scaled    : 归一化后的特征矩阵（非负，取值 [0,1]）
    - y           : 标签数组
    - feature_names: 特征名称列表
    - k           : 保留的特征数量（默认 20）

    【返回值】
    - X_chi2      : 卡方过滤后的特征矩阵
    - chi2_scores : 每个特征的卡方统计量（越大越好）
    - chi2_pvalues: 每个特征的 p 值（越小越显著）
    - kept_names  : 保留的特征名称
    """
    print("=" * 60)
    print("  第三步：卡方检验过滤（chi2）")
    print("=" * 60)
    print(f"  【原理】卡方值衡量特征与标签的相关性，选取最高的 {k} 个特征\n")

    # ── 计算所有特征的卡方统计量和 p 值 ────────────────────────────────────
    chi2_scores, chi2_pvalues = chi2(X_scaled, y)
    # chi2_scores  : 形状 (30,)，每个特征对应一个卡方值
    # chi2_pvalues : 形状 (30,)，对应 p 值（p < 0.05 认为显著相关）

    # ── 打印排名前10的特征（按卡方值降序） ──────────────────────────────────
    sorted_idx = np.argsort(chi2_scores)[::-1]  # argsort 升序，[::-1] 反转为降序
    print(f"  卡方值排名前10的特征（越高越重要）：")
    print(f"  {'排名':<5}{'特征名':<35}{'卡方值':>10}{'p值':>12}{'显著?':>8}")
    print(f"  {'-'*70}")
    for rank, idx in enumerate(sorted_idx[:10], 1):    # 遍历前10名
        sig = "★ 显著" if chi2_pvalues[idx] < 0.05 else "  不显著"
        print(f"  {rank:<5}{feature_names[idx]:<35}{chi2_scores[idx]:>10.2f}"
              f"{chi2_pvalues[idx]:>12.4f}{sig:>8}")

    # ── 用 SelectKBest 选出卡方值最高的 K 个特征 ────────────────────────────
    selector = SelectKBest(score_func=chi2, k=k)        # 创建选择器，指定统计量和 K
    X_chi2 = selector.fit_transform(X_scaled, y)        # 拟合+变换，返回选中的 K 列

    # ── 获取保留的特征名 ────────────────────────────────────────────────────
    kept_mask = selector.get_support()                   # 布尔掩码，True 表示保留
    kept_names = feature_names[kept_mask]                # 取出保留特征的名称

    print(f"\n  过滤前特征数 : {X_scaled.shape[1]}")
    print(f"  过滤后特征数 : {X_chi2.shape[1]}")
    print(f"  保留特征名   : {list(kept_names)}\n")

    return X_chi2, chi2_scores, chi2_pvalues, kept_names


# =============================================================================
# 第四步：F检验过滤（f_classif）
# =============================================================================
def ftest_filter(X, y, feature_names, k=20):
    """
    【函数作用】用 F 检验（方差分析 ANOVA）选出对标签有显著线性影响的特征。

    【核心原理】
    - F 检验（ANOVA）比较：「组间方差」vs「组内方差」
    - 若某特征在不同类别中的均值差异大（组间方差大），且同一类别内部方差小（组内方差小）
      → F 值大 → 该特征能很好地区分不同类别 → 应该保留
    - 公式：F = 组间均方 / 组内均方 = MS_between / MS_within
    - 【适用】连续特征 + 分类标签（比卡方检验更适合连续特征）

    【参数】
    - X           : 特征矩阵（可用原始数据，F检验无非负要求）
    - y           : 标签数组
    - feature_names: 特征名称列表
    - k           : 保留的特征数量（默认 20）

    【返回值】
    - X_ftest  : F检验过滤后的特征矩阵
    - f_scores : 每个特征的 F 统计量
    - f_pvalues: 每个特征的 p 值
    - kept_names: 保留的特征名称
    """
    print("=" * 60)
    print("  第四步：F检验过滤（f_classif / ANOVA）")
    print("=" * 60)
    print(f"  【原理】F值 = 组间方差 / 组内方差，F越大说明特征越能区分类别\n")

    # ── 计算 F 统计量和 p 值 ────────────────────────────────────────────────
    f_scores, f_pvalues = f_classif(X, y)
    # f_scores  : 形状 (30,)，每个特征的 F 值
    # f_pvalues : 形状 (30,)，对应的 p 值

    # ── 打印 F 检验排名前10 ──────────────────────────────────────────────────
    sorted_idx = np.argsort(f_scores)[::-1]     # 按 F 值从大到小排序
    print(f"  F值排名前10的特征（越高越重要）：")
    print(f"  {'排名':<5}{'特征名':<35}{'F值':>10}{'p值':>12}{'显著?':>8}")
    print(f"  {'-'*70}")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        sig = "★ 显著" if f_pvalues[idx] < 0.05 else "  不显著"
        print(f"  {rank:<5}{feature_names[idx]:<35}{f_scores[idx]:>10.2f}"
              f"{f_pvalues[idx]:>12.2e}{sig:>8}")

    # ── 选出 F 值最高的 K 个特征 ────────────────────────────────────────────
    selector = SelectKBest(score_func=f_classif, k=k)   # 用 F 检验评分，保留 k 个
    X_ftest = selector.fit_transform(X, y)               # 拟合并变换

    kept_mask = selector.get_support()                   # 获取保留特征的布尔掩码
    kept_names = feature_names[kept_mask]                # 对应特征名

    print(f"\n  过滤前特征数 : {X.shape[1]}")
    print(f"  过滤后特征数 : {X_ftest.shape[1]}")
    print(f"  保留特征名   : {list(kept_names)}\n")

    return X_ftest, f_scores, f_pvalues, kept_names


# =============================================================================
# 第五步：互信息过滤（mutual_info_classif）
# =============================================================================
def mutual_info_filter(X, y, feature_names, k=20):
    """
    【函数作用】用互信息选出与标签信息量最大的 K 个特征。

    【核心原理】
    - 互信息（Mutual Information）衡量两个变量共享的"信息量"。
    - I(X;Y) = H(Y) - H(Y|X)，其中 H 为信息熵
      - H(Y)    : 不知道 X 时，Y 的不确定性（熵）
      - H(Y|X)  : 已知 X 后，Y 的剩余不确定性（条件熵）
      - 差值越大 → X 能消除更多 Y 的不确定性 → X 与 Y 关联越强
    - 互信息 = 0 → 完全独立；互信息 > 0 → 有关联
    - 【优势】能捕捉非线性关系（卡方和 F 检验只能捕捉线性关系）

    【参数】
    - X            : 特征矩阵
    - y            : 标签数组
    - feature_names: 特征名称列表
    - k            : 保留的特征数量（默认 20）

    【返回值】
    - X_mi       : 互信息过滤后的特征矩阵
    - mi_scores  : 每个特征的互信息值
    - kept_names : 保留的特征名称
    """
    print("=" * 60)
    print("  第五步：互信息过滤（mutual_info_classif）")
    print("=" * 60)
    print(f"  【原理】互信息 = 特征能消除多少标签的不确定性，能捕捉非线性关系\n")

    # ── 计算每个特征与标签的互信息值 ────────────────────────────────────────
    # random_state=42 保证结果可复现（互信息估计有随机性）
    mi_scores = mutual_info_classif(X, y, random_state=42)
    # mi_scores : 形状 (30,)，每个特征的互信息值（越大越好，最小为0）

    # ── 打印互信息排名前10 ──────────────────────────────────────────────────
    sorted_idx = np.argsort(mi_scores)[::-1]    # 从大到小排序
    print(f"  互信息排名前10的特征（越高与标签关联越强）：")
    print(f"  {'排名':<5}{'特征名':<35}{'互信息值':>12}")
    print(f"  {'-'*55}")
    max_mi = mi_scores[sorted_idx[0]]           # 最大互信息值（用于计算进度条比例）
    for rank, idx in enumerate(sorted_idx[:10], 1):
        bar = "█" * int(mi_scores[idx] / max_mi * 20)  # 文字进度条，最长20格
        print(f"  {rank:<5}{feature_names[idx]:<35}{mi_scores[idx]:>8.4f}  {bar}")

    # ── 选出互信息最高的 K 个特征 ────────────────────────────────────────────
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_mi = selector.fit_transform(X, y)         # 拟合并变换

    kept_mask = selector.get_support()           # 布尔掩码
    kept_names = feature_names[kept_mask]        # 保留的特征名

    print(f"\n  过滤前特征数 : {X.shape[1]}")
    print(f"  过滤后特征数 : {X_mi.shape[1]}")
    print(f"  保留特征名   : {list(kept_names)}\n")

    return X_mi, mi_scores, kept_names


# =============================================================================
# 第六步：模型性能对比（验证特征选择效果）
# =============================================================================
def evaluate_model_performance(X_raw, X_chi2, X_ftest, X_mi, y):
    """
    【函数作用】对比"不做特征选择"vs"各种过滤法"的模型分类准确率。

    【方法】使用逻辑回归 + 5折交叉验证，以准确率衡量特征选择效果。
    - 交叉验证：将数据随机分成5份，轮流用4份训练、1份测试，取平均准确率
    - 这样能避免单次训练/测试划分的偶然性，评估更公平

    【参数】
    - X_raw  : 原始30个特征
    - X_chi2 : 卡方过滤后20个特征
    - X_ftest: F检验过滤后20个特征
    - X_mi   : 互信息过滤后20个特征
    - y      : 标签

    【返回值】
    - results: 字典，key=方法名, value=(均值准确率, 标准差)
    """
    print("=" * 60)
    print("  第六步：模型性能对比（逻辑回归 + 5折交叉验证）")
    print("=" * 60)
    print("  【方法】对比各过滤方法在逻辑回归上的5折交叉验证准确率\n")

    # ── 定义逻辑回归分类器 ──────────────────────────────────────────────────
    # max_iter=10000 增大迭代次数，保证模型收敛
    lr = LogisticRegression(max_iter=10000, random_state=42)

    # ── 准备各方案的数据（名称 + 矩阵） ─────────────────────────────────────
    datasets = {
        "原始特征（30个）": X_raw,            # 不做特征选择的基准
        "卡方过滤（20个）": X_chi2,            # 卡方检验过滤后
        "F检验过滤（20个）": X_ftest,          # F检验过滤后
        "互信息过滤（20个）": X_mi,            # 互信息过滤后
    }

    results = {}                               # 存储各方案的准确率结果

    print(f"  {'方法':<20}{'特征数':>6}{'准确率均值':>12}{'准确率标准差':>14}{'进度条'}")
    print(f"  {'-'*70}")

    for name, X in datasets.items():
        # cross_val_score: cv=5 表示5折，scoring='accuracy' 用准确率评估
        scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
        mean_acc = scores.mean()               # 5次准确率的均值
        std_acc = scores.std()                 # 5次准确率的标准差（反映稳定性）

        results[name] = (mean_acc, std_acc)    # 存入结果字典

        bar = "█" * int(mean_acc * 30)         # 按准确率画文字进度条（总长度30）
        print(f"  {name:<20}{X.shape[1]:>6}{mean_acc:>10.4f}{std_acc:>14.4f}  {bar}")

    print()
    return results                             # 返回所有方法的准确率结果


# =============================================================================
# 第七步：可视化——四图综合展示
# =============================================================================
def visualize_results(feature_names, variances, chi2_scores, f_scores, mi_scores,
                      performance_results):
    """
    【函数作用】用四个子图可视化特征选择过程与结果。

    【图表内容】
      图1（左上）：各特征的方差柱状图（红线为过滤阈值）
      图2（右上）：卡方值、F值、互信息的对比折线图（归一化）
      图3（左下）：三种方法的特征得分归一化热力图
      图4（右下）：各方法的模型准确率对比柱状图

    【参数】
    - feature_names     : 30个特征的名称
    - variances         : 方差数组（30,）
    - chi2_scores       : 卡方值数组（30,）
    - f_scores          : F值数组（30,）
    - mi_scores         : 互信息数组（30,）
    - performance_results: 模型准确率字典
    """
    print("=" * 60)
    print("  第七步：可视化结果（生成4个子图）")
    print("=" * 60)

    # ── 创建 2×2 的大图画布 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))    # 2行2列，总大小 16×12 英寸
    fig.suptitle('过滤法特征选择 · 完整可视化报告', fontsize=16, fontweight='bold')

    # ────────────────────────────────────────────────────────────────────────
    # 子图1（左上）：特征方差柱状图
    # ────────────────────────────────────────────────────────────────────────
    ax1 = axes[0, 0]                           # 取左上角的坐标轴
    threshold_val = 0.01                       # 方差过滤阈值（与第二步保持一致）

    bar_colors = ['#e74c3c' if v < threshold_val else '#3498db'
                  for v in variances]           # 低于阈值用红色，高于用蓝色

    ax1.bar(range(len(variances)), variances, color=bar_colors, alpha=0.8, edgecolor='white')
    # bar：绘制柱状图，x 轴为特征索引，y 轴为方差值

    ax1.axhline(y=threshold_val, color='red', linestyle='--', linewidth=2,
                label=f'过滤阈值 = {threshold_val}')
    # axhline：画水平参考线，虚线红色

    ax1.set_title('方差过滤：各特征方差分布', fontsize=12, fontweight='bold')
    ax1.set_xlabel('特征索引（0-29）')
    ax1.set_ylabel('方差值')
    ax1.legend()                               # 显示图例（过滤阈值说明）
    ax1.set_xticks(range(len(variances)))      # 设置 x 轴刻度为 0~29
    ax1.set_xticklabels(range(len(variances)), fontsize=7)

    # ── 标注前3个低方差特征 ─────────────────────────────────────────────────
    low_var_idx = np.argsort(variances)[:3]    # 找出方差最小的3个特征的索引
    for idx in low_var_idx:
        ax1.annotate(f'#{idx}', xy=(idx, variances[idx]),
                     xytext=(idx + 0.5, variances[idx] + 0.002),
                     fontsize=7, color='red')  # 在柱子上方标注特征编号

    # ────────────────────────────────────────────────────────────────────────
    # 子图2（右上）：三种方法的得分对比折线图（归一化后叠加）
    # ────────────────────────────────────────────────────────────────────────
    ax2 = axes[0, 1]                           # 取右上角坐标轴

    def normalize(arr):
        """将数组归一化到 [0,1]，防止分母为0"""
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / (rng + 1e-10)

    chi2_norm = normalize(chi2_scores)         # 归一化卡方值
    f_norm = normalize(f_scores)               # 归一化 F 值
    mi_norm = normalize(mi_scores)             # 归一化互信息值

    x_idx = range(len(feature_names))          # x轴为特征索引 0~29
    ax2.plot(x_idx, chi2_norm, 'o-', color='#e74c3c', label='卡方值（归一化）',
             markersize=5, linewidth=1.5)      # 红色折线+圆点
    ax2.plot(x_idx, f_norm, 's-', color='#2ecc71', label='F值（归一化）',
             markersize=5, linewidth=1.5)      # 绿色折线+方块
    ax2.plot(x_idx, mi_norm, '^-', color='#9b59b6', label='互信息（归一化）',
             markersize=5, linewidth=1.5)      # 紫色折线+三角

    ax2.set_title('三种评分方法对比（归一化到[0,1]）', fontsize=12, fontweight='bold')
    ax2.set_xlabel('特征索引（0-29）')
    ax2.set_ylabel('归一化得分')
    ax2.legend(loc='upper right', fontsize=9)  # 图例放右上角
    ax2.set_xticks(range(len(feature_names)))
    ax2.set_xticklabels(range(len(feature_names)), fontsize=7)
    ax2.grid(True, alpha=0.3)                  # 添加网格线（半透明）

    # ────────────────────────────────────────────────────────────────────────
    # 子图3（左下）：三种方法的特征重要性热力图
    # ────────────────────────────────────────────────────────────────────────
    ax3 = axes[1, 0]

    # 构建 3×30 的得分矩阵（3种方法 × 30个特征）
    score_matrix = np.vstack([chi2_norm, f_norm, mi_norm])
    # vstack：将三个一维数组垂直堆叠为二维矩阵，形状 (3, 30)

    im = ax3.imshow(score_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    # imshow：将矩阵绘制为热力图；cmap='YlOrRd'：黄→橙→红，值越大颜色越深

    plt.colorbar(im, ax=ax3, label='归一化得分（0=低, 1=高）')
    # colorbar：在图旁添加颜色条（色标）

    ax3.set_yticks([0, 1, 2])                  # y 轴3个刻度（对应3种方法）
    ax3.set_yticklabels(['卡方值', 'F值', '互信息'], fontsize=10)
    ax3.set_xticks(range(len(feature_names)))
    ax3.set_xticklabels(range(len(feature_names)), fontsize=7)
    ax3.set_title('特征重要性热力图（3种方法综合）', fontsize=12, fontweight='bold')
    ax3.set_xlabel('特征索引（0-29）')

    # ── 在热力图上标注高分区域的数值 ────────────────────────────────────────
    for method_i in range(3):                  # 遍历3种方法
        for feat_j in range(len(feature_names)):
            val = score_matrix[method_i, feat_j]
            if val > 0.7:                      # 只标注得分 > 0.7 的单元格
                ax3.text(feat_j, method_i, f'{val:.1f}', ha='center', va='center',
                         fontsize=6, color='white', fontweight='bold')

    # ────────────────────────────────────────────────────────────────────────
    # 子图4（右下）：各方法准确率对比柱状图
    # ────────────────────────────────────────────────────────────────────────
    ax4 = axes[1, 1]

    method_names = list(performance_results.keys())           # 方法名列表
    mean_accs = [v[0] for v in performance_results.values()]  # 均值准确率列表
    std_accs = [v[1] for v in performance_results.values()]   # 标准差列表

    colors = ['#95a5a6', '#e74c3c', '#2ecc71', '#9b59b6']     # 4种颜色对应4种方法
    bars = ax4.bar(range(len(method_names)), mean_accs,
                   color=colors, alpha=0.85, edgecolor='white',
                   yerr=std_accs, capsize=6)                  # yerr添加误差棒

    # ── 在每个柱子上方标注准确率数值 ─────────────────────────────────────────
    for bar, mean, std in zip(bars, mean_accs, std_accs):
        ax4.text(bar.get_x() + bar.get_width() / 2,          # x: 柱子中心
                 bar.get_height() + std + 0.001,               # y: 柱子顶部+间距
                 f'{mean:.4f}',                                # 显示4位小数准确率
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.set_xticks(range(len(method_names)))
    ax4.set_xticklabels(method_names, rotation=15, ha='right', fontsize=9)
    ax4.set_ylabel('5折交叉验证准确率')
    ax4.set_title('各方法模型准确率对比', fontsize=12, fontweight='bold')
    ax4.set_ylim(0.9, 1.02)                    # y 轴范围放大差异
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2%}'))
    # ↑ y 轴刻度格式化为百分比
    ax4.grid(True, axis='y', alpha=0.3)        # 只画水平网格线

    # ── 最高准确率的柱子加金色边框突出显示 ────────────────────────────────────
    best_idx = int(np.argmax(mean_accs))        # 找出准确率最高的方法索引
    bars[best_idx].set_edgecolor('gold')        # 金色边框
    bars[best_idx].set_linewidth(3)             # 加粗边框

    # ── 调整子图间距并保存 ───────────────────────────────────────────────────
    plt.tight_layout(pad=2.0)                  # 自动调整子图间距
    save_path = r'C:\Users\16384\Desktop\AI001\02_machine_learning\过滤法特征选择结果.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # dpi=150：分辨率；bbox_inches='tight'：裁掉多余空白
    print(f"  [OK] 图表已保存至: {save_path}")
    plt.show()                                 # 在窗口中显示图表
    print()


# =============================================================================
# 第八步：三方法特征共识分析（哪些特征被多种方法认为重要）
# =============================================================================
def consensus_analysis(feature_names, chi2_scores, f_scores, mi_scores, top_k=15):
    """
    【函数作用】分析哪些特征被多种过滤方法共同认为是重要的（共识特征）。

    【原理】
    - 若某特征在卡方、F检验、互信息三种方法中都排名靠前，
      说明它"多角度证明"自己的重要性，可信度更高。
    - 用"投票"策略：给每种方法的 top_k 名特征各加1票，统计总票数。

    【参数】
    - feature_names : 特征名称列表
    - chi2_scores   : 卡方值（30,）
    - f_scores      : F值（30,）
    - mi_scores     : 互信息值（30,）
    - top_k         : 每种方法选前几名（默认15）
    """
    print("=" * 60)
    print("  第八步：三方法共识特征分析（投票策略）")
    print("=" * 60)
    print(f"  【方法】每种评分方法选出前{top_k}名，统计每个特征被选中几次\n")

    # ── 获取各方法的前 top_k 名特征索引 ─────────────────────────────────────
    chi2_top = set(np.argsort(chi2_scores)[-top_k:])   # 卡方值最高的 top_k 个索引
    f_top = set(np.argsort(f_scores)[-top_k:])         # F值最高的 top_k 个索引
    mi_top = set(np.argsort(mi_scores)[-top_k:])       # 互信息最高的 top_k 个索引

    # ── 计算每个特征的投票数（0~3） ─────────────────────────────────────────
    votes = np.zeros(len(feature_names), dtype=int)    # 初始化投票数组全为0
    for idx in chi2_top:
        votes[idx] += 1                                 # 卡方选中 → +1票
    for idx in f_top:
        votes[idx] += 1                                 # F检验选中 → +1票
    for idx in mi_top:
        votes[idx] += 1                                 # 互信息选中 → +1票

    # ── 按投票数分组打印 ────────────────────────────────────────────────────
    for vote_count in [3, 2, 1]:                        # 从高到低遍历投票数
        selected = np.where(votes == vote_count)[0]     # 找出得票数等于 vote_count 的索引
        if len(selected) == 0:
            continue                                    # 没有该得票数则跳过

        label = {3: "三方法共识（最可靠）", 2: "两方法共识（较可靠）",
                 1: "单方法选中（参考）"}[vote_count]
        print(f"  ★{'★'*(vote_count-1)} {label}：共 {len(selected)} 个特征")

        for idx in selected:
            # 计算该特征在三种方法中各自的排名（第几名）
            chi2_rank = int(np.sum(chi2_scores > chi2_scores[idx])) + 1
            f_rank = int(np.sum(f_scores > f_scores[idx])) + 1
            mi_rank = int(np.sum(mi_scores > mi_scores[idx])) + 1
            print(f"     [{idx:2d}] {feature_names[idx]:<35}"
                  f"(卡方#{chi2_rank:2d}, F检验#{f_rank:2d}, 互信息#{mi_rank:2d})")
        print()

    # ── 打印三方法共识特征列表（最终推荐） ──────────────────────────────────
    consensus_idx = np.where(votes == 3)[0]            # 三种方法都选中的特征索引
    print(f"  最终推荐保留的特征（三方法共识 {len(consensus_idx)} 个）：")
    print(f"  {[feature_names[i] for i in consensus_idx]}\n")


# =============================================================================
# 主函数：统一调用所有步骤
# =============================================================================
def main():
    """
    【主函数】按顺序调用所有功能函数，完成完整的过滤法特征选择流程。

    执行流程：
    1. 加载数据            → 获取特征矩阵 X 和标签 y
    2. 方差过滤            → 删除低方差特征
    3. 卡方检验过滤        → 按与标签的卡方相关性选特征
    4. F检验过滤           → 按组间/组内方差比选特征
    5. 互信息过滤          → 按信息量选特征（支持非线性）
    6. 模型性能对比        → 验证各过滤法对准确率的影响
    7. 可视化              → 生成4图综合报告
    8. 共识分析            → 投票找出最可靠的特征
    """
    print("\n" + "█" * 60)
    print("  机器学习 · 过滤法特征选择 完整实战")
    print("  数据集：乳腺癌（Breast Cancer Wisconsin）")
    print("  方法：方差过滤 / 卡方检验 / F检验 / 互信息")
    print("█" * 60 + "\n")

    # ── Step1: 数据加载与预处理 ──────────────────────────────────────────────
    X_raw, X_scaled, y, feature_names = load_and_preprocess()

    # ── Step2: 方差过滤（使用原始数据） ─────────────────────────────────────
    X_var_filtered, var_selector, var_kept_names = variance_filter(
        X_raw, feature_names, threshold=0.01
    )
    variances = np.var(X_raw, axis=0)           # 保存方差数组，后续可视化用

    # ── Step3: 卡方检验（使用归一化数据，卡方检验要求非负） ─────────────────
    X_chi2, chi2_scores, chi2_pvalues, chi2_kept = chi2_filter(
        X_scaled, y, feature_names, k=20
    )

    # ── Step4: F检验（使用原始数据，F检验无非负要求） ──────────────────────
    X_ftest, f_scores, f_pvalues, ftest_kept = ftest_filter(
        X_raw, y, feature_names, k=20
    )

    # ── Step5: 互信息（使用原始数据） ────────────────────────────────────────
    X_mi, mi_scores, mi_kept = mutual_info_filter(
        X_raw, y, feature_names, k=20
    )

    # ── Step6: 模型性能对比 ───────────────────────────────────────────────────
    results = evaluate_model_performance(X_raw, X_chi2, X_ftest, X_mi, y)

    # ── Step7: 可视化（保存图片并展示） ─────────────────────────────────────
    visualize_results(feature_names, variances, chi2_scores, f_scores, mi_scores, results)

    # ── Step8: 共识分析（三方法投票） ────────────────────────────────────────
    consensus_analysis(feature_names, chi2_scores, f_scores, mi_scores, top_k=15)

    # ── 最终总结 ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  运行完成！总结")
    print("=" * 60)
    print(f"  原始特征数      : {X_raw.shape[1]} 个")
    print(f"  方差过滤后      : {X_var_filtered.shape[1]} 个（阈值=0.01）")
    print(f"  卡方过滤后      : {X_chi2.shape[1]} 个（选前20）")
    print(f"  F检验过滤后     : {X_ftest.shape[1]} 个（选前20）")
    print(f"  互信息过滤后    : {X_mi.shape[1]} 个（选前20）")
    print()

    # 找出准确率最高的方法并标注
    best_method = max(results, key=lambda k: results[k][0])
    print(f"  模型准确率对比：")
    for name, (mean, std) in results.items():
        best_mark = " ← 最优" if name == best_method else ""
        print(f"    {name:<20}: {mean:.4f} ± {std:.4f}{best_mark}")

    print()
    print("  过滤法优缺点总结：")
    print("    [+] 优点：计算快速、无需训练模型、通用性强")
    print("    [+] 方差过滤 ：简单粗暴，删除无变化特征（近常数列）")
    print("    [+] 卡方检验 ：适合非负特征与分类标签的相关性检验")
    print("    [+] F检验    ：适合连续特征，假设线性关系，速度快")
    print("    [+] 互信息   ：最强大，能捕捉非线性关系，无分布假设")
    print("    [-] 缺点：忽略特征间的相互作用（交互项）")
    print("    [-] 缺点：未考虑具体模型的偏好")
    print()
    print("  【下一步建议】")
    print("    → 尝试包装法（Wrapper）：RFE 递归特征消除")
    print("    → 尝试嵌入法（Embedded）：L1正则化、树模型特征重要性")
    print("=" * 60)


# =============================================================================
# 程序入口：当直接运行本文件时执行 main()
# =============================================================================
if __name__ == '__main__':
    main()                                     # 调用主函数，启动整个流程
