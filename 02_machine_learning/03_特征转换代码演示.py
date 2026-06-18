import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# 归一化（Normalization）
def normal():
    x = [[1, 2], [0.5, 6], [0, 10], [1, 18]]
    # 得到归一化的缩放器对象
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)
    print(x)

# 标准化（Standardization）
def standard():
    x = [[1, 2], [0.5, 6], [0, 10], [1, 18]]
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    print(x)

# 标签编码（Label Encoding）
def label():
    data = ['本科', '硕士', '博士', '本科']
    encoder = LabelEncoder()
    result = encoder.fit_transform(data)
    print(result)

# 独热编码（One-Hot Encoding）
def one_hot():
    X = np.array([['红'],
                  ['蓝'],
                  ['绿'],
                  ['红']])

    encoder = OneHotEncoder()
    result = encoder.fit_transform(X).toarray()

    print(result)

if __name__ == '__main__':
    print("归一化:")
    print(normal())
    print()

    print("标准化:")
    print(standard())
    print()

    print("标签编码:")
    print(label())
    print()

    print("独热编码:")
    print(one_hot())
    print()
