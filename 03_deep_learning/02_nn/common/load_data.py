import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_digital_data():
    # 1.加载数据集
    BASE = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE, '..', 'data', 'train.csv')
    data = pd.read_csv(data_path)

    # 2.区分特征和标签
    x = data.drop("label", axis=1)
    y = data['label']

    # 3.划分数据集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    # 4.归一化————没有异常值，范围是定死的//  标准化是未知范围（它就把它化成标准正态分布）
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 5.全部转成tensor
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch. tensor(y_train.to_numpy()).float()
    y_test = torch.tensor(y_test.to_numpy()).float()

    return x_train,x_test,y_train,y_test

if __name__ == '__main__':
    x_train,x_test,y_train,y_test = get_digital_data()
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)