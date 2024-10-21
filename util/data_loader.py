"""
@author : ZhengHeJie
@when : 2022-12-2
@homepage :
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset
# from torchtext.data import Field, BucketIterator
# from torchtext.datasets.translation import Multi30k
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader(object):
    """An abstract class representing a Dataset.
    数据表示的抽象类

    All other datasets should subclass it. All subclasses should override
    所有数据类都需要继承它，并重写它的方法
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, train=True):                    #csv_file 参数对象

        self.is_train = train                               #训练标志

    #数据加载方法，核心方法（数据加载，数据处理，数据增强等都将在这里进行）
    # index:数据索引
    def __getitem__(self, index):
        self.data_frame = pd.read_csv(index)
        # 按class标签进行分类
        labels = self.data_frame['class']
        # 获取除class外的数据
        x_df = self.data_frame.drop(['class'], axis=1)

        x = x_df.values
        print("Data shape: (%d, %d)" % x.shape)

        return x, labels

    #获取当前数据集长度
    def __len__(self):
        return len(self.data_frame) # 返回数据集长度


class DataSetAD(object):
    """An abstract class representing a Dataset.
    数据表示的抽象类

    All other datasets should subclass it. All subclasses should override
    所有数据类都需要继承它，并重写它的方法
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, path, train=True):
        # 读文件夹下每个数据文件的名称
        self.data_frame = pd.read_csv(path)
        # 获取标签
        self.labels = np.asarray(self.data_frame.iloc[:, 0])
        self.is_train = train                               #训练标志

    #数据加载方法，核心方法（数据加载，数据处理，数据增强等都将在这里进行）
    # index:数据索引
    def __getitem__(self, index):
        # 按class标签进行分类
        # 获取列元素
        labels = self.data_frame.iloc[index, -1:]
        labels = torch.tensor(labels).to(torch.float32).to(device)
        # 获取除class外的数据
        x_df = self.data_frame.iloc[index, :-1]
        # x = x_df.values
        # print("Data shape: (%d)" % x_df.shape)
        # x = torch.Tensor(x_df)
        x = torch.Tensor(x_df).to(torch.float32).to(device)
        return x, labels

    #获取当前数据集长度
    def __len__(self):
        return len(self.data_frame) # 返回数据集长度

class DataSetADFew(object):
    """An abstract class representing a Dataset.
    数据表示的抽象类

    All other datasets should subclass it. All subclasses should override
    所有数据类都需要继承它，并重写它的方法
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, feature, labels, train=True):
        # 读文件夹下每个数据文件的名称
        # self.train = pd.read_csv(path)
        self.feature = feature
        self.labels = labels
        # # 获取标签
        # self.labels = np.asarray(self.data_frame.iloc[:, 0])
        # self.is_train = train                               #训练标志

    #数据加载方法，核心方法（数据加载，数据处理，数据增强等都将在这里进行）
    # index:数据索引
    def __getitem__(self, index):
        # 按class标签进行分类
        # 获取列元素
        label = [self.labels[index]]
        label = torch.tensor(label).to(torch.float32).to(device)
        # 获取除class外的数据
        x_df = self.feature[index]
        # x = x_df.values
        # print("Data shape: (%d)" % x_df.shape)
        # x = torch.Tensor(x_df)
        x = torch.Tensor(x_df).to(torch.float32).to(device)
        return x, label

    #获取当前数据集长度
    def __len__(self):
        return len(self.feature) # 返回数据集长度

class myDataset(Dataset):
    def __init__(self, data_dir):
        # data_dir:输入路径

        # 读文件夹下每个数据文件的名称
        self.file_name = os.listdir(data_dir)

        self.data_path = []
        # 把每一个文件的路径拼接起来
        for index in range(len(self.file_name)):
            self.data_path.append(os.path.join(data_dir, self.file_name[index]))

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        # 读取每一个数据
        data = pd.read_csv(self.data_path[index], header=None)
        # 转成张量
        data = torch.tensor(data.values)

        return data


if __name__ == '__main__':
    train_data = DataSetAD('../dataset/UNSW_NB15_traintest_backdoor.csv')
    seed = 1024
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # torch的GPU随机性，为所有GPU设置随机种子
    # 打印所有数据
    # for i in range(len(train_data)):
    #     print(train_data[i])
    # 划分测试集和训练集
    train_size = int(len(train_data) * 0.7)
    test_size = len(train_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

    # np.where(train_dataset)
    print(len(train_dataset))
    print(len(test_dataset))

    # dataloader加载数据集
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    # validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(len(train_loader))
    print(len(test_loader))

    countTrainAnorm = 0
    countTrainNorm = 0
    for i, (train_sample,label) in enumerate(train_loader):
        # print("{} {} {}".format(i, train_sample,label))
        for i in label:
            if i == 1:
                countTrainAnorm += 1
            else:
                countTrainNorm += 1

    print("countAnorm", countTrainAnorm, "TrainAnorm", (countTrainAnorm / len(train_dataset)), "TrainNorm",
          (countTrainNorm / len(train_dataset)))

    countAnorm = 0
    countNorm = 0
    for k, (test_sample,label) in enumerate(test_loader):
        # print("{} {}".format(k, test_sample))
        # print(label.mean())
        for i in label:
            if i == 1:
                countAnorm += 1
            else:
                countNorm += 1

    print("countAnorm",countAnorm,"AnormRate",(countAnorm/len(test_dataset)),"NormRate",(countNorm/len(test_dataset)))
