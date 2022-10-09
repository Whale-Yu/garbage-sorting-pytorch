# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2022/9/9 16:20
# @Author :yujunyu
# @Site :
# @File :test.py
# @software: PyCharm

"""
加载数据集、数据预处理
    数据集目录格式:
        |-dataset
            |-images
                |-类别1
                |-类别2
                |-类别3
                |-类别4
                |-...
    注意:同一类别在同一文件夹下，不同类别文件夹是同级目录
"""

from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomResizedCrop, ColorJitter, RandomGrayscale
import torch.utils.data

# 加载指定目录下的图像，返回根据切分比例形成的数据加载器
def load_data(img_dir, shape=(224, 224), rate=0.7, batch_size=128):
    transform = Compose(
        [
            Resize(shape),
            # RandomResizedCrop((224, 224)),
            RandomHorizontalFlip(),  # 0.5的进行水平翻转
            RandomVerticalFlip(),  # 0.5的进行垂直翻转
            ToTensor(),  # PIL转tensor
            # Normalize(mean=[0.56719673, 0.5293289, 0.48351972], std=[0.20874391, 0.21455203, 0.22451781])
            Normalize(mean=[0.6719, 0.6417, 0.6101], std=[0.2078, 0.2083, 0.2302])  # 表示图像集每个通道的均值和标准差序列。
            # 归一化   # 输入必须是Tensor
        ]
    )
    # 加载数据集
    dataset = ImageFolder(img_dir, transform=transform)

    all_num = len(dataset)
    # print(l)
    train_num = int(all_num * rate)
    # print(train_num)
    # 划分数据集
    train, test = torch.utils.data.random_split(dataset, [train_num, all_num - train_num])

    # 封装批处理的迭代器（加载器）
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, dataset.class_to_idx

# # # 测试
# train, test, class_to_idx = load_data("./dataset/image")
# # # train
# img_num = 0
# lab_num = 0
# for image, label in train:
#     print(image, label)
#     img_num += len(image)
#     lab_num += len(label)
# print(img_num, lab_num)
# # test
# img_num = 0
# lab_num = 0
# for image, label in test:
#     print(image, label)
#     img_num += len(image)
#     lab_num += len(label)
# print(img_num, lab_num)
#
# print(class_to_idx)
