# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2022/9/12 16:20
# @Author :yujunyu
# @Site :
# @File :test.py
# @software: PyCharm

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
import random


class GarbageRecognizer:
    def __init__(self, module_file="./model/garbage_9.pth"):
        super(GarbageRecognizer, self).__init__()
        self.module_file = module_file
        self.CUDA = torch.cuda.is_available()
        self.net = resnet18(pretrained=False, num_classes=6)
        if self.CUDA:
            self.net.cuda()
            device = 'cuda'
        else:
            device = 'cpu'
        state = torch.load(self.module_file, map_location=device)  # torch.load("./models", map_location=device)
        self.net.load_state_dict(state)
        print("加载模型完毕!")
        self.net.eval()

    @torch.no_grad()
    def recognzie(self, img):
        with torch.no_grad():
            # 开始识别
            if self.CUDA:
                img = img.cuda()
            # print(pre_img)
            img = img.view(-1, 3, 224, 224)
            y = self.net(img)
            p_y = torch.nn.functional.softmax(y, dim=1)
            p, cls_idx = torch.max(p_y, dim=1)
            return cls_idx.cpu(), p.cpu()


# 识别测试
if __name__ == "__main__":
    # 模型
    model_file = 'model/resnet18_6.pth'
    recognizer = GarbageRecognizer(model_file)

    # # 下面转换用于独立的图像，并对其做预处理
    transform = Compose(
        [
            Resize((224, 224)),
            # RandomHorizontalFlip(),  # 0.5的进行水平翻转
            # RandomVerticalFlip(),  # 0.5的进行垂直翻转
            ToTensor(),  # PIL转tensor
            Normalize(mean=[0.6719, 0.6417, 0.6101], std=[0.2078, 0.2083, 0.2302])
        ]
    )
    dataset = ImageFolder('./dataset/image', transform=transform)
    pre_dataset = ImageFolder("./input/pre_img1", transform=transform)
    print(pre_dataset.class_to_idx)

    '''
    数据集随机预测
    '''
    sample_num = len(dataset)
    print(sample_num)
    pre_num = 10
    samples_idx = random.sample(range(0, sample_num), pre_num)
    print(f'数据集随机采样{pre_num}张')
    # 循环识别
    true_num1 = 0
    label_list, pre_list, img_list = [], [], []
    for idx in samples_idx:
        cls, p = recognizer.recognzie(dataset[idx][0])
        # print(dataset[idx][0])  # dataset[idx][0]像素  dataset[idx][1]类别标签（0~5）
        real_cls1 = dataset.classes[dataset[idx][1]]
        pre_cls1 = dataset.classes[cls]

        label_list.append(real_cls1)
        pre_list.append(pre_cls1)
        img = dataset[idx][0]
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        img_list.append(img)
        # plt.imshow(img)
        # plt.show()
        if real_cls1 == pre_cls1:
            true_num1 += 1
        print(
            f'真实:{dataset[idx][1]}-{real_cls1} \t\t\t\t 预测:{cls.numpy()[0]}-{pre_cls1} \t\t\t\t 概率:{p.numpy()[0]}')
    print(f'准确率:{true_num1 / pre_num}')
    # print(label_list,pre_list)
    fig = plt.gcf()
    fig.set_size_inches(10, 12)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i in range(0, pre_num):
        ax = plt.subplot(4, 4, i + 1)
        ax.imshow(img_list[i])
        title = "true:" + str(label_list[i])
        title += ", pre:" + str(pre_list[i])
        ax.set_title(title, fontsize=10)
        # ax.set_xticks([])  # 不显示坐标轴
        # ax.set_yticks([])
    plt.xlabel(f'准确率:{true_num1 / pre_num * 100}%', loc='center')
    plt.show()

    '''
    预测图片随机预测
    '''
    # 样本总数
    sample_num = len(pre_dataset)
    print(sample_num)
    # 随机采样，采样数 10
    pre_num = 20
    samples_idx = random.sample(range(0, sample_num), pre_num)
    print(f'预测图片随机采样{pre_num}张')
    # 循环识别
    true_num = 0
    label_list, pre_list, img_list = [], [], []
    for idx in samples_idx:
        cls, p = recognizer.recognzie(pre_dataset[idx][0])
        # print(dataset[idx])
        real_cls1 = pre_dataset.classes[pre_dataset[idx][1]]
        pre_cls1 = pre_dataset.classes[cls]

        label_list.append(real_cls1)
        pre_list.append(pre_cls1)
        img = dataset[idx][0]
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        img_list.append(img)

        # 累加预测正确数量
        if real_cls1 == pre_cls1:
            true_num += 1
        print(
            f'真实:{pre_dataset[idx][1]}-{real_cls1} \t\t\t\t 预测:{cls.numpy()[0]}-{pre_cls1} \t\t\t\t 概率:{p.numpy()[0]}')
    print(f'准确率:{true_num / pre_num}')
    fig = plt.gcf()
    fig.set_size_inches(10, 12)
    for i in range(0, pre_num):
        ax = plt.subplot(5, 4, i + 1)
        ax.imshow(img_list[i])
        title = "true:" + str(label_list[i])
        title += ", pre:" + str(pre_list[i])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])  # 不显示坐标轴
        ax.set_yticks([])
    plt.xlabel(f'准确率:{true_num / pre_num * 100}%', loc='center')
    plt.show()

    '''
    单张图片预测
    '''
    img_filename = './input/pre_img/metal/img_5.png'
    print("预测单张图像：", img_filename)
    img = Image.open(img_filename)
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    cls, p = recognizer.recognzie(img)
    cls = pre_dataset.classes[cls]
    print(cls, '{:}%'.format(p.numpy()[0] * 100))
