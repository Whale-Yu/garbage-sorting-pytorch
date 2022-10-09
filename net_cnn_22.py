# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2022/9/11 18:11
# @Author :yujunyu
# @Site :
# @File :net.py
# @software: PyCharm

import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.max_pool2d = torch.nn.MaxPool2d(2, 2)

        self.layer1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1)

        self.layer2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)

        self.layer3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1)

        self.layer4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)

        self.layer5 = torch.nn.Linear(in_features=14 * 14 * 32, out_features=64)

        self.layer6 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, input):
        t = self.layer1(input)
        t = torch.nn.functional.relu(t)
        t = self.max_pool2d(t)

        t = self.layer2(t)
        t = torch.nn.functional.relu(t)
        t = self.max_pool2d(t)

        t = self.layer3(t)
        t = torch.nn.functional.relu(t)
        t = self.max_pool2d(t)

        t = self.layer4(t)
        t = torch.nn.functional.relu(t)
        t = self.max_pool2d(t)

        t = t.view(-1, 14 * 14 * 32)

        t = self.layer5(t)
        t = torch.nn.functional.relu(t)

        t = torch.nn.functional.dropout(t, p=0.5, training=self.training)

        t = self.layer6(t)

        return t


# 测试
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
summary(net, (3, 224, 224))
