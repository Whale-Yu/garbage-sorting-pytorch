# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2022/9/12 16:20
# @Author :yujunyu
# @Site :
# @File :train_1_resnet18_22.py
# @software: PyCharm

"""
修改：
    和’train_1_resnet18.py‘一致,只是修改了可视化的指标
        原有只是可视化val_accuracy、train_loss,当前可视化train_accuracy、train_loss、val_accuracy、val_loss
"""

import torch
from torchvision.models import resnet18
import os
from load_dataset_1 import load_data
from matplotlib import pyplot as plt


class Train:
    def __init__(self, data_path="./dataset/image", start_epoch=0, epoch=100, lr=0.0001, batch_size=128,
                 module_file='./model/garbage_2.pth'):
        super(Train, self).__init__()
        print('训练准备......')
        # ---训练相关的初始化---
        # cuda是否可用 True——>可用——>在gpu中运算  .cuda()
        self.CUDA = torch.cuda.is_available()
        #
        self.batch_size = batch_size
        # 数据集
        self.train, self.test, self.cls_idx = load_data(data_path, shape=(224, 224), rate=0.8, batch_size=batch_size)
        print(self.cls_idx)
        # 网络 累加训练
        self.module_file = module_file
        if os.path.exists(self.module_file):
            print("加载本地模型")
            self.net = resnet18(pretrained=False)
            fc_features = self.net.fc.in_features
            self.net.fc = torch.nn.Linear(in_features=fc_features, out_features=6)  # 6分类 修改输出,用in_features得到该层的输入，重写这一层
            if self.CUDA:
                self.net.cuda()
            state = torch.load(self.module_file)
            self.net.load_state_dict(state)
        else:
            print("加载预训练模型")
            self.net = resnet18(pretrained=True)
            fc_features = self.net.fc.in_features
            self.net.fc = torch.nn.Linear(in_features=fc_features, out_features=6)
            if self.CUDA:
                self.net.cuda()

        # 迭代轮数epoch
        self.epoch = epoch
        # 学习率lr
        self.lr = lr
        # 优化器
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # 损失函数——交叉熵
        self.loss_function = torch.nn.CrossEntropyLoss()
        if self.CUDA:
            self.loss_function = self.loss_function.cuda()

        # 断点续训
        self.start_epoch = start_epoch

    # 训
    def execute(self):
        print('训练开始......')
        # 保存频率
        save_epoch = 1
        train_accuracy_list = []
        val_accuracy_list = []
        train_loss_list = []
        val_loss_list = []
        for e in range(self.start_epoch, self.epoch):
            self.net.train()  # 训练前加
            num_samples = 0.0
            num_correct = 0.0
            for samples, labels in self.train:
                # 导数清零
                self.optimizer.zero_grad()
                if self.CUDA:
                    samples = samples.cuda()
                    labels = labels.cuda()
                # 计算输出
                y = self.net(samples.view(-1, 3, 224, 224))
                # train_acc
                pre = torch.nn.functional.softmax(y, 1)
                pre = torch.argmax(pre, 1)
                num_correct += (pre == labels).float().sum()
                num_samples += len(samples)
                # 计算损失
                loss = self.loss_function(y, labels)
                # 求导
                loss.backward()
                # 更新梯度
                self.optimizer.step()
            train_accuracy = num_correct * 100.0 / num_samples

            # 使用验证数据集验证
            val_accuracy, val_loss = self.validate()

            # 记录acc、loss
            str = f"epoch:{e}/{self.epoch} \t train_acc:{train_accuracy} \t val_acc:{val_accuracy} \t train_loss:{loss} \t val_loss:{val_loss}"
            print(str)
            with open(f'{self.module_file}.txt', 'a+', encoding='utf-8') as f:
                f.write(str)
                f.write('\n')

            train_accuracy_list.append(train_accuracy.detach().cpu().numpy())
            val_accuracy_list.append(val_accuracy.detach().cpu().numpy())
            train_loss_list.append(loss.detach().cpu().numpy())
            val_loss_list.append(val_loss.detach().cpu().numpy())
            # print(Accuracy_list, Loss_list)

            # 根据save_epoch保存模型
            if e % save_epoch == 0:
                torch.save(self.net.state_dict(), self.module_file)

        # 可视化
        x1 = range(self.start_epoch, self.epoch)
        plt.subplot(2, 1, 1)
        plt.plot(x1, train_accuracy_list, '.-', label='train accuracy')
        plt.plot(x1, val_accuracy_list, '.-', label='val_acc')
        plt.title('accuracy/loss')
        plt.ylabel('accuracy')
        plt.legend(loc='best')
        plt.subplot(2, 1, 2)
        plt.plot(x1, train_loss_list, '.-', label='train_loss')
        plt.plot(x1, val_loss_list, '.-', label='val_loss')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.savefig(f"{self.module_file}_accuracy_loss_{self.start_epoch}_{self.epoch}.jpg")
        plt.show()

        # 保存模型  torch.save(model.state_dict(), model_path)
        torch.save(self.net.state_dict(), self.module_file)


    # 评估
    @torch.no_grad()
    def validate(self):
        num_samples = 0.0
        num_correct = 0.0
        self.net.eval()  # 测试前加
        for samples, labels in self.test:
            if self.CUDA:
                samples = samples.cuda()
                labels = labels.cuda()
            # 累加验证数据集的总数量
            num_samples += len(samples)
            # 输出
            out = self.net(samples.view(-1, 3, 224, 224))
            # val_loss
            loss = self.loss_function(out, labels)
            # 转换为概率[0, 1)
            out = torch.nn.functional.softmax(out, dim=1)
            # 输出预测类别
            y = torch.argmax(out, dim=1)
            # 累加预测正确的数量
            num_correct += (y == labels).float().sum()
            # print(y, labels, y == labels)
        # 返回准确率
        return num_correct * 100.0 / num_samples, loss


if __name__ == "__main__":
    dataset_path = './dataset/image'
    start_epoch = 0
    epoch = 100
    # lr = 0.0001
    lr = 5.5e-5
    batch_size = 128
    module_file = './model/resnet18_7.pth'

    trainer = Train(dataset_path, start_epoch, epoch, lr, batch_size, module_file)
    trainer.execute()

    print("训练结束！")
