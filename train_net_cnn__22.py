import torch
import os
from matplotlib import pyplot as plt

# 注意自定义模型的导入
from load_dataset_1 import load_data
from net_cnn_22 import Net

import wandb

wandb.init(project="垃圾分类（cnn）", entity="yujunyu")

class Train:
    def __init__(self, data_path="./dataset/image", start_epoch=0, epoch=10, lr=0.0001, batch_size=128,
                 module_file='./model/garbage_2.pth'):
        super(Train, self).__init__()
        print('训练准备......')
        # ---训练相关的初始化---
        self.CUDA = torch.cuda.is_available()
        self.batch_size = batch_size
        # 数据集
        self.train, self.test, self.cls_idx = load_data(data_path, shape=(224, 224), rate=0.8,
                                                        batch_size=self.batch_size)
        print(self.cls_idx)
        # 网络 累加训练
        self.module_file = module_file
        if os.path.exists(self.module_file):
            print("加载本地模型")
            self.net = Net()
            if self.CUDA:
                self.net.cuda()
            state = torch.load(self.module_file)
            self.net.load_state_dict(state)
        else:
            print("加载预训练模型")
            self.net = Net()
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

        # ReduceLROnPlateau
        # self.scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5,
        #                                                        verbose=True, threshold=0.0001, threshold_mode='rel',
        #                                                        cooldown=0, min_lr=0, eps=1e-10)
        # StepLR
        # self.scheduler_lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)    # 每20epoch——>lr*0.1，

        self.scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.epoch,eta_min=0,last_epoch=-1)

    # 记录train
    def Log(self, e, epoch, train_accuracy, val_accuracy, train_loss, val_loss):
        str = f"epoch:{e}/{epoch} \t train_acc:{train_accuracy} \t val_acc:{val_accuracy} \t train_loss:{train_loss} \t val_loss:{val_loss}"
        print(str)
        with open(f'{self.module_file}.txt', 'a+', encoding='utf-8') as f:
            f.write(str)
            f.write('\n')

    # 训练
    def execute(self):
        print('训练开始......')
        # 保存频率
        save_epoch = 1
        lr_down_epoch = 20
        lr_list = []
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
            # 更新lr
            # self.scheduler_lr.step()
            # self.scheduler_lr.step(loss)
            self.scheduler_lr.step()


            train_accuracy = num_correct * 100.0 / num_samples


            # 使用验证数据集验证
            val_accuracy, val_loss = self.validate()

            print(
                f"epoch:{e}/{epoch} \t train_acc:{train_accuracy} \t val_acc:{val_accuracy} \t train_loss:{loss} \t val_loss:{val_loss}")
            print(self.optimizer.state_dict()['param_groups'][0]['lr'])
            wandb.log({
                "Train Accuracy": train_accuracy,
                "Test Accuracy": val_accuracy,
                # "Accuracy": {"Train:": train_accuracy, "Test": val_accuracy},
                "Train Loss": loss,
                "Test Loss": val_loss,
                "Epoch": e,  # 加上epoch，可视化可以使step变epoch
                "Learning Rate": self.optimizer.state_dict()['param_groups'][0]['lr'],
            })

            # 根据save_epoch保存模型
            if e % save_epoch == 0:
                torch.save(self.net.state_dict(), self.module_file)


        # 保存模型  torch.save(model.state_dict(), model_path)
        torch.save(self.net.state_dict(), self.module_file)

    # 评估
    @torch.no_grad()
    def validate(self):
        self.net.eval()  # 测试前加
        num_samples = 0.0
        num_correct = 0.0
        for samples, labels in self.test:
            if self.CUDA:
                samples = samples.cuda()
                labels = labels.cuda()
            # 累加验证集的总数量
            num_samples += len(samples)
            # 输出
            out = self.net(samples.view(-1, 3, 224, 224))
            val_loss = self.loss_function(out, labels)
            # 转换为概率[0, 1)
            out1 = torch.nn.functional.softmax(out, dim=1)
            # 输出预测类别
            y = torch.argmax(out1, dim=1)
            # 累加预测正确的数量
            num_correct += (y == labels).float().sum()
            # print(y, labels, y == labels)
            # print(num_samples)
            # print(num_correct)
        # 返回准确率
        return num_correct * 100.0 / num_samples, val_loss

if __name__ == "__main__":
    dataset_path = './dataset/image'
    start_epoch = 1
    epoch = 100
    lr = 0.001
    batch_size = 128
    module_file = './model/cnn_15.pth'

    trainer = Train(dataset_path, start_epoch, epoch, lr, batch_size, module_file)
    trainer.execute()

    print("训练结束！")
