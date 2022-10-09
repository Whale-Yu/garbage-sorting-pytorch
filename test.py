# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2022/9/9 16:20
# @Author :yujunyu
# @Site :
# @File :test.py
# @software: PyCharm

"""
      ┏┛ ┻━━━━━┛ ┻┓
      ┃　　　　　　 ┃
      ┃　　　━　　　┃
      ┃　┳┛　  ┗┳　┃
      ┃　　　　　　 ┃
      ┃　　　┻　　　┃
      ┃　　　　　　 ┃
      ┗━┓　　　┏━━━┛
        ┃　　　┃   神兽保佑
        ┃　　　┃   代码无BUG！
        ┃　　　┗━━━━━━━━━┓
        ┃CREATE BY SNIPER┣┓
        ┃　　　　         ┏┛
        ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
          ┃ ┫ ┫   ┃ ┫ ┫
          ┗━┻━┛   ┗━┻━┛

"""
# save_epoch=1
#
# for e in range(100):
#     if e%save_epoch==0:
#         print('y')
#     else:
#         print('n')

# import os
# filePath = './dataset/image'
# for i,j,k in os.walk(filePath):
#     # print(k)#直接输出文件名
#     #一个文件名一行，输出
#     for s in k:
#         print(s)

lr = 0.001
lr_down_epoch = 20
lr_list=[]
for e in range(1, 100):
    print(lr)
    # 按lr_down_epoch降低lr
    lr_list.append(lr)
    if e % lr_down_epoch == 0:
        lr = lr / 5

#
#
# from matplotlib import pyplot as plt
#
# x1 = range(1, 100)
# plt.subplot(2, 2, 1)
# # plt.plot(x1, train_accuracy_list, 'o-', label='train accuracy')
# # plt.plot(x1, val_accuracy_list, 'o-', label='val_acc')
# plt.ylabel('accuracy')
# plt.legend(loc='best')
# plt.subplot(2, 2, 2)
# # plt.plot(x1, train_loss_list, '.-', label='train_loss')
# # plt.plot(x1, val_loss_list, '.-', label='val_loss')
# plt.ylabel('loss')
# plt.legend(loc='best')
# plt.subplot(2, 2, 3)
# plt.plot(x1, lr_list, '.-', label='learing_rate')
# plt.ylabel('lr')
# plt.legend(loc='best')
# plt.show()
from numpy import array,float32

import wandb

#
# wandb.init(project="垃圾分类", entity="yujunyu")
#
# epoch_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# train_accuracy_list = [array(99.782135, dtype=float32), array(99.6732, dtype=float32), array(99.782135, dtype=float32),
#                        array(99.782135, dtype=float32), array(99.72767, dtype=float32), array(99.782135, dtype=float32),
#                        array(99.72767, dtype=float32), array(99.8366, dtype=float32), array(99.782135, dtype=float32),
#                        array(99.782135, dtype=float32)]
# test_accuracy_list = [array(99.99999, dtype=float32), array(99.99999, dtype=float32), array(99.99999, dtype=float32),
#                       array(99.99999, dtype=float32), array(99.99999, dtype=float32), array(99.99999, dtype=float32),
#                       array(99.99999, dtype=float32), array(99.99999, dtype=float32), array(99.99999, dtype=float32),
#                       array(99.99999, dtype=float32)]
#
# wandb.log({"train/test accuracy": wandb.plot.line_series(
#     xs=epoch_list,
#     ys=[train_accuracy_list, test_accuracy_list],
#     keys=["train_acc", "test_acc"],
#     title="Train/Test Accuracy",
#     xname="epoch")
# })
