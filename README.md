# project

## 介绍

## 环境依赖 
pytorch框架

## 目录结构

```
.
|-Project
    ├─dataset           //数据集
    │  └─image                  
    │      ├─cardboard
    │      ├─glass
    │      ├─metal
    │      ├─paper
    │      ├─plastic
    │      └─trash
    ├─input             //待预测图片————同数据集类别目录要一致
    │  ├─pre_img
    │  │  ├─cardboard
    │  │  ├─glass
    │  │  ├─metal
    │  │  ├─paper
    │  │  ├─plastic
    │  │  └─trash
    │  └─pre_img1
    │  │  ├─cardboard
    │  │  ├─glass
    │  │  ├─metal
    │  │  ├─paper
    │  │  ├─plastic
    │  │  └─trash
    │  └─5.png          //单张待预测图
    ├─model                 //模型保存
    ├─recycleBin            //临时回收站
    ├─wandb                 //wandb可视化日志
    ├─detect_1_resnet18.py           //使用resnet18预测单张/多张
    ├─detect_1_resnet50.py           //使用resnet50预测单张/多张
    ├─detect_net_cnn_22.py           //使用net_cnn_22预测单张/多张
    ├─load_dataset_1.py              //数据预处理
    ├─net_cnn_22.png                 //自定义网络架构
    ├─net_cnn_22.py                  //自定义cnn网络
    ├─normalize.py                   //计算mean、std
    ├─test.py                        //测试代码
    ├─train.py                       //train1.0
    ├─train_1_resnet_18.py           //train2.0
    ├─train_1_resnet_18_22.py        //train3.0
    ├─train_1_resnet_18_wandb.py     //train4.0
    ├─train_1_resnet50.py            //train_resnet50
    ├─train_net_cnn_22.py            //train_cnn_22
    ├─参考资料                        //资料  
    └─__pycache__
```


## 代码使用说明

###说明：
#### 1、数据集：
链接：https://pan.baidu.com/s/1LMGoLA4f_xuC3q-7YdJweg 
提取码：cz7y
##### 2、训练
使用train.py训练
##### 3、模型保存
model文件夹下有一个训练好的模型,效果不错;
#### 4、预测
使用detect.py预测单张或多张

### 主要使用代码
1、train.py——训练

2、detect.py———预测
