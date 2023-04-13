# 👇体感互动——Hand Pose Estimation Based on Mediapipe And LSTM

[TOC]
## 🐱Introduce
**体感互动——Hand Pose Estimation Based on Mediapipe And LSTM**


基于mediapipe提取人体、手部等关键点和LSTM算法实现手势识别，并进一步实现体感互动的功能，如进行隔空移动、抓取、放大、缩小等手势操作，可为商业显示提供智能交互应用，如3D模型的展示。


* demo1:hand_pose_estimation
![demo1.gif](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZTg2MGZkZmRmMmM3ZWM5MzFiYmZkOTNiMDE3YmY4YWUxZjAwOWYxMiZjdD1n/vDUQG87L8nGSPZPqhg/giphy-downsized-large.gif)

* demo2:体感互动
![demo2.gif](https://media.giphy.com/media/HTxgKNyrBWmCCfZTqY/giphy.gif)

## 😔‍Process

### 1、训练模型
* train_my_hand_pose文件夹
### 2、预测
* predict系列代码
### 3、体感互动
* 03_predict之体感互动.py


## 🐖How to Use Code

1）依赖包

``pip install tensorflow-gpu mediapipe
  ``

2）手势实时预测

``python 02_predict.py
  ``

3）体感互动

``python 03_predict之体感互动.py
  ``

## 🐏Reference

[Mediapipe](https://google.github.io/mediapipe/)

[ActionDetectionforSignLanguage](https://github.com/nicknochnack/ActionDetectionforSignLanguage)

[VRMap](https://720.vrqjcs.com/t/9332870054821ffc)

## 🐕Thanks
@Studio:JHC Software Dev Studio 

@Mentor:HuangRiChen

@Author:YuJunYu
