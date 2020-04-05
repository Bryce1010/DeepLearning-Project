# Crowd Counting



### 一. 赛题简介  
本次实战项目需要参赛者基于给定的场景图片，开发出能够同时适用于密集和稀疏等多种复杂场景的人数统计算法，准确输出图片中的总人数。

<img src = 'https://ftp.bmp.ovh/imgs/2019/11/dca8590110d1df2e.png' />



### 二. 模型介绍

#### 1. 人头检测 or 回归密度图  

比赛数据集提供了两种标注方式    
- 提供bounding box标注（大人头）  
- 提供point标注（小人头）   



检测or 回归?     
- 基于检测的方法，由于有些人头没有给出bounding box标注，因此在密集区域，很难训练检测网络，同时，人头与人头之间还会存在多尺度、遮挡、光照等复杂问题，实际测试时难以检测出人头位置。
- 基于回归的方法，通过回归密度图的方式，能较好的解决密集、遮挡等问题，同时也仅需提供point标注。


#### 2. 生成GroundTruth     
- 统一用点作为标注，对于bounding box标注则取其中心  
- 对每个坐标点，做高斯变换得到密度图  
- 对密度图做积分，得到人头数  

<img src = 'https://ftp.bmp.ovh/imgs/2019/11/572b99fb8df302bd.png' />



#### 3. Learn to Scale  
观察正常人是怎样标人头数据的？   
对于稀疏区域，直接计数或者标注；  
对于密集区域，肯定是先把密集部分放大再计数或者标注；    

<img src = 'https://ftp.bmp.ovh/imgs/2019/11/ba8f06b8bf94187d.png' />



同时，还得考虑放大系数的问题，如果放大过度，导致失真；如果放大过小，导致计数不准确；   



#### 4. L2SM  
<img src = 'https://ftp.bmp.ovh/imgs/2019/11/7fe7bd49b0be3a3c.png' />



#### 5. 本次比赛采用的模型  

<img src = 'https://ftp.bmp.ovh/imgs/2019/11/ba1ff89a1db70a47.png' />

本质在于如何能同时优化稀疏、密集区域


#### 6. Baseline  
<img src = 'https://ftp.bmp.ovh/imgs/2019/11/24b518be5f21508f.png' />



#### 7. 如何寻找密集区域？  
<img src = 'https://ftp.bmp.ovh/imgs/2019/11/dcbffbf294a9457d.png' /> 



#### 8. 总结    
<img src = 'https://ftp.bmp.ovh/imgs/2019/11/617bbccb7d870459.png' />  





### 三. 比赛结果  
<img src = 'https://ftp.bmp.ovh/imgs/2019/11/02327e22b048e8c4.png' />   



### 四. 比赛细节   

- 比赛数据集较大，普遍为1920 * 1080的分辨率，易爆显存，将长边resize至1024，短边按照相同比例resieze。
- 数据增强： 0.8-1.3倍随机缩放、 50%概率水平翻转、 50%概率增加椒盐噪声。实际测试效果： 随机缩放>椒盐噪声>翻转
- 寻找联连通区域阈值为density map均值
- 学习率1e-5，每50 epoch 衰减0.5倍




