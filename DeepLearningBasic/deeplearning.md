## 为什么深层神经网络难以训练？  
梯度消失
梯度消失是指通过隐藏层从后向前看，梯度会变的越来越小，说明前面层的学习会显著慢于后面层的学习，所以学习会卡住，除非梯度变大。

​	梯度消失的原因受到多种因素影响，例如学习率的大小，网络参数的初始化，激活函数的边缘效应等。在深层神经网络中，每一个神经元计算得到的梯度都会传递给前一层，较浅层的神经元接收到的梯度受到之前所有层梯度的影响。如果计算得到的梯度值非常小，随着层数增多，求出的梯度更新信息将会以指数形式衰减，就会发生梯度消失

梯度爆炸
在深度网络或循环神经网络（Recurrent Neural Network, RNN）等网络结构中，梯度可在网络更新的过程中不断累积，变成非常大的梯度，导致网络权重值的大幅更新，使得网络不稳定；在极端情况下，权重值甚至会溢出，变为$NaN$值，再也无法更新


权重矩阵的退化导致模型的有效自由度减少。

​	参数空间中学习的退化速度减慢，导致减少了模型的有效维数，网络的可用自由度对学习中梯度范数的贡献不均衡，随着相乘矩阵的数量（即网络深度）的增加，矩阵的乘积变得越来越退化。在有硬饱和边界的非线性网络中（例如 ReLU 网络），随着深度增加，退化过程会变得越来越快

## 常见的激活函数  

### sigmoid激活函数  
![image](https://user-images.githubusercontent.com/30361513/81413490-199c5400-9178-11ea-8e64-f6655bad537b.png)

通常$x=0$时，给定其导数为1和0
### tanh激活函数   
![image](https://user-images.githubusercontent.com/30361513/81413130-83682e00-9177-11ea-9eff-0916867eede3.png)



### ReLU激活函数  
![image](https://user-images.githubusercontent.com/30361513/81413156-8d8a2c80-9177-11ea-9a84-79dbb64f983e.png)




### LeakyReLU激活函数  
![image](https://user-images.githubusercontent.com/30361513/81413196-9b3fb200-9177-11ea-9434-86e623d9d354.png)



### softmax激活函数  
![image](https://user-images.githubusercontent.com/30361513/81413211-a1359300-9177-11ea-9482-c3dacdde9f82.png)


### Mish 激活函数  

![image](https://user-images.githubusercontent.com/30361513/82744956-b46c7380-9db1-11ea-86e3-15a6172a2866.png)

Mish=x * tanh(ln(1+e^x))
![image](https://user-images.githubusercontent.com/30361513/82744958-b7fffa80-9db1-11ea-9c38-43021c7fb4b1.png)

其他的激活函数，ReLU是x = max(0,x)，Swish是x * sigmoid(x)




## Batch Normalization  

![image](https://user-images.githubusercontent.com/30361513/81414060-ffaf4100-9178-11ea-833f-c7f240faaa5d.png)


| 名称                                           | 特点                                                         |
| ---------------------------------------------- | :----------------------------------------------------------- |
| 批量归一化（Batch Normalization，以下简称 BN） | 可让各种网络并行训练。但是，批量维度进行归一化会带来一些问题——批量统计估算不准确导致批量变小时，BN 的误差会迅速增加。在训练大型网络和将特征转移到计算机视觉任务中（包括检测、分割和视频），内存消耗限制了只能使用小批量的 BN。 |
| 群组归一化 Group Normalization (简称 GN)       | GN 将通道分成组，并在每组内计算归一化的均值和方差。GN 的计算与批量大小无关，并且其准确度在各种批量大小下都很稳定。 |
| 比较                                           | 在 ImageNet 上训练的 ResNet-50上，GN 使用批量大小为 2 时的错误率比 BN 的错误率低 10.6％ ;当使用典型的批量时，GN 与 BN 相当，并且优于其他标归一化变体。而且，GN 可以自然地从预训练迁移到微调。在进行 COCO 中的目标检测和分割以及 Kinetics 中的视频分类比赛中，GN 可以胜过其竞争对手，表明 GN 可以在各种任务中有效地取代强大的 BN。 |


