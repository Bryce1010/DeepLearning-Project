
## 综述  
-  STOA papers with code [[url]](https://paperswithcode.com/sota/object-detection-on-coco)  
- 【DeepLearning综述】 DeepLearning Detection 综述 [[blog]](https://bryce1010.blog.csdn.net/article/details/103892627)
-   MMDetection: Open MMLab Detection Toolbox and Benchmark [[paper]](https://arxiv.org/pdf/1906.07155.pdf)  
    -  香港中文大学，商汤等联合提出的MMDetection，包括检测模型，实体分割等state-of-art模型框架源码，属业界良心。
- [2019.07] A Survey of Deep Learning-based Object Detection [[paper]](https://arxiv.org/pdf/1907.09408.pdf)  
    - 西安电子科技大学关于目标检测的论文综述。  
- [2019.08]Recent Advances in Deep Learning for Object Detection [[paper]](https://arxiv.org/pdf/1908.03673.pdf)  
    - 新加坡管理大学论文综述。
- [2019.09]Imbalance Problems in Object Detection: A Review  [[paper]](https://arxiv.org/pdf/1909.00169.pdf) [[github]](https://github.com/kemaloksuz/ObjectDetectionImbalance)   
    - 中东科技大学(Middle East Technical University)一篇关于目标检测领域imbalance problem的综述。imbalance problem包括Class imbalance， Scale imbalance，Spatial imbalance， objective imbalance。论文对各个方面进行归纳，提出问题和分析解决方案。
- (2019.05) Object Detection in 20 Years: A Survey  [[paper]](https://arxiv.org/pdf/1905.05055.pdf)    

- (2020.02) DEEP DOMAIN ADAPTIVE OBJECT DETECTION: A SURVEY [[paper]](https://arxiv.org/pdf/2002.06797.pdf)  

- deep learning object detection [[github]](https://github.com/hoya012/deep_learning_object_detection)  
    > A paper list of object detection using deep learning.
## 书籍  
- [ ] 视觉计算基础：计算机视觉、图形学和图像处理的核心概念 
- [ ] 计算机视觉――一种现代方法（第二版）
- [ ] 数字图像目标检测与识别―理论与实践 [Object Detection and Recognition in Digital Images]    
- [ ] 目标检测好文分享 [[zihu]](https://zhuanlan.zhihu.com/p/140036646)   




## Two-Stage
- [ ] (2020 CVPR ) EfficientDet: Scalable and Efficient Object Detection [[paper]](https://arxiv.org/pdf/1911.09070.pdf)  
- [ ] (2020 CVPR) 1st Place Solutions for OpenImage2019 - Object Detection and Instance Segmentation [[paper]](https://arxiv.org/pdf/2003.07557.pdf)     
- [ ] (2020 CVPR)TridenNet
- [ ] (2020 AAAI) CBNet: A Novel Composite Backbone Network Architecture for Object Detection [[paper]](https://arxiv.org/abs/1909.03625)   
- [ ] (2019 ICCV) Hybrid Task Cascade for Instance Segmentation [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Hybrid_Task_Cascade_for_Instance_Segmentation_CVPR_2019_paper.pdf)  
- [ ] (2019 ICCV)GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond [[paper]](https://arxiv.org/abs/1904.11492)
- [ ] (2019 ICCV) Scale-Aware Trident Networks for Object Detection [[paper]](https://arxiv.org/abs/1901.01892)    
- [ ] (2019 ICCV)ThunderNet: Towards Real-time Generic Object Detection [[paper]](https://arxiv.org/abs/1903.11752)  
- [ ] (2019) Cascade R-CNN: High Quality Object Detection and Instance Segmentation [[paper]](https://arxiv.org/abs/1906.09756)   
- [x] (2018 CVPR)Cascade R-CNN: Delving into High Quality Object Detection [[paper]](https://arxiv.org/abs/1712.00726)    [[mmdetection]](https://github.com/open-mmlab/mmdetection)
- [ ] (2018.12) Deformable ConvNets v2: More Deformable, Better Results [[paper]](https://arxiv.org/abs/1811.11168)
- [x] (2017.03 ICCV) Deformable Convolutional Networks [[paper]](https://arxiv.org/abs/1703.06211)
- [x] (2015 NIPS) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks [[paper]](https://arxiv.org/abs/1506.01497)  

### (2015 NIPS)Faster R-CNN
![image](https://user-images.githubusercontent.com/30361513/81698518-b4f33900-9498-11ea-887b-e861a4fbe1e0.png)
Faster R-CNN的改进在于图像生成feature map后， 使用一个网络RPN自动生成proposals， 而不是使用slective search， 这样可以减少大量生成proposals的时间。


### (2017 NIPS) R-FCN: Object Detection via Region-based Fully Convolutional Networks

整体来说，本文在FCN网络中使用一个位置敏感的RoI pooling层，得到一个“位置敏感度图”作为输出，完成一个端到端的目标检测网络结构，其主要网络流程如下图：
![image](https://user-images.githubusercontent.com/30361513/81698279-50d07500-9498-11ea-942c-a86f0099f80d.png)

图中可以清楚看出，整个R-FCN网络依旧是采用RPN+detection两个部分，分别进行候选proposal提取和检测。RPN类似于原始设计，进行前景背景的分离，而在R-FCN的结尾连接着RoI pooling层，该层产生对应于每一个RoI区域的分数。
在R-FCN的后面阶段里，所有卷积权值共享。和fast rcnn相比，主要差别就在后面跟的是ResNet，ResNet101有100个卷积层，一个pooling层一个1000类的fc层，本文为了应用在目标检测，将pooling层和fc层去除，只保留其卷积层得到的feature map，进一步产生分数图进行检测。

![image](https://user-images.githubusercontent.com/30361513/81698313-62198180-9498-11ea-8a39-b6f160906a64.png)

- OHEM
- 空洞卷积


### (2019 CVPR) Libra R-CNN Towards Balanced Learning for Object Detection   
在检测过程中，作者对在CNN中连续卷积不同尺寸的feature map划分成三个层次：sample level, feature level, objective level. 并提出 Libra R-CNN 来对object detection也就是上面三个level进行平衡。其中， Libra R-CNN 集称了三个部件：

- IoU-balanced sampling
用于减少样本数（sample）
- balanced feature pyramid
用于减少特征数（feature）
- balanced L1 loss
用于减少目标水平的不平衡（objective level）
Libra R-CNN在没有bells和whistles的情况下，在MSCOCO上分别比FPN和RetinaNet在AP(平均精度)上提高了2.5和2.0 points
![image](https://user-images.githubusercontent.com/30361513/81705372-8ded3500-94a1-11ea-9b39-a63d8084f543.png)

**Balanced IoU sampling**
![image](https://user-images.githubusercontent.com/30361513/81705486-aa896d00-94a1-11ea-8b08-51fd4adc946e.png)

**Balanced feature pyramid**
![image](https://user-images.githubusercontent.com/30361513/81705496-ad845d80-94a1-11ea-93ed-0c8b84e8b9a8.png)
将AVG的feature经过 non-local network 或者 conv 再次得到和之前各层的相同尺寸4个的feature {P2, P3, P4, P5}
**Balanced L1 loss**
- outliers：sample(采样)到的gradient(梯度)大于等于1的样本，可视为 hard sample，不易于训练
- inliers：sample(采样)到的gradient(梯度)小于1的样本，可视为 easy sample，对训练较友好
![image](https://user-images.githubusercontent.com/30361513/81705655-dad10b80-94a1-11ea-931c-005434d377f8.png)
~

## One-Stage  
- [x] (2020.04) YOLO V4: Optimal Speed and Accuracy of Object Detection [[paper]](https://arxiv.org/abs/2004.10934)   
    > Input-> Backbone -> Neck -> Dense Prediction -> Sparse Prediction  
    > Bags of freebies & Bags of specials    
    > (CSPDarknet) + (SPP+ PAN) + YOLO v3
- [ ] (2019.04) FCOS: Fully Convolutional One-Stage Object Detection [[paper]](https://arxiv.org/pdf/1904.01355.pdf)    
- [ ] (2019.04) HAR-Net: Joint Learning of Hybrid Attention for Single-stage Object Detection  [[paper]](https://arxiv.org/pdf/1904.11141.pdf)
- [ ] (2019.04) CenterNet: Object Detection with Keypoint Triplets [[paper]](https://arxiv.org/pdf/1904.08189.pdf)   



### (2016 CVPR) YOLO — You only look once, real time object detection
![image](https://user-images.githubusercontent.com/30361513/81757627-13003a80-94f2-11ea-82eb-9a2ab3dc92aa.png)
YOLO的思想是one-stage， 不需要提前生成proposal， 而是将图片划分为S*S的网格， 然后每一个网格取bounding box， 最后将取到的bounding box用于网络最后的classification和regression。

### (2016 ECCV) SSD: Single Shot MultiBox Detector 
![image](https://user-images.githubusercontent.com/30361513/81758289-e6e5b900-94f3-11ea-8bfc-87cb07fa799b.png)
- 图像经过CNN以后，会生成一个mxnxp的feature maps， 比如上图中的8x8, 4x4的feature， 再后面接一个3x3的conv；   
- 对于feature的每一个location，生成k个bounding box，有不同的尺寸和比例
- 每一个bounding box，计算c 个类别的分数，和4个offset
- 最后输出(c+4)kmn个数值  

![image](https://user-images.githubusercontent.com/30361513/81758533-b05c6e00-94f4-11ea-85ee-cf6ecaa89b72.png)


- Hard Negative Mining  
- Data Augmentation  
- Atrous Convolution空洞卷积

### (2017 CVPR) YOLO9000: Better, Faster, Stronger   

- Batch normalization
Add batch normalization in convolution layers. This removes the need for dropouts and pushes mAP up 2%.  

- High-resolution classifier  

- Convolutional with Anchor Boxes
![image](https://user-images.githubusercontent.com/30361513/81758940-d6ced900-94f5-11ea-8b50-bb9c3e7b4a22.png)


### YOLO V3
- Class Prediction
YOLOv3 replaces the softmax function with independent logistic classifiers to calculate the likeliness of the input belongs to a specific label. Instead of using mean square error in calculating the classification loss, YOLOv3 uses binary cross-entropy loss for each label.

- Bounding box prediction & cost function calculation

- Feature Pyramid Networks (FPN) like Feature Pyramid  


- Feature extractor  
A new 53-layer Darknet-53 is used to replace the Darknet-19 as the feature extractor.
Darknet-53 mainly compose of 3 × 3 and 1× 1 filters with skip connections like the residual network in ResNet. Darknet-53 has less BFLOP (billion floating point operations) than ResNet-152, but achieves the same classification accuracy at 2x faster。
 
![image](https://user-images.githubusercontent.com/30361513/81759096-52308a80-94f6-11ea-938a-ef000ad3c6a3.png)



### (2017 ICCV Best Student Paper Award)  RetinaNet — Focal Loss   
two-stage可以使用OHEM增加hard example的作用， 但是one-stage没有相似应用。  
- CE loss
![image](https://user-images.githubusercontent.com/30361513/81759302-eac70a80-94f6-11ea-9c32-e4ee352be1ab.png)
![image](https://user-images.githubusercontent.com/30361513/81759327-f9adbd00-94f6-11ea-8032-bedfc0be066d.png)

- α-Balanced CE Loss
![image](https://user-images.githubusercontent.com/30361513/81759340-04685200-94f7-11ea-9efe-8323a86e294f.png)
这种只能解决样本不平衡的问题， 无法解决困难样本的问题。  

- Focal loss
![image](https://user-images.githubusercontent.com/30361513/81759403-3083d300-94f7-11ea-9aa0-4fb6f57fedcb.png)

- α-Balanced Variant of FL  
![image](https://user-images.githubusercontent.com/30361513/81759425-45f8fd00-94f7-11ea-929c-0669fda27c77.png)

![image](https://user-images.githubusercontent.com/30361513/81759444-53ae8280-94f7-11ea-8b99-80fe7959d52f.png)

## Tricks  


![image](https://user-images.githubusercontent.com/30361513/81765108-95dec080-9505-11ea-88b9-723e451e926c.png)



![image](https://user-images.githubusercontent.com/30361513/81765116-9a0ade00-9505-11ea-9e96-5c44abd369b7.png)




## neck
additional blocks
- [ ] SPP 
- [ ] ASPP  
- [ ] RFB  
- [ ] SAM  


path-aggregation blocks  
- [ ] FPN  
- [ ] PAN  
- [ ] NAS-FPN  
- [ ] Fully-connected FPN  
- [ ] Bi-FPN   
- [ ] ASFF  
- [ ] SFAM  


## 目标检测比赛中的tricks  
https://zhuanlan.zhihu.com/p/102817180  

### 1. 数据增强  
引入albumentations数据增强库进行增强

MMDetection自带数据增强  

Bbox增强  


### 2. Multi-scale Training/ Testing 多尺度训练或测试  
[[MMdetection中文文档-4. 技术细节]](https://zhuanlan.zhihu.com/p/101222759)   
输入图片的尺寸对检测模型的性能影响相当明显，事实上，多尺度是提升精度最明显的技巧之一。在基础网络部分常常会生成比原图小数十倍的特征图，导致小物体的特征描述不容易被检测网络捕捉。通过输入更大、更多尺寸的图片进行训练，能够在一定程度上提高检测模型对物体大小的鲁棒性，仅在测试阶段引入多尺度，也可享受大尺寸和多尺寸带来的增益。

multi-scale training/testing最早见于[[Spatial Pyramid Pooling]](https://arxiv.org/abs/1406.4729)，训练时，预先定义几个固定的尺度，每个epoch随机选择一个尺度进行训练。测试时，生成几个不同尺度的feature map，对每个Region Proposal，在不同的feature map上也有不同的尺度，我们选择最接近某一固定尺寸（即检测头部的输入尺寸）的Region Proposal作为后续的输入。在[[Convolutional Feature Maps]](https://arxiv.org/abs/1504.06066)中，选择单一尺度的方式被Maxout（element-wise max，逐元素取最大）取代：随机选两个相邻尺度，经过Pooling后使用Maxout进行合并，如下图所示。  

![image](https://user-images.githubusercontent.com/30361513/82113544-617b3680-9789-11ea-8801-b263bfd79ed0.png)


  
### 3.Global Context 全局语境  
这一技巧在[[ResNet的工作]](https://arxiv.org/abs/1512.03385) 中提出，做法是把整张图片作为一个RoI，对其进行RoI Pooling并将得到的feature vector拼接于每个RoI的feature vector上，作为一种辅助信息传入之后的R-CNN子网络。目前，也有把相邻尺度上的RoI互相作为context共同传入的做法。



### 4. Box Refinement/ Voting 预测框微调法/投票法/ 模型融合
微调法和投票法由工作[[Object detection via a multi-region & semantic segmentation-aware CNN model]](https://arxiv.org/abs/1505.01749)  提出，前者也被称为Iterative Localization。


微调法最初是在SS算法得到的Region Proposal基础上用检测头部进行多次迭代得到一系列box，在ResNet的工作中，作者将输入R-CNN子网络的Region Proposal和R-CNN子网络得到的预测框共同进行NMS（见下面小节）后处理，最后，把跟NMS筛选所得预测框的IoU超过一定阈值的预测框进行按其分数加权的平均，得到最后的预测结果。

投票法可以理解为以顶尖筛选出一流，再用一流的结果进行加权投票决策。

不同的训练策略，不同的 epoch 预测的结果，使用 NMS 来融合，或者soft_nms

需要调整的参数：

- box voting 的阈值。
- 不同的输入中这个框至少出现了几次来允许它输出。
- 得分的阈值，一个目标框的得分低于这个阈值的时候，就删掉这个目标框。

模型融合主要分为两种情况：

1. 单个模型的不同epoch进行融合
这里主要是在nms之前，对于不同模型预测出来的结果，根据score来排序再做nms操作。

2. 多个模型的融合

这里是指不同的方法，比如说faster rcnn与retinanet的融合，可以有两种情况：

a) 取并集，防止漏检。

b) 取交集，防止误检，提高精度。



### 5. 随机权值平均 (Stochastic Weight Averaging, SWA)  
[[Stochastic Weight Averaging (SWA) github]](https://github.com/timgaripov/swa)    
随机权值平均只需快速集合集成的一小部分算力，就可以接近其表现。SWA 可以用在任意架构和数据集上，都会有不错的表现。根据论文中的实验，SWA 可以得到我之前提到过的更宽的极小值。在经典认知下，SWA 不算集成，因为在训练的最终阶段你只得到一个模型，但它的表现超过了快照集成，接近 FGE（多个模型取平均）。  

![image](https://user-images.githubusercontent.com/30361513/82114578-1e709180-9790-11ea-862a-bd592a7c0a69.png)
左图:W1、W2、W3分别代表3个独立训练的网络，Wswa为其平均值。中图：WSWA 在测试集上的表现超越了SGD。右图：WSWA 在训练时的损失比SGD要高。  

结合 WSWA 在测试集上优于 SGD 的表现，这意味着尽管 WSWA 训练时的损失较高，它的泛化性更好。

SWA 的直觉来自以下由经验得到的观察：每个学习率周期得到的局部极小值倾向于堆积在损失平面的低损失值区域的边缘（上图左侧的图形中，褐色区域误差较低，点W1、W2、3分别表示3个独立训练的网络，位于褐色区域的边缘）。对这些点取平均值，可能得到一个宽阔的泛化解，其损失更低（上图左侧图形中的 WSWA）。


### 6. OHEM 在线难例挖掘 
[[OHEM]](https://arxiv.org/pdf/1604.03540.pdf)(Online Hard negative Example Mining，在线难例挖掘) 。  
两阶段检测模型中，提出的RoI Proposal在输入R-CNN子网络前，我们有机会对正负样本（背景类和前景类）的比例进行调整。通常，背景类的RoI Proposal个数要远远多于前景类，Fast R-CNN的处理方式是随机对两种样本进行上采样和下采样，以使每一batch的正负样本比例保持在1:3，这一做法缓解了类别比例不均衡的问题，是两阶段方法相比单阶段方法具有优势的地方，也被后来的大多数工作沿用。

![image](https://user-images.githubusercontent.com/30361513/82114653-abb3e600-9790-11ea-9369-e53fd09f7323.png)
作者将OHEM应用在Fast R-CNN的网络结构，如上图，这里包含两个RoI network，上面一个RoI network是只读的，为所有的RoI 在前向传递的时候分配空间，下面一个RoI network则同时为前向和后向分配空间。在OHEM的工作中，作者提出用R-CNN子网络对RoI Proposal预测的分数来决定每个batch选用的样本。这样，输入R-CNN子网络的RoI Proposal总为其表现不好的样本，提高了监督学习的效率。

首先，RoI 经过RoI plooling层生成feature map，然后进入只读的RoI network得到所有RoI 的loss；然后是hard RoI sampler结构根据损失排序选出hard example，并把这些hard example作为下面那个RoI network的输入。

实际训练的时候，每个mini-batch包含N个图像，共|R|个RoI ，也就是每张图像包含|R|/N个RoI 。经过hard RoI sampler筛选后得到B个hard example。作者在文中采用N=2，|R|=4000，B=128。 另外关于正负样本的选择：当一个RoI 和一个ground truth的IoU大于0.5，则为正样本；当一个RoI 和所有ground truth的IoU的最大值小于0.5时为负样本。

总结来说，对于给定图像，经过selective search RoIs，同样计算出卷积特征图。但是在绿色部分的（a）中，一个只读的RoI网络对特征图和所有RoI进行前向传播，然后Hard RoI module利用这些RoI的loss选择B个样本。在红色部分（b）中，这些选择出的样本（hard examples）进入RoI网络，进一步进行前向和后向传播。


### 7. Soft NMS 软化非极大抑制
在传统的NMS中，跟最高预测分数预测框重合度超出一定阈值的预测框会被直接舍弃，作者认为这样不利于相邻物体的检测。提出的改进方法是根据IoU将预测框的预测分数进行惩罚，最后再按分数过滤。配合Deformable Convnets（将在之后的文章介绍），Soft NMS在MS COCO上取得了当时最佳的表现。算法改进如下
![image](https://user-images.githubusercontent.com/30361513/82114693-ed449100-9790-11ea-9bd7-acce4a1bffb7.png)

上图中的[公式]即为软化函数，通常取线性或高斯函数，后者效果稍好一些。当然，在享受这一增益的同时，Soft-NMS也引入了一些超参，对不同的数据集需要试探以确定最佳配置。

```python
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
      rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)   # max_per_img表示最终输出的det bbox数量
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.001)            # soft_nms参数
)
```

### 8. ROIAlign ROI对齐  
RoIAlign是Mask R-CNN（[7]）的工作中提出的，针对的问题是RoI在进行Pooling时有不同程度的取整，这影响了实例分割中mask损失的计算。文章采用双线性插值的方法将RoI的表示精细化，并带来了较为明显的性能提升。这一技巧也被后来的一些工作（如light-head R-CNN）沿用。



- [ ] ATSS  


### NMS算法过程  
（1）将所有框的得分排序，选中最高分及其对应的框
（2）遍历其余的框，如果和当前最高分框的重叠面积(IOU)大于一定阈值，我们就将框删除。
（3）从未处理的框中继续选一个得分最高的，重复上述过程。
![image](https://user-images.githubusercontent.com/30361513/82114894-fb46e180-9791-11ea-81bc-276d911ee113.png)

```python
# python3
import numpy as np

def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep

# test
if __name__ == "__main__":
    dets = np.array([[30, 20, 230, 200, 1], 
                     [50, 50, 260, 220, 0.9],
                     [210, 30, 420, 5, 0.8],
                     [430, 280, 460, 360, 0.7]])
    thresh = 0.35
    keep_dets = py_nms(dets, thresh)
    print(keep_dets)
    print(dets[keep_dets])


```







