## 资源
- OCR dataset [[github]](https://github.com/WenmuZhou/OCR_DataSet)  
- awesome-deep-text-detection-recognition [[github]](https://github.com/hwalsuklee/awesome-deep-text-detection-recognition)  
- SceneTextPapers [[github]](https://github.com/Jyouhou/SceneTextPapers)    
- [ ] Scene Text Detection and Recognition: The Deep Learning Era [[paper]](https://paperswithcode.com/paper/scene-text-detection-and-recognition-the-deep)    



- Scene Text Recognition  [[paper with code]](https://paperswithcode.com/task/scene-text-recognition)  

- Scene Text Detection [[papers with code]](https://paperswithcode.com/task/scene-text-detection)  

- Multi-Oriented Scene Text Detection [[papers with code]](https://paperswithcode.com/task/multi-oriented-scene-text-detection)   

- Curved Text Detection [[papers with code]](https://paperswithcode.com/task/curved-text-detection)   


## 综述

- Scene Text Detection and Recognition 旷视北大联合公开课 [[pdf]](https://zsc.github.io/megvii-pku-dl-course/slides/Lecture7(Text%20Detection%20and%20Recognition_20171031).pdf)   
- Irregular Text Detection and Recognition (CBDAR2019 keynote) [[url]](http://122.205.5.5:8071/~xbai/Talk_slice/IrregularText-CBDAR2019.pptx)  
- HCIILAB  [[github]](https://github.com/HCIILAB)    
## Background  
text detection & Recognition的难点 和已得到解决的点
- 文字密集和稀疏
- 多方向
- 多语言混合  
对于拉丁型文字和非拉丁型文字, 相对于英文,汉字往往会有很长的文本行  
检测器往往不能同时对两者做很好的检测结果

针对汉字问题呢, 有关工作提出采用长卷积:  



## Scene Text Recognition  
场景文字识别可以分为, 字符识别或者文本识别, 字符识别可以采用字符分类器, 而文本识别首先要提取sequence feature, 然后采用RNN生成序列结果:  
![image](https://user-images.githubusercontent.com/30361513/81030256-c2b62680-8eba-11ea-972f-0c2b93a89541.png)


- 字符识别的工作主要包含  
[1] M. Jaderberg et al. Reading text in the wild with convolutional neural networks. IJCV, 2016.
- 文本识别的工作主要有
[2] B. Su et al. Accurate scene text recognition based on recurrent neural network. ACCV, 2014.
[3] He et al. Reading Scene Text in Deep Convolutional Sequences. AAAI, 2016.
[4] Shi B et al. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text
recognition. TPAMI, 2017.



根据文字的背景与弯曲分为两种任务  
### Regular Text Recognition  
[1] CRNN: Shi B et al. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. TPAMI, 2017.
![image](https://user-images.githubusercontent.com/30361513/81042461-bfcf2c00-8ee2-11ea-80be-d78dd6f03b6d.png)

CRNN网络其实很简单, 首先是采用CNN提取feature, 由于CNN的感受野有限, 不能关注到长文本信息, 所以将feature放入到RNN中, 通过RNN输出text label , 最后通过后处理将RNN输出的text label连城文本行.   
![image](https://user-images.githubusercontent.com/30361513/81042706-47b53600-8ee3-11ea-85fc-b83535acc25f.png)

![image](https://user-images.githubusercontent.com/30361513/81042736-57cd1580-8ee3-11ea-811d-4b2de1c83b12.png)
这篇文章的优点在于:  
- 第一次实现文本识别端到端的训练
- 不需要字符级别的标注
- lexicon-free



### Irregular Text Recognition  
[2] RARE: Shi B et al. Robust scene text recognition with automatic rectification. CVPR, 2016.
第一个问题就是, 什么叫做Irregular text呢?  
![image](https://user-images.githubusercontent.com/30361513/81043257-84cdf800-8ee4-11ea-9304-e2c8599cce34.png)

RARE网络结构包含了两部分:  
第一部分是STN, 将原来的曲型文字或者透视文字通过STN transform成正常的水平文字;  
![image](https://user-images.githubusercontent.com/30361513/81043835-a7144580-8ee5-11ea-861b-5dfb722105ca.png)
这个思想与STN相似;  

第二部分是SRN, 是一个Encoder-Decoder网络, Encoder是一个CNN+Bi-LSTM 生成在sequence feature, 然后Decoder是Attention + GRU.  
![image](https://user-images.githubusercontent.com/30361513/81044101-3b7ea800-8ee6-11ea-9f4f-e61eec8c3410.png)






## Classic Method  
- Detection: MSER  
- Detection: SWT

- Recognition: Top-Down and Bottom-Up Cues
- Recognition: Tree-Structured Model  

## Scene Text Detection   
我们将场景文字检测 按照方法划分为两个时代:  
### before 2016  
在2016前的文字检测中, 一般采用detection的常用pipeline, proposals - > filtering -> regression

![image](https://user-images.githubusercontent.com/30361513/81030058-fba1cb80-8eb9-11ea-8faf-353c3096c560.png)

采用这一方法的工作主要有:  
[1] Jaderberg et al. Deep features for text spotting. ECCV, 2014.
[2] Jaderberg et al. Reading text in the wild with convolutional neural networks. IJCV, 2016.
[3] Huang et al. Robust scene text detection with convolution neural network induced mser trees. ECCV, 2014.
[4] Zhang et al. Symmetry-based text line detection in natural scenes. CVPPR, 2015.
[5] LGómez, D Karatzas. Textproposals: a text-specific selective search algorithm for word spotting in the wild. Pattern Recognition 70, 60-74 

### after 2016
2016年后, 由于针对inregular text的尝试, 出现了三种主流方法:  
- segmentation-based method  
[1] Zhang Z, et al. Multi-oriented text detection with fully convolutional networks. CVPR, 2016.
- proposal-based method
[2] Gupta A, et al. Synthetic data for text localisation in natural images. CVPR, 2016.
- hybrid method
[3] He W, et al. Deep Direct Regression for Multi-Oriented Scene Text Detection. ICCV, 2017





### segmentation-based 
[2] B. Shi et al. Detecting Oriented Text in Natural Images by Linking Segments. IEEE CVPR, 2017.

![image](https://user-images.githubusercontent.com/30361513/81041886-7f22e300-8ee1-11ea-8f4d-f369397f7347.png)
 
![image](https://user-images.githubusercontent.com/30361513/81041933-9cf04800-8ee1-11ea-936a-5fcdd371496a.png)
不同的网络层, 可以看到采用了不同的detect 尺度, 最后将不同网络层的box都组合起来, 这样就有两种组合方式, 一种是同一网络层, 另一种是不同网络层; 

检测Long text 结果  
![image](https://user-images.githubusercontent.com/30361513/81042109-f8223a80-8ee1-11ea-840d-6f24f2c3b5e4.png)

也可以检测曲型文字
![image](https://user-images.githubusercontent.com/30361513/81042145-0b350a80-8ee2-11ea-82cf-d8760e2ead05.png)

### proposal-based
[1] M. Liao et al. TextBoxes: A Fast Text Detector with a Single Deep Neural Network. AAAI, 2017. [[paper]](https://arxiv.org/abs/1611.06779)  [[caffe]](https://github.com/MhLiao/TextBoxes)

![image](https://user-images.githubusercontent.com/30361513/81039550-f3f31e80-8edb-11ea-8311-20fea4af836d.png)
28 层的全卷积网络, 13层是VGG16, 然后额外添加了9层卷积到VGG16后面, Text-box layers连接了6层卷积层; 每一个map location都输出一个72d的向量, 分别是text presense score (2d) 和 offesets(4d) , 总共输出12个这样的box;  然后采用NMS, aggregate 所有输出的box.  

![image](https://user-images.githubusercontent.com/30361513/81040652-649b3a80-8ede-11ea-8fd3-6599ddb13eba.png)
- SSD backbone  
- Long default boxs 
- Long default kernels


![image](https://user-images.githubusercontent.com/30361513/81040956-1c304c80-8edf-11ea-8595-03ffd8b5b639.png)


我觉得廖博师兄提出的这个text convolution非常的简单, 但是还是需要一定的ocr基础积累才能察觉到的idea, 不过缺点可能是论文中只尝试了1x5的卷积, 但是没有给出为什么这种卷积就是有效的? 那么1x7, 1x8的呢? vertical 文字检测效果不明显, 同时无法适用于inregular的文字场景.  


### Hybrid method  

EAST: An Efficient and Accurate Scene Text Detector [Zhou et al., CVPR 2017]

![image](https://user-images.githubusercontent.com/30361513/81133523-759e8700-8f84-11ea-8409-aa7366f663da.png)

![image](https://user-images.githubusercontent.com/30361513/81133524-76371d80-8f84-11ea-9ca8-259cf9db7071.png)








## End-to-End Scene Text Detection & Recognition

## Datasets and Evaluation  

### ICDAR2015 - Incidental Scene Text dataset 
- Focus on the incidental scene where text may appear in any orientation any location
with small size or low resolution.
- Includes 1000 training images containing about 4500 readable words and 500 testing
images.
![image](https://user-images.githubusercontent.com/30361513/81045486-e1cbad00-8ee8-11ea-98ed-6144620e0524.png)
### MSRA-TD500

- Contains 500 natural images taken from indoor and outdoor.
- Texts in different languages (Chinese, English or mixture of both), fonts, sizes,
colors and orientations.
- Annotated with text line bounding box.
- Ref. Detecting Texts of Arbitrary Orientations in Natural Images, CVPR12
![image](https://user-images.githubusercontent.com/30361513/81045561-04f65c80-8ee9-11ea-9dc1-a85b586f5342.png)
### RCTW-17 dataset
- Chinese Text in the Wild(12,034 images, 8034 images for training and 4000 images for
testing)
- The text annotated in RCTW-17 consists of Chinese characters, digits, and English
characters, with Chinese characters taking the largest portion.
-  ICDAR2017 Competiton on Reading Chinese Scene Text in the Wild (RCTW-17
![image](https://user-images.githubusercontent.com/30361513/81045636-2ce5c000-8ee9-11ea-9a8f-674722728c82.png)

