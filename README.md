# Learning_DataScience


## Abstract  
这是一个个人的DataScience代码存储仓库，主要目的如下：
- 存储DataScience的代码模板；
- 存储DataScience的资料；

阅读方式主要如下：  
- 第一遍全篇浏览，注重全文的摘要，标题，总结；  
- 第二遍操作实践，注重全文的细节，并实践操作；  
- 第三遍自我总结，总结技术，生成个人感悟；   



## Prepare
- Python DataScience Book [[book]](https://jakevdp.github.io/PythonDataScienceHandbook/)   
- OpenCV-Python  



## Plan
- Numpy  
- Pandas  
- [ ] Matplotlib  
- MachineLearning  




## Data Augmentation  
- Image Augmentation for Deep Learning using PyTorch — Feature Engineering for Images  [[medium]](https://medium.com/analytics-vidhya/image-augmentation-for-deep-learning-using-pytorch-feature-engineering-for-images-3f4a64122614)     


## Visuaization with Matplotlib  
##### Simple Line Plots

##### Simple Scatter Plots  

##### Visualizing Errors  


##### Density and Contour Plots  


##### Histograms, Binnings, and Density   



##### Customizing Plot Legends    



##### Customizing Colorbars   


##### Multiple Subplots  


##### Text and Annotation   




##### Customizing Ticks   

##### Customizing Matplotlib: Configurations and Stylesheets   


##### Three-Dimensional Plotting in Matplotlib   


##### Geographic Data with Basemap  


##### Visualization with Seaborn  


##### Further Resources   




## Detection  

### MMDetection  







## Conclude
- Final Project




## 3-1-1  
> 此处记录每周看的论文，表示3篇略读+1篇精读+每月1个项目     
> 略读： 大部分略读，看问题，看方法，看角度，拓展视野“   
> 精读： 极少数精读，深入学习不错过每一点细节   
> 不急于深入细节，不在”知道“与”明白“之间徘徊   
> 多思考，多关联，加深，拓展，衍生往往比阅读本身更有价值    



### 2Week


- (项目) 2019 Data Science Bowl [[kaggle]](https://www.kaggle.com/c/data-science-bowl-2019/discussion)  [[project]](./Projects/2019Data_Science_Bowl)   
 







### 1Week  
- (略读1)(2020 TPAMI Detection) Gliding vertex on the horizontal bounding box for multi-oriented object detection [[github]](https://github.com/MingtaoFu/gliding_vertex)   
> This paper propose a new method about how to detect multi-oriented objection.    
> According to Mask-RCNN which output four regression value, the author add another four value to as obliquity offset.     

![](./images/multi_oriented_network.png)


- (略读2) (2019 ICCV Attention) GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond  [[github]](https://github.com/xvjiarui/GCNet)   
> Via absorbing advantages of Non-Local Networks(NLNet) and Squeeze-Excitation Network(SENet), GCNet provides a simple, fast and effecitive approach for global context modeling, which generally ouputperforms both NLNet and SENet on major benchmark for various recognition tasks.   
![](./images/global_context_network.png)  



- (略读3) (2019 NIPS Detection) **DetNAS**: Backbone Search for Object Detection  [[paper]](https://arxiv.org/abs/1903.10979)  
> In this work, the author presenet DetNAS to use Neural Architecture Search for the design of better backbone for object detection. Detector training schedule includes: ImageNet pre-training, detection fine-tuning. and architecture search using the detection task as the guidance.   
![](./images/DetNAS.png)  



- (精读1) (2017 ICCV) Mask R-CNN  [[paper]](http://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)   
> 第一遍，浏览全文，看标题，看简介，看图表，看结论     
> 第二遍，看引用，看实验，看结果   
> 第三遍，看写作技巧，复现，拓展    
    [[Mask-RCNN]](./Segmentation/README.md)       

