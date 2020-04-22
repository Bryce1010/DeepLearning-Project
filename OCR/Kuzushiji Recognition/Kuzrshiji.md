

## Preprocess

- Kuzushiji_PyTorch_Data_preprocessing[[discussion]](https://www.kaggle.com/wakamezake/kuzushiji-pytorch-data-preprocessing)  
- Ben's Preprocessing [[discussion]](https://www.kaggle.com/banzaibanzer/applying-ben-s-preprocessing)  
    - 增强细节
    - 解决不同区域的照明问题
- Denoising + Ben's Preprocessing [[discussion]](https://www.kaggle.com/hanmingliu/denoising-ben-s-preprocessing-better-clarity)  
    作者认为引入Ben's processing会引入 guassian noise, 所以引入了去噪部分;  





## EXP  

### EXP1   1st place solution  [[discussion]](https://www.kaggle.com/c/kuzushiji-recognition/discussion/112788) 

#### dataset
train.csv: 
- image_id
- labels: Unicode character, X, Y , W, H

sample_submission.csv:  
- image_id
- labels: Unicode character, X, Y, W , H


train/test_images.zip

#### Model 
- Strong backbones
- Multi-scale train& test  




#### results
LB scores 0.935 with:
- HRNet w32 
- train scales 512-768  
- test scales [0.5,0.625, 0.75]  

LB scores 0.946 with: 
- HRNet w32  
- train scales 768-1280  
- test scales [0.75, 0.875, 1.0, 1.125, 1.25]  

Ensembling HRNet_w32 and HRNet_w48 results -> 0.950.


- [x] Cascade R-CNN  
- [x] HRNet   
- [ ] torch.distributed



### EXP2 4th place solution  [[discussion]](https://www.kaggle.com/c/kuzushiji-recognition/discussion/114764)  

character detection -> get lines -> line recognition -> postprocessing   

#### Detection   

- [x] Hybrid Task Cascade   


- [ ] ocr detection overview  

#### get lines  
- [ ] CTPN   


#### recognition  
- [ ] CRNN  
- [ ] CTC    

- ocr recognition overview   



### EXP3 7th place solution [[discusssion]](https://www.kaggle.com/c/kuzushiji-recognition/discussion/112899)  

#### Approach  
- Two stage (Detection and Classification)
- Validation: Group Split by book titles.
![](https://ftp.bmp.ovh/imgs/2020/04/a1b354092efa7528.jpg)



#### Detection  
- Architecture: CenterNet (input:512x512, output:128x128)
- Augmentation: Cropping, Brightness, Contrast, Horizontal flip
- TTA(flip) & Ensemble



#### Classification  
- Architecture: Resnet base (input:64x64), aspect ratio and size of object are concatenated at FC layer
- Augmentation: Cropping, Erasing, Brightness, Contrast,
- TTA(Size, Brightness) & Ensemble
- Pseudo labeling
-> Pseudo labeling worked better.


## Ref

- [ ] CornerNet: https://arxiv.org/abs/1808.01244
- [ ] CenterNet(Objects as points) : https://arxiv.org/abs/1904.07850


- [ ] 1st place solution [[discussion]](https://www.kaggle.com/c/kuzushiji-recognition/discussion/112788)
- [ ] 2nd place solution overview: detection + full-page classification [[discussion]](https://www.kaggle.com/c/kuzushiji-recognition/discussion/112712)   
- [ ] 4th place solution  [[discussion]](https://www.kaggle.com/c/kuzushiji-recognition/discussion/114764)  


- [ ] awesome scene text detection and recognition [[github]](https://github.com/hwalsuklee/awesome-deep-text-detection-recognition)
- [ ] Scene Text Detection and Recognition: The Deep Learning Era [[paper]](https://arxiv.org/pdf/1811.04256.pdf)  

- [x] [TPAMI 2016] (CRNN) An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition [[paper]](https://ieeexplore.ieee.org/abstract/document/7801919/) [[crnn torch7/PyTorch]](https://github.com/bgshih/crnn) 
- [ ] 使用TensorFlow Attention API进行OCR文本图像提取 [[blog]](https://lijiancheng0614.github.io/2018/10/02/2018_10_02_TensorFlow-Attention-OCR/) 
- [ ] OCR 文本检测综述
- [ ] awesome-deep-text-detection-recognition [[github]](https://github.com/hwalsuklee/awesome-deep-text-detection-recognition)  
- [ ] OCR project [[github]]


- [ ] ICDAR  
