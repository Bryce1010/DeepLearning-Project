





## Top solutions  
### 1st place solution  
- [albumentations](https://github.com/albumentations-team/albumentations)
- [[discussion]](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/discussion/95247)
- [[code]](https://github.com/amirassov/kaggle-imaterialist)
- [[mmdetection Hybrid task cascade Resnext FPN]](https://github.com/open-mmlab/mmdetection/blob/master/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py)  
- [iterative stratification](https://github.com/trent-b/iterative-stratification) 

#### train 
- pre-train from COCO
- optimizer: SGD(lr=0.03, momentum=0.9, weight_decay=0.0001)
- batch_size: 16 = 2 images per gpu x 8 gpus Tesla V100
- learning rate scheduler:
if iterations < 500: lr = warmup(warmup_ratio=1 / 3) if epochs == 10: lr = lr ∗ 0.1 if epochs == 18: lr = lr ∗ 0.1 if epochs > 20: stop
- training time: ~3 days.


#### ensemble
![](https://ftp.bmp.ovh/imgs/2020/04/9f55d59efe9e2ab8.png)

### 3rd place solution

#### Data preprocessing 
We used different sizes for trainings:
minsize: (800, … 960); maxsize <=1600


#### Models 
- facebook repo - Mask-RCNN x-101
- mmdetection repo - Hybrid Task Cascade with X-101-64x4d-FPN backbone and c3-c5 DCN




## Papers 
- [ ] Hybrid Task Cascade for Instance Segmentation  

- [ ] SyncBN





