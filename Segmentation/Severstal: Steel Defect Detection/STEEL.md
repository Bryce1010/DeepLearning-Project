
## Metrics  
- mean Dice coefficient  
![](https://img-blog.csdn.net/20180607161135809?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xlZ2VuZF9odWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![](https://i.bmp.ovh/imgs/2020/04/e26c6cf0e5d79934.png)


## Data  
- train_images 训练图像   
-  test_images 测试图像  
- train.csv 训练集标注, [0,1,2,3]四种分割类别
- sample_submission.csv 分割像素表示采用encoded pixels


## EXP  







## Top solutions  
### 1st place solution 

Classification:
- batchsize: 8 for efficientnet-b1, 16 for resnet34 (both accumulate gradients for 32 samples)
- Optimizer: SGD
- Model Ensemble:
3 x efficientnet-b1+1 x resnet34
- TTA: None, Hflip, Vflip
- Threshold: 0.6,0.6,0.6,0.6


segmentation:
- Train data: 256x512 crop images
- Augmentations: Hflip, Vflip, RandomBrightnessContrast (from albumentations)
- Batchsize: 12 or 24 (both accumulate gradients for 24 samples)
- Optimizer: Rectified Adam
- Models: Unet (efficientnet-b3), FPN (efficientnet-b3
- Loss:
BCE (with posweight = (2.0,2.0,1.0,1.5)) 0.75BCE+0.25DICE (with posweight = (2.0,2.0,1.0,1.5))
- Model Ensemble:
1 x Unet(BCE loss) + 3 x FPN(first trained with BCE loss then finetuned with BCEDice loss) +2 x FPN(BCEloss)+ 3 x Unet from mlcomp+catalyst infer
- TTA: None, Hflip, Vflip
- Label Thresholds: 0.7, 0.7, 0.6, 0.6
- Pixel Thresholds: 0.55,0.55,0.55,0.55
- Postprocessing:
Remove whole mask if total pixel < threshold (600,600,900,2000) + remove small components with size <150
- [ ]  Pesudo label  




### 4th place solution  
- Training two-headed NN for segmentation and classification.
- Combine heads at inference time as with soft gating (mask.sigmoid() * classifier.sigmoid())
- Focal loss / BCE + Focal loss
- Training with grayscale instead of gray-RGB
- FP16 with usage of Catalyst and Apex


### 5th place solution
- inceptionresnetv2, efficientnetv4, seresnext50, se_resnext101, all pretrained on Imagenet
- [ ] Lovász-Softmax loss [[github]](https://github.com/bermanmaxim/LovaszSoftmax)



### 7th place solution 
work:
- mixup and label smoothing
- fine tuning on the full resolution
- random sampler
- cross entropy loss

didn't work:
- SGD
- SWA worked really good for the salt competition, but for this competition it didn't work at all
- pseudo labeling (trained on last 2, 3 days)
- training a classifier
- balanced sampler


### 31pth palce solution  [[code]](https://github.com/Diyago/Severstal-Steel-Defect-Detection)
- Validation: kfold with 10 folds. Despite the shake-up – local, public and private correlated surprisingly good.
- Pseudolabing;

### 36th place solution  [[code]](https://github.com/mobassir94/Severstal-Steel-Defect-Detection-Kaggle)
unetresnet34, unetseresnext5032x4d,
unetmobilenet2, fpnb0, fpnb1, fpnb2, fpnseresnext5032x4d,
fpnresnet34v2

with popular kaggler heng's resnet34 classifier


## Top Notebooks  
- clear mask visualization and simple eda [[notebook]](https://www.kaggle.com/go1dfish/clear-mask-visualization-and-simple-eda)  
- Loss function library [[notebook]](https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch)  
- Mask RCNN - Detailed Starter Code [[notebook]](https://www.kaggle.com/robinteuwens/mask-rcnn-detailed-starter-code)
- [ ]  Semantic Segmentation — Popular Architectures [[medium]](https://towardsdatascience.com/semantic-segmentation-popular-architectures-dff0a75f39d0) 

## Other Top Discussions
- Competition is Finalized - Congrats to our Winners; Takeaways [[discussion]](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114394)  
- Best solutions from Instance Segmentations comps [dicussion]](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/101698)
- segmentation_models.pytorch [[github]](https://github.com/qubvel/segmentation_models.pytorch)



## Papers
- [ ]  Understanding Deep Learning Techniques for Image Segmentation [[paper]](https://arxiv.org/pdf/1907.06119v1.pdf)   
  

## Segmentation  


### train
- FP16 with usage of Catalyst and Apex




### Test  
You can train segmentation on crops and predict on full images
- ensemble 
[[temperature shaping]](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/107716)  

- adversarial validation  [[url]](https://www.kaggle.com/kevinbonnes/adversarial-validation)
> for selecting validation set close to LB test set
> also for selecting pseudo label




