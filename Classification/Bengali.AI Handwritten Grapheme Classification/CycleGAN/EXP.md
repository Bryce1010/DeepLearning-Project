# Bengali.AI Handwritten Grapheme Classification 1st Place Solution --- Cyclegan Based Zero Shot Learning  

第一名的工作真的是Impressive , 我还是第一次见到GAN应用到数据增强方向, 严格来讲也不算是是数据增强;   

## Data
比赛的任务是对孟加拉语的手写字进行识别;
孟加拉语由三个部分组成: 168\*字根( grapheme root), 11\*元音 (vowel diacritic), 7\*辅音 (consonant diacritic)  
每张手写图片的大小是137 x 236 ; 
在训练集中的大小是: 200840 
Public A的大小是: 36
其中, Private test的数据中有可能出现train中没有出现过的手写字, 但是168个字根,11个元音,7个辅音在训练集中全部出现过了;   
所以这有一点zero-shot的感觉.    




## Model  

思路最重要的依据在于, 将测试的数据分为两部分: 一部分为见过的数据(Seen), 一部分为没见过的数据(Unseen);  
- 那么怎么将数据划分为见过和还是没见过的呢?   
作者通过设置一个Out of distribution CNN模型,输出1295个置信度, 如果所有的置信度都低于设定的超参数阈值, 那么认定为Unseen; 如果至少有一个置信度高于设置的阈值, 那么判断为Seen;  
参数设置如下: 
- No resize and crop
- Preprocess --- AutoAugment Policy for SVHN (https://github.com/DeepVoltaire/AutoAugment)
- CNN --- EfficientNet-b7(ImageNet Pretrained)
- Optimizer --- torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False) use defaule value
- LRScheduler --- WarmUpAndLinearDecay
```python
def warmup_linear_decay(step):
    if step < WARM_UP_STEP:
        return step/WARM_UP_STEP
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)
```
- Output Lyaer --- LayerNorm-FC(2560 -> 1295)-BCEWithLogitsLoss + OHEM?
```python
        out = model(image)
        sig_out = out.sigmoid()
        loss = criterion(out, one_hot_label)
        train_loss_position = (1-one_hot_label)*(sig_out.detach() > 0.1) + one_hot_label
        loss = ((loss*train_loss_position).sum(dim=1)/train_loss_position.sum(dim=1)).mean()
```
- Epoch --- 200
- Batch size --- 32
- dataset split --- 1:0
- single fold
- Machine Resource --- 1 Tesla V100 6 days

![inbox_4114579_5231a0ab7a0bf93611fb6fd3d72c6295_1.png](https://ws1.sinaimg.cn/large/006rhxrOgy1gdj50qhiccj31z414011f.jpg)


- 对于Seen的数据, 怎么做识别呢?   
采用比赛中的训练集, 训练一个efficientnet_b7分类器;  
参数设置如下:  
- No resize and crop
- Preprocess --- AutoAugment Policy for SVHN (https://github.com/DeepVoltaire/AutoAugment)
- CNN --- EfficientNet-b7(ImageNet Pretrained)
- Optimizer --- torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False) use defaule value
- LRScheduler --- WarmUpAndLinearDecay
```python
def warmup_linear_decay(step):
    if step < WARM_UP_STEP:
        return step/WARM_UP_STEP
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)
```
- Output Lyaer --- LayerNorm-FC(2560 -> 14784)-SoftmaxCrossEntropy
- Epoch --- 200
- Batch size --- 32
- dataset split --- 9:1 random split
- single fold
- Machine Resource --- 1 Tesla V100 6 days





- 对于Unseen的数据, 怎么做识别呢?  
这就是本文作者的厉害之处了, 我们并没有Unseen的数据, 自然就没法训练一个Unseen的分类器; 
但是孟加拉语手写字一般是由 字根, 元音, 辅音三个排列组合起来 (不过这里作者也说了, 虽然随机排列组合可能会创造不存在的文字出来,但是由于重新制作的代价巨大,所以作者要也只好使用了随机排列生成的数据)  
这一方法通过cyclegan来生成Unseen 数据, 训练一个Unseen 分类器;
这里作者为了稳定性,特定训练了两个Efficientnet_b0作为Ensemble; 

学习此模型分为两个阶段。
第一步是训练从ttf文件合成的图像的分类器。
第二步是训练生成器，该生成器将手写字符转换为所需的合成数据状图像。
要进行这些学习，首先选择一个ttf并生成一个综合数据集

### Font Classifier Pre-training  
- crop and resize to 224x224
- preprocess --- random affine, random rotate, random crop, cutout
- CNN --- EfficientNet-b0
- Optimizer --- torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False) use defaule value
- LRScheduler --- LinearDecay
```python
WARM_UP_STEP = train_steps*0.5

def warmup_linear_decay(step):
    if step < WARM_UP_STEP:
        return 1.0
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)
```
- Output Lyaer --- LayerNorm-FC(2560 -> 14784)-SoftmaxCrossEntropy
- Epoch --- 60
- Batch size --- 32
- Machine Resource --- 1 Tesla V100 4 hours

### CycleGan training
![inbox_4114579_ac780d7f0e17487c7ac2338346da5e93_3.png](https://ws1.sinaimg.cn/large/006rhxrOgy1gdj59wkvcuj31z4140gs9.jpg)

- crop and resize to 224x224
- preprocess --- random affine, random rotate, random crop (smaller than pre-training one), and no cutout
- Model --- Based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix + Pre-trained Font Classifier(fixed parameter and eval mode)
- Optimzer --- torch.optim.Adam(params, lr=0.0002, betas=(0.5, 0.999))
- LRScheduler --- LinearDecay

```python
WARM_UP_STEP = train_steps*0.5

def warmup_linear_decay(step):
    if step < WARM_UP_STEP:
        return 1.0
    else:
        return (train_steps-step)/(train_steps-WARM_UP_STEP)
generator_scheduler = torch.optim.lr_scheduler.LambdaLR(generator_optimizer, warmup_linear_decay)
discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(discriminator_optimizer, warmup_linear_decay)
```
- Epoch --- 40
- Batch size --- 32
- Machine Resource --- 4 Tesla V100 2.5 days
- HyperParameter --- lambdaconsistency=10, lambdacls=1.0~5.0


## Inf or Result  

|Local CV Score for Unseen Class	| Local CV Score for Seen Class | 
|--------------------------------|-------------------------------|
|0.8804|	0.9377|


## Conclude  
这次的比赛, 有些许遗憾, 我的最好模型应该是银牌的样子, 但是也不遗憾, 这只能说明我距离银牌的水平很有很大的差距;通过这样的比赛, 虽然这只是一个分类比赛,但是我还是受益匪浅.   
真的拖了很久才来学习这位大佬的操作;
首先是将数据分成Seen 和 Unseen很重要, 因为测试集中包含的Unseen还是挺多的, 最后导致了很严重的shake, 第一次全身投入的比赛就遇见了剧烈shake真是有意思;
其次, 能够想到通过CycleGan来训练Unseen模型也是很强的, 让我开了眼界了.  在大多数人都为了过拟合训练集的情况下,这位大佬闷声发大财, 默默的就把第一收入囊中, 确实厉害.  



## ref
- first soulution [[discussion]](https://www.kaggle.com/c/bengaliai-cv19/discussion/135984)


- my conclude [[github]](https://github.com/Bryce1010/DeepLearning-Project/tree/master/Classification/Bengali.AI%20Handwritten%20Grapheme%20Classification)

