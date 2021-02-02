http://speech.ee.ntu.edu.tw/~tlkagk/courses_DLHLP20.html




![image](https://user-images.githubusercontent.com/30361513/81050439-91f1e380-8ef2-11ea-8a70-792bb48b1014.png)


## Introduction and Fine-Tune  

- 李宏毅 BERT and its family - Introduction and Fine-tune [[youtube]](https://www.youtube.com/watch?v=1_gRK9EIQpc)   

## What is pre-train model  ?  
- fastText [[github]](https://github.com/facebookresearch/fastText)  

### Pretrain Model  
- LSTM
- Self-Attention layers
- Tree-based model  

![image](https://user-images.githubusercontent.com/30361513/81144506-caeb9000-8fa6-11ea-8f0a-2d8301df5358.png)

![image](https://user-images.githubusercontent.com/30361513/81144494-c32beb80-8fa6-11ea-8f63-55bb42b2397d.png)

#### Smaller model  
- Network Compression
    - Network Pruning
    - Knowledge Distillation
    - Parameter Quantization
    - Architecture Design
Ref: https://youtu.be/dPp8rCAnU_A
Excellent reference:
http://mitchgordon.me/machine/learning/2019/11/18/all-theways-to-compress-BERT.html


## How to fine-tune ?   
NLP tasks  
![image](https://user-images.githubusercontent.com/30361513/81145264-7c3ef580-8fa8-11ea-9ffc-e6ca388eaf80.png)
展开描述这么多组合  


如何fine-tune模型呢?  
有两种常见做法, 一种是fix feature extractor; 另一种是non-fix;  
![image](https://user-images.githubusercontent.com/30361513/81145624-346c9e00-8fa9-11ea-8c81-0fef5275d079.png)
![image](https://user-images.githubusercontent.com/30361513/81145641-3d5d6f80-8fa9-11ea-8a13-313957c13072.png)


但是这样fine-tune的模型往往会非常巨大, 那就变成富人玩的游戏, 穷人怎么分一杯羹?  
![image](https://user-images.githubusercontent.com/30361513/81145774-89a8af80-8fa9-11ea-89ed-6e5cefc39cf7.png)
我们固定住模型, 添加adaptor, 只要学习adaptor的参数.   

Adaptor举例  
![image](https://user-images.githubusercontent.com/30361513/81145892-d2f8ff00-8fa9-11ea-81b2-820e98f772dd.png)
Adaptor结果
![image](https://user-images.githubusercontent.com/30361513/81145918-ddb39400-8fa9-11ea-9274-29875a187526.png)

Pretrain 的魔力  
![image](https://user-images.githubusercontent.com/30361513/81146153-4e5ab080-8faa-11ea-9a8f-13c13b25c2b3.png)




## How to pre-train?  

根据输出的类型不同, 李宏毅老师将NLP任务分为两种:  
第一种是sequence to class 
第二种是 sequence to sequence  
那怎么解决multi-sequence的问题?  
第一种是采用multi-model 每一个model训练一个sequence, 然后合并起来;  
第二种是, 将mutli-sequence合并成一个sequence, 中间采用特殊符号分隔;  
![image](https://user-images.githubusercontent.com/30361513/81080081-ef9e2400-8f22-11ea-89e9-beaab02930c4.png)



### Part-of-Speech (POS) Tagging  
对句子中每个词的词性做tagging, 动词, 名词, 形容词  
![image](https://user-images.githubusercontent.com/30361513/81080320-30963880-8f23-11ea-878f-f72090c1829d.png)

### Word Segmentation  
比如中文来说 ,需要对句子断句;  


### Coreference Resolution 指代消解
比如人名代称, he she it 等  

  
### Summarization
指摘要
 
### Machine Translation
 input: sequence  
output: sequence  

### Grammar error correction
input: sequence
output: sequence  

### Sentiment Classification  
input: sequence  
output: class

### Stance Detection
立场检测
input: two sequence  
output: class 



### Veracity Prediction  
真实性评估  

### NLI (Natural Language Inference)  
![image](https://user-images.githubusercontent.com/30361513/81087063-d6e63c00-8f2b-11ea-9f18-f7fe25ef0187.png)

### Question Answering  
- • Extractive QA: Answer in the document 
 

### Natural Language Understanding (NLU)  

### GLUE  
  

## Transformer   
[[youtube]](https://www.youtube.com/watch?v=ugWDIIOHtPA)  

### self Attention  
[[paper]](https://arxiv.org/abs/1706.03762)  

![image](https://user-images.githubusercontent.com/30361513/82393796-05126280-9a7a-11ea-9332-2c7ace814906.png)
self attention 的每一个token或者输入都生成三个向量：Q, K, V.   
Q的作用是匹配其他的输入，K的作用是被Q匹配，吃入两个向量得到一个attention；V是用来与attention做矩阵相乘后提取的特征。   


![image](https://user-images.githubusercontent.com/30361513/82394077-95e93e00-9a7a-11ea-9c06-698f1685d466.png)
然后拿每一个query q都跟key k做attention； 其中d表示q和k的维度大小，表示一个小的正则化。  


![image](https://user-images.githubusercontent.com/30361513/82394783-83700400-9a7c-11ea-8d90-f4af4b4c9c7b.png)
然后通过softmax生成attention map

![image](https://user-images.githubusercontent.com/30361513/82394834-a0a4d280-9a7c-11ea-8b2c-1296313f0c42.png)
得到了attention map后，乘上V然后累加起来，就能得到输出  
这是一种position independent的考虑方式，不同于RNN 与时间序列相关是一种position relative。  

![image](https://user-images.githubusercontent.com/30361513/82394943-ef526c80-9a7c-11ea-8078-21af3591f541.png)
算出第二个输出
b1, b2, b3, b4是并行输出的，所以可以用matrix来计算，而且matrix在GPU中很容易加速。  
所以接下来看一下self attentition的矩阵形式  
![image](https://user-images.githubusercontent.com/30361513/82395018-2759af80-9a7d-11ea-8b50-3616413cd67e.png)
![image](https://user-images.githubusercontent.com/30361513/82395056-3a6c7f80-9a7d-11ea-8bdb-64e3f53f02dd.png)
![image](https://user-images.githubusercontent.com/30361513/82395109-5b34d500-9a7d-11ea-9e8f-3f7049d3eb38.png)
![image](https://user-images.githubusercontent.com/30361513/82395141-71db2c00-9a7d-11ea-9875-eb21a9f552d3.png)
最终形式如下：  
![image](https://user-images.githubusercontent.com/30361513/82395200-9505db80-9a7d-11ea-9647-844b6241f19a.png)

### multi-head self-attention  
![image](https://user-images.githubusercontent.com/30361513/82395303-d1d1d280-9a7d-11ea-9915-d685f2b687af.png)
multi-head就是每一个输入对应了多个Q, K, V向量；然后每一个head都是相互独立计算的，计算过程跟上面介绍的一致。  


### Positional Encoding  
在self-attention中没有位置信息，论文中的做法是：  
对每一个输入xi，可以加上一个位置向量，这个位置向量可以是one-hot向量。  
![image](https://user-images.githubusercontent.com/30361513/82395563-5cb2cd00-9a7e-11ea-8b3a-b8427cb9a1f5.png)


### seq2seq with attention  
![image](https://user-images.githubusercontent.com/30361513/82395643-8835b780-9a7e-11ea-8801-901530795965.png)



### transformer
![image](https://user-images.githubusercontent.com/30361513/82396339-3e4dd100-9a80-11ea-8072-f2c823f519f8.png)







