

- [[mask-rcnn benchmark pytorch1.0]](https://github.com/facebookresearch/maskrcnn-benchmark)   

- [[pytorch-mask-rcnn pytorch0.3 ]](https://github.com/multimodallearning/pytorch-mask-rcnn)  








摘抄：  
> Our approach efficiently detects objects in an image **while** simultaneously generating a high-quality segmentation mask for each instance. 

>  **Moreover**, Mask R-CNN is easy to generalize to other tasks, e.g., **allowing** us to estimate human poses in the same framework. 

> We show top results in all three tracks of the COCO suite of challenges, **including** instance segmentation, boundingbox object detection, and person keypoint detection. **Without bells and whistles**, Mask R-CNN outperforms all existing, single-model entries on every task, **including** the COCO 2016 challenge winners. 

> Instance segmentation is challenging **because** it requires the correct detection of all objects in an image **while** also precisely segmenting each instance.

> It **therefore** combines elements from the classical computer vision tasks of object detection, **where the goal is to** classify individual objects and localize each using a bounding box, and semantic segmentation, **where the goal is to** classify each pixel into a fixed set of categories without differentiating object instances


> **However**, we show **that** a surprisingly simple, flexible, and fast system can surpass prior **state-of-the-art** instance segmentation results.


> **Additionally**, the mask branch only adds a small computational overhead, **enabling** a fast system and rapid experimentation.   


> We **believe** the fast train and test speeds, **together with** the framework’s flexibility and accuracy, will benefit and ease future research on instance segmentation.  

> **Finally**, we showcase the generality of our framework **via** the task of human pose estimation on the COCO keypoint dataset.  

> We begin **by** briefly reviewing the Faster R-CNN detector.  

> The second stage, **which** is **in essence** Fast R-CNN , extracts features using RoIPool from each candidate box and performs classification and bounding-box regression.   

> **Our definition of L_mask** allows the network to generate masks for every class **without** competition among classes; we **rely on** the dedicated classification branch to predict the class label used to select the output mask.

> We show **by experiments** that this formulation is **key** for good instance segmentation results.

> **Thus**, **unlike** class labels or box offsets **that** are inevitably collapsed into short output vectors by fully-connected (fc) layers **extracting** the spatial structure of masks can be addressed naturally by the pixel-to-pixel correspondence provided by convolutions.   

> This allows each layer in the mask branch to maintain the explicit m × m object spatial layout **without** collapsing it into a vector representation **that** lacks spatial dimensions.     

> To **demonstrate** the generality of our approach, we **instantiate** Mask R-CNN with multiple architectures. 


