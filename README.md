# CGD
A PyTorch implementation of CGD based on the paper [Combination of Multiple Global Descriptors for Image Retrieval](https://arxiv.org/abs/1903.10663v3).

![Network Architecture image from the paper](results/structure.png)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
- thop
```
pip install thop
```

## Datasets
[CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [CUB200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), 
[Standard Online Products](http://cvgl.stanford.edu/projects/lifted_struct/) and 
[In-shop Clothes](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) are used in this repo.

You should download these datasets by yourself, and extract them into `${data_path}` directory, make sure the dir names are 
`car`, `cub`, `sop` and `isc`. Then run `data_utils.py` to preprocess them.

## Usage
### Train CGD
```
python train.py --feature_dim 512 --gd_config SM
optional arguments:
--data_path                   datasets path [default value is '/home/data']
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub', 'sop', 'isc'])
--crop_type                   crop data or not, it only works for car or cub dataset [default value is 'uncropped'](choices=['uncropped', 'cropped'])
--backbone_type               backbone network type [default value is 'resnet50'](choices=['resnet50', 'resnext50'])
--gd_config                   global descriptors config [default value is 'SG'](choices=['S', 'M', 'G', 'SM', 'MS', 'SG', 'GS', 'MG', 'GM', 'SMG', 'MSG', 'GSM'])
--feature_dim                 feature dim [default value is 1536]
--smoothing                   smoothing value for label smoothing [default value is 0.1]
--temperature                 temperature scaling used in softmax cross-entropy loss [default value is 0.5]
--margin                      margin of m for triplet loss [default value is 0.1]
--recalls                     selected recall [default value is '1,2,4,8']
--batch_size                  train batch size [default value is 128]
--num_epochs                  train epoch number [default value is 20]
```

### Test CGD
```
python test.py --retrieval_num 10
optional arguments:
--query_img_name              query image name [default value is '/home/data/car/uncropped/008055.jpg']
--data_base                   queried database [default value is 'car_uncropped_resnet50_SG_1536_0.1_0.5_0.1_128_data_base.pth']
--retrieval_num               retrieval number [default value is 8]
```

Results


Triplet loss Results

Stanford Cars Dataset (20 epochs):

Using the MG global feature descriptors, we construct an embedding dimension of size 1536. To prevent overfitting on our training dataset, we add L2 Regularization with a weight decay of 1e-8. We tuned the Triplet Loss margin to 0.4 and trained with a batch size of 128. 


We describe our Recall@K performance results below:

Model	                R@1    R@2    R@4    R@8
HTL-Inception.        81.4%  88.0%  92.7%  95.7%
CGD (base)            86.4%  92.1%  95.6%  97.5%
CGD-Reg-1536 (ours)   88.55% 93.63% 95.86% 97.74%



We evaluate Precision@k for k=[1,2,4,8] below:

Model         P@1     P@2     P@4     P@8
CGD-Reg-1536  88.55%  81.88%  73.42%  51.42%


CUB Dataset (20 epochs):

Using the MG global feature descriptors, we construct an embedding dimension of size 1536.  To prevent overfitting on our training dataset, we add L2 Regularization with a weight decay of 1e-8. We tuned the Triplet Loss margin to 0.4 and trained with a batch size of 128. 


We describe our Recall@K performance results below:

Model                  R@1    R@2    R@4    R@8
HTL-Inception          57.1%  68.8%  78.7%  86.5%
CGD (base)             66.0%  76.4%  84.8%  90.7%
CGD-Reg-1536 (ours)    66.9%  77.48% 85.03% 90.56%



We evaluate Precision@k for k=[1,2,4,8] below:

Model                P@1    P@2     P@4     P@8
CGD-Reg-1536         66.9%  61.77%  56.91%  53.33%



Circle loss results

Stanford Cars Dataset (20 epochs):

Using the MG global feature descriptors, we construct an embedding dimension of size 1536.  To prevent overfitting on our training dataset, we add L2 Regularization with a weight decay of 1e-8. We tuned the Circle Loss margin to 0.4, set gamma = 10 and trained with a batch size of 128.


We describe our Recall@K performance results below:

Model                           R@1     R@2    R@4    R@8
CL-Inception                    83.4%   89.8%  94.1%  96.5%
CGD (base)                      86.4%   92.1%  95.6%  97.5%
CGD-Reg-CL-1536 (ours)          84.68%  90.59% 94.21% 96.63%



We evaluate Precision@k for k=[1,2,4,8] below:

Model            P@1     P@2    P@4    P@8
CGD-Reg-CL-1536  84.68%  76.2%  66.67% 58.2%


CUB Dataset (20 epochs):

Using the SMG global feature descriptors, we construct an embedding dimension of size 1536.  To prevent overfitting on our training dataset, we add L2 Regularization with a weight decay of 1e-8. We tuned the Circle Loss margin to 0.4, set gamma = 10 trained with a batch size of 128.


We describe our Recall@K performance results below:

Model               R@1   R@2   R@4   R@8
CL-Inception	       66.7%	77.4%	86.2%	91.2%
CGD (base)	         66.0%	76.4%	84.8%	90.7%
SMG-Reg-1536 (ours)	63.47%	74.30%	82.71%	89.24%
