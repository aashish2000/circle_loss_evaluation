##  Analyzing Pairwise Loss from a Deep Metric Learning Perspective

 

![image](https://user-images.githubusercontent.com/11198090/167319635-0c975812-0c6b-41ef-b8b3-4b23320256d6.png)
### Figure 1. Image Representation space visualization [Image courtesy: Google]
##



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
are used in this repo.

You should download these datasets by yourself, and extract them into `${data_path}` directory, make sure the dir names are 
`car`, `cub`. Then run `data_utils.py` to preprocess them.

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

https://github.com/aashish2000/circle_loss_evaluation/blob/origin/dev/Results
