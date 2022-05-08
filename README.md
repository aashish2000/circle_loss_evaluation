## Results


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

