
#### Week 17 Report

##### Combination with DNN BP


[Acceleration of DNN Backward Propagation by Selective Computation of Gradients](https://dl.acm.org/citation.cfm?id=3317755) 较难于现有融合的框架中。

通过添加buffer，将非0的feature map从memory加入到buffer。


[Faster CNNs with Direct Sparse Convolutions and Guided Pruning](https://arxiv.org/abs/1608.01409)是由Intel所作的、对sparse在Intel CPU上做的支持。其支持的方法为non-structured sparsity。其写法对于我们之后的paper有参考意义。

快手的两篇文章:
1. [ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model](https://arxiv.org/abs/1812.01803)
2. [Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking](https://openreview.net/forum?id=BylBr3C9K7)

将model的能耗在剪枝/训练过程中添加到考虑中。两篇文章想法类似，不过都是通过计算的形式来计算能耗而非实际测量。

不同的是前一篇target channel pruning on pretrained model，后一篇target training from scratch。前一篇用3层神经网络训练预测能耗的model，后一篇定义能耗为计算能耗加访存能耗。



