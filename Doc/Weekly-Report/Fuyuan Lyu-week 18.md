
#### Week 17 Report

##### Combination with DNN BP


[Acceleration of DNN Backward Propagation by Selective Computation of Gradients](https://dl.acm.org/citation.cfm?id=3317755) 较难于现有融合的框架中。

通过添加buffer，将非0的feature map从memory加入到buffer。


[Faster CNNs with Direct Sparse Convolutions and Guided Pruning](https://arxiv.org/abs/1608.01409)是由Intel所作的、对sparse在Intel CPU上做的支持。其支持的方法为non-structured sparsity。其写法对于我们之后的paper有参考意义。

##### 快手的两篇文章:
1. [ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model](https://arxiv.org/abs/1812.01803)
2. [Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking](https://openreview.net/forum?id=BylBr3C9K7)

将model的能耗在剪枝/训练过程中添加到考虑中。两篇文章想法类似，不过都是通过计算的形式来计算能耗而非实际测量。

不同的是前一篇target channel pruning on pretrained model，后一篇target training from scratch。前一篇用3层神经网络训练预测能耗的model，后一篇定义能耗为计算能耗加访存能耗。

##### 第二阶段计划
1. 完成稀疏化data flow的设置
2. 根据上述data flow，高效执行conv操作
3. 部分算子返工，如BN
4. Schedule优化
5. import的问题（优先级低）

##### 第二阶段关于稀疏性可能遇到的6个问题及要点
1. ElementWise 加和乘法要做最坏打算，即50%稀疏度，channel各为4的两个tensor相加，输出index长度为最大可能性，即4；相乘，输出index长度亦为最大可能性，即2。因为我们无法在compile的时候获得相应的信息，因此只好做worst case
2. 对于不同的Op是否需要重写运算

| Op | 是否需要重写 |
| ------- | ------ |
| ReLU | 否 | 
| Pooling | 否 |
| BN | 是 |
| element-wise add/time | 是 |
| dense | 是 |
| conv | 是 |

3. 对于Winograd, im2col, FFT不做支持，因为这些需要算法层面的解释
4. Conv2DParams参数需要添加以下几点：
  - sparsity ratio，需要确定下一步data tensor的大小
  - full length of input/output channel，确保能够还原原来的tensor
5. conv中data和index flow两者分开计算
6. 能否简单复用，作为一个数据量更小的conv层？如果复用的话，需要牵扯构建新的tensor，导致内存的增大

