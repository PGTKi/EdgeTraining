### 本周进度
- Some paper
- BN alghorithm
- conv2d_sparse in TVM

#### Papers
[Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124)
将BN中的gamma与channel pruning结合，对网络整体的channel数量进行压缩，数学上论证了一些直观方法的不可行性

__可参考的思想：__
算法层面优化需要整体考虑整个网络

#### BN Algorithm
期望能够参考王艺星同学的方法，进行一些验证：
1. BN算法中的beta能够和channel一样减去
2. BN中pruning后mean和var是否需要更新，对acc是否会有影响

#### conv2d_sparse in TVM
- NNVM层面大体完成
- TVM层面：
    - LLVM上报错FCmpInst。猜测原因是return的是tensor list而非tensor
    - LLVM extern compute实现已经完成，default schdule报错FCmpInst
    - OpenCL extern compute可以参考
    - OpenCL extern schedule如何实现正在调研
    
--------------------

### 下周完成
- conv2d_sparse在TVM完成
- Element-wise Add/Multi layer 尽量开始写
- BN期望等一下Algorithm层面的结论
- 整理在实现过程中遇到和解决的问题
