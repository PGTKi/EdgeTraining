#### 第二阶段总体计划
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


-------------------------------------------------


#### Week 17 Report
##### Combination with DNN BP

[Acceleration of DNN Backward Propagation by Selective Computation of Gradients](https://dl.acm.org/citation.cfm?id=3317755) 较难于现有融合的框架中。

通过添加buffer，将非0的feature map从memory加入到buffer，在FPGA/ASIC上验证想法（本文用的是DianNao）。

[Faster CNNs with Direct Sparse Convolutions and Guided Pruning](https://arxiv.org/abs/1608.01409)是由Intel所作的、对sparse在Intel CPU上做的支持。其支持的方法为non-structured sparsity。其写法对于我们之后的paper有参考意义。

##### 快手的两篇文章:
1. [ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model](https://arxiv.org/abs/1812.01803)
2. [Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking](https://openreview.net/forum?id=BylBr3C9K7)

将model的能耗在剪枝/训练过程中添加到考虑中。两篇文章想法类似，不过都是通过计算的形式来计算能耗而非实际测量。

不同的是前一篇target channel pruning on pretrained model，后一篇target training from scratch。前一篇用3层神经网络训练预测能耗的model，后一篇定义能耗为计算能耗加访存能耗。

##### Week 18 规划
1. LFY Conv2d实现初步+毕旅
2. XXQ BN重写+Schedule优化+DMAC

##### 在Conv2d层中是否需要创建新kernel

这一段在回答之前的问题6。

在之前实现的版本中，我们是根据sparsity信息的对kernel进行重构，创造一个更紧凑的dense tensor，并复用TVM本来的conv2d实现，来完成实现。

但这样会导致新的内存占用，并不符合我们“减少30%内存占用”的要求。

倘若在conv2d_sparse的data flow之外，添加index flow，以新的格式传输和解析数据，可以较好地避免大规模传递数值为0的无意义的数据。

但根据以上规划，在conv2d内部计算的时候就会产生一定的问题。用于kernel多是以__OIHW__形式存储的，其中 __O__ 的稀疏性独立，__I__ 的稀疏性依赖于feature map，而kernel的index flow只记录__O__的稀疏性，为了避免计算0，需要根据feature map的index flow，避免对应kernel中的计算。这就有两种实现方式：

1. 根据runtime时的index flow的稀疏性，重构一个dense kernel，复用相应的conv2d实现
2. 直接读入，在schedule和compute中跳过相应的计算

在[convnet-burden](<https://github.com/albanie/convnet-burden>)中，我们发现resnet-18的feature map memory为23MB，kernel memory为45MB，重构一个新的kernel（即使它更小）的代价并不小，无法完全满足30%内存减少的要求。因此路线1放弃，只能选择路线2。



##### 实现Conv2D Sparse Op可能需要的步骤
- [ ] NNVM_REGISTER_OP以及对应的Conv2DSparseParam: 主要将index flow加入到layer中
- [ ] nnvm.register_compute和nnvm.register_schedule，进行断言推断和嵌套实现，调用下层TOPI算子
- [ ] tvm.register_compute default实现
- [ ] tvm.register_schedule default实现
- [ ] tvm.register_compute mali实现
- [ ] tvm.register_schedule mali实现



##### 和XXQ讨论之后的若干要点
1. Pooling无法做layer fusion
2. Relay暂无对Dynamic Shape的支持，目前只有一个[Request for Comments](https://github.com/dmlc/tvm/issues/3042)
3. 支持sparsity的element-wise add的代码逻辑。先用scatter的方式找到output_index，构建output_data所需要的计算表达式，在用gather的方式计算出output_data
4. 我的TODO：需要check BN层正常的反向逻辑，support XXQ这周的工作
5. XXQ实现BN的两种技术路线：
    1. 类似BN正向，通过Simplify解构成若干乘法和加法
    2. 类似一般的反向，在TVM中定义算子，整体计算。（遇到的问题是如果有类似Momentum和Adam这类需要知道上一个batch信息的函数，就需要配有Assign和update操作。而Assign和update操作只能在NNVM中实现，TVM不能处理这种信息）

