### 本周目标
- [ ] conv2d_sparse在opencl上运行成功
- [ ] 开始做elementwise-add/multi
- [x] 整理问题
- [x] Prof. Yuan Xie's paper

### 本周进度
#### 问题整理
整理在实践过程中遇到的[问题](https://github.com/acada-sjtu/EdgeTraining/blob/master/Doc/Weekly-Report/Some%20question%20we%20met%20in%20the%20paper.md)

---------------------

#### Conv2d_sparse
In progress， test function finished

##### tvm.extern
Extern实现中不存在compute和schedule的概念，若想要在TVM中实现conv2d_sparse函数，则应该：
1. TVM 中定义conv2d_sparse_extern函数，实现原TVM中Compute和Schedule的功能
2. NNVM 中Compute调用topi.conv2d_sparse_extern函数，Schedule使用默认调用实现（将之当作黑盒处理）

#### Elementwise-add/multi
Freezed

--------------------
#### Prof. Yuan Xie's paper
1. 本文的claim主要在于加速both training and inference process
2. 文章的算法由三部分组成
    1. Dimension-Reduction Search: 主要用top-k搜索方法，预测哪些output neurons是重要的，来做处理。每次并不是prune掉neurons，而是只计算部分
    2. Sparse-Random Projection for efficient dimension reduction search: 数学证明自己的方法work，有好的性质
    3. Double-mask selection for BN compatibility：这是和我们关系最大的部分。做法是把之前1中的mask拿来用一边，来保证sparsity
3. Drawbacks for my point of view
    1. real-speedup test只在Intel MKL上做了，作者在open review中也承认在GPU上实现比较困难
    2. 作者似乎没有考虑到稀疏性的传递。虽然double-mask selection保证了在单iteration中稀疏性没问题，使得反向好算，但这种稀疏性并没有被传递到下一层，用于减少下一层中不必要的计算量。作者对稀疏性的算法依然是每层依据自己的local feature来进行的，而不看network中的global feature。或许与之前一篇文章[Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124)可以集合？
    3. 在GEMM上加速效果相较于VMM差很多（open review中承认GEMM没有实测过，VMM），作者解释他们的加速算法在GEMM上变成了irregular。原因在于structured pruning GEMM省去该行与另一个矩阵中所有的列的计算，而本文对于不同的行会省去不同的列的计算（如行1计算列1、3、5，行2计算猎2、4、6）。因此，或许VMM是GEMM的更细粒度版本？（this needs to be verified）
4. Some other points:
    1. open review中似乎讨论到和Dropout的关系。author claim自己的方法比dropout更有针对性地保留重要neurons
    2. Double-mask selection能否被借鉴到我们的method中？如果在Algorithm level证明可行，在System level，依照我们目前的design应该不难实现
    3. [文章和open review](https://openreview.net/forum?id=H1goBoR9F7)及[3rd-party reimplementation](https://github.com/mtcrawshaw/dynamic-sparse-graph)
