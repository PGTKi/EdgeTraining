#### 本周的工作
- 写论文
- 完成初步闭环

#### 至小组答辩前的工作安排
- 在mali上运行，需要@xuxiaoqiao帮助
- 完成profiling测试，需要@xuxiaoqiao帮助
- 完成反向时间的测试
- 完成前向精度的测试，resnet-18 model

#### 当前工作
以sparsity tensor而非sparsity index的形式控制矩阵的运算

---------------------------------

#### 和蒋老师讨论之后的工作
1. 首先ad-hoc的方法，以dense的形式输入weight，确保有一个不错的training和inference结果
2. 以下二选一：
    - 以sparse tensor的形式输入到kernel中，确保对所有结构化稀疏算法的支持。最好能做到在dense kernel尺寸不变的情况下，以batch为单位更改数据（这一batch 13579，下一batch 246810）
    - 更focus在layer之间的信息传递。对feature map加1 0 mask，仅传递mask=1的filter和mask本身。减少内存通信。

#### 帮@xuxiaoqiao记录
1. 实现完整网络训练（能够与我的1相连）
2. devonv算子与ACL等别的库比较（ad-hoc的算法是在ACL等库中实现一个转置+conv2d_transpose+另一个转置）
3. 支持稀疏化相关的工作

---------------------------------

#### ResNet-18 前向
需要注意的是，PyTorch中部分写法在转ONNX再转NNVM的过程会产生错误：
详见[链接](https://shimo.im/docs/ueStE3bAUNkQL5P3/)
