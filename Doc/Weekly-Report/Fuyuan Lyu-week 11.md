#### 本周任务：
- 继续完成end-to-end testing code
- 开始对结构性稀疏进行支持

#### 下周任务：

#### 可能风险：


----------------

#### 结构性稀疏
已经确认应该在TVM层面进行优化，NNVM层面的optimzation主要是以layer fusion为主。

已经成功在TVM中重新定义了conv2d_sparse和_conv2d_sparse_gradient操作符。目前已经完成两者的接口的支持和FTVMcompute，正在通过传入的sparse matrix对两者的schedule做支持。

当前优先考虑前向。反向稍后。


#### end2end testing code
TVM本身为测试做了一些API，其中比较有用的是nnvm/testing/utils.py中的create_workload函数。可以根据神经网络定义的handle，返回网络结构和参数。

目前已经能够成功create_workload，并完成前向计算。

之前所说的optimizer的问题暂时通过重新定义param来解决。

目前卡在如何生成optimizer的结果。



