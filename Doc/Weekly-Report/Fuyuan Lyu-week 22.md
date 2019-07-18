### 本周目标
- [ ] conv2d_sparse在opencl上运行成功
- [ ] 开始做elementwise-add/multi
- [x] 整理问题

### 本周进度
#### 问题整理
整理在实践过程中遇到的[问题](https://github.com/acada-sjtu/EdgeTraining/blob/master/Doc/Weekly-Report/Some%20question%20we%20met%20in%20the%20paper.md)


#### Conv2d_sparse
In progress， test function finished

##### tvm.extern
Extern实现中不存在compute和schedule的概念，若想要在TVM中实现conv2d_sparse函数，则应该：
1. TVM 中定义conv2d_sparse_extern函数，实现原TVM中Compute和Schedule的功能
2. NNVM 中Compute调用topi.conv2d_sparse_extern函数，Schedule使用默认调用实现（将之当作黑盒处理）

#### Elementwise-add/multi
Freezed
