#### 本周任务
1. Thesis
2. PPT框架
3. 完成实验部分

------------
#### 修改意见汇总
1. 在introduction部分可以提到customization作为edge-training的理由
2. 请求@xuxiaoqiao帮忙，测出profile信息，如内存占用、带宽占用，为进一步优化提供信息
3. 可以提一下federated learning
4. Inference应该测结构性与非结构性稀疏的性能差异，@cqchu可以提供CIFAR10-Resnet18的稀疏版本
5. Training应该测试优化与非优化实现的性能差别，精度差别要report，但差别不大。

#### 具体实现办法
1. 在周五中午前完成thesis框架，将Training的部分实现方法写出来，之后在答辩的时候再填
2. 周日前实现conv2d-sparse training，可以采用加mask的ad-hoc方式

------------

#### Test report on mali / opencl

| Sparsity_index  | num of channel calculated | time per 100 inference |
| ------------- | ------------- | ------------- |
| 12 | 1 | 0.631 |
| 6 | 2  | 0.768 |
| 4 | 3 | 0.522 |
| 3 | 4 | 0.598 | 
| 2 | 6 | 0.553 |
| 1 | 12 | 1.076 |
| NAN | 12 | 1.211 |

#### Test report on LLVM
| Sparsity_index  | num of channel calculated | time per 10^4 inference |
| ------------- | ------------- | ------------- |
| 12 | 1 | 0.787 |
| 6 | 2 | 0.868 |
| 4 | 3 | 0.809 |
| 3 | 4 | 0.801 | 
| 2 | 6 | 0.795 |
| 1 | 12 | 0.790 |
| NAN | 12 | 1+ ~ 10+ (not optimzied) |
