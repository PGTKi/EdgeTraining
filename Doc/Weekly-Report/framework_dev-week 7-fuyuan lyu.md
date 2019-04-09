#### Week 7 Report
本周的任务主要为两个方面：
1. NNVM层面training支持的实现
2. Paper reading

##### NNVM层面training支持的实现
目前的状况：
- dense, ReLU(上周实现的), leaky ReLU, softmax, log_softmax 在NNVM层面上可以生成反向code
- Maxpool, conv2d, avgpool, conv2d_transpose 仅在NNVM正向pass可以

主要的难点在于：
- 对于code的写法太不熟悉
- 对于NNVM架构下的各种概念还在熟悉中

##### paper reading: 

