### Week 8 Report

本周任务：
- 整理并撰写Huawei Report
- 完成NNVM maxpool2d反向图的构建


#### 关于Maxpool2d反向传播中某一点可能影响到的前向点的范围的计算

设在反向图中后一层的某个点为 ( i, j ),
则其影响到的在前一层中的点的范围为：

X(i) = [ MAX( newheight-padding[0], 0 ), MIN( newheight-padding[0]+Poolsize[0]-1, max_height-1 ) ]

X(j) = [ MAX( newweight-padding[1], 0 ), MIN( newweight-padding[1]+Poolsize[1]-1, max_weight-1 ) ]

newheight = stride[0] * i

newweight = stride[1] * i


设在反向图中前一层的某个点为 ( i, j ),
则其影响到的在后一层中的点的范围为：

Y(i) = [ MAX( floor( ( i+padding[0]-Poolsize[0] ) / stride[0] ), 0 ), MIN ( ceil( ( i + padding[0] ) / stride[0] ) , MAXPOSSIBLE(i)) ]


Y(j) = [ MAX( floor( ( j+padding[1]-Poolsize[1] ) / stride[1] ), 0 ), MIN ( ceil( ( j + padding[1] ) / stride[1] ) , MAXPOSSIBLE(j)) ]



#### 对于Pooling在别的架构中实现参考
[这里](https://shimo.im/docs/E4pBu1ZQn60bqpHJ/)

