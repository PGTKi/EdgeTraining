### Week 8 Report

#### 本周任务：
- [x] 整理并撰写Huawei Report
- [x] 完成maxpool2d testing code
- [ ] 完成NNVM maxpool2d反向图的构建

#### 下周任务
- 完成conv2d testing code撰写
- 完成end-to-end testing code(likely MNIST)
- 调研conv2d如何用已有的ops实现，写出伪代码
- 完成 _max_pool2d_gradient FTVMCompute
- 尽量完成 _conv2d_gradient FTVMCompute
- HalideIR gather/scatter问题
- _max_pool2d_grad 尽量按照input_grad做循环
- 其他

#### 潜在风险
- nnvm.testing.check_computation.check_function函数仅支持传回output_grad和input，maxpool2d尚且可以根据input直接找到对应的max点，完成计算；conv2d一定要借助kernel的存在。如何传递kernel是一个问题。
- check_function只能测试output，但conv2d需要测试kernel和output的update。conv2d怎么测试对kernel的update？
- 当前版本v0.5能否避免Relay的干扰

----------------------------------------

#### Maxpool2d testing code

##### 关于Maxpool2d反向传播中某一点可能影响到的前向点的范围的计算
设在反向图中后一层的某个点为 ( i, j ),
则其影响到的在前一层中的点的范围为：

X(i) = [ MAX( newheight-padding[0], 0 ), MIN( newheight-padding[0]+Poolsize[0], max_height ) ]

X(j) = [ MAX( newweight-padding[1], 0 ), MIN( newweight-padding[1]+Poolsize[1], max_weight ) ]

newheight = stride[0] * i

newweight = stride[1] * i

设在反向图中前一层的某个点为 ( i, j ),
则其影响到的在后一层中的点的范围为：

Y(i) = [ MAX( floor( ( i+padding[0]-Poolsize[0] ) / stride[0] ), 0 ), MIN ( ceil( ( i + padding[0] ) / stride[0] ) , MAXPOSSIBLE(i)) ]

Y(j) = [ MAX( floor( ( j+padding[1]-Poolsize[1] ) / stride[1] ), 0 ), MIN ( ceil( ( j + padding[1] ) / stride[1] ) , MAXPOSSIBLE(j)) ]


##### 关于check_function
nnvm.testing.check_computation.check_function可以帮助测试新的Op。支持前向，反向和numerical测试。

check_function会先build整个网络的graph（包括反向图）。所以对于maxpool2d这种没有注册FTVMCompute属性的Operator，即使只测试前向，也会build失败。而为_max_pool2d_grad注册FTVMCompute后，即可完成前向测试。


##### final code
[这里](https://github.com/acada-sjtu/EdgeTraining/blob/master/Code/edge-tvm/op-test-maxpool2d.py)

----------------------------------------

#### NNVM maxpool2d反向图的构建
##### 对于Pooling在别的架构中实现参考
[这里](https://shimo.im/docs/E4pBu1ZQn60bqpHJ/)

----------------------------------------


#### 关于开发版本的选择
为了避免Relay对于NNVM的干扰，我们选定v0.5作为版本。


