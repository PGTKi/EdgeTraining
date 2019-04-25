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
<a href="https://www.codecogs.com/eqnedit.php?latex=X(i)&space;=&space;[max(newheight-padding[0],0),&space;min(newheight-padding[0]&plus;Poolsize[0],&space;maxheight)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X(i)&space;=&space;[max(newheight-padding[0],0),&space;min(newheight-padding[0]&plus;Poolsize[0],&space;maxheight)]" title="X(i) = [max(newheight-padding[0],0), min(newheight-padding[0]+Poolsize[0], maxheight)]" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=X(j)&space;=&space;[max(newweight-padding[1],0),&space;min(newweight-padding[1]&plus;Poolsize[1],&space;maxweight)]&space;newheight&space;=&space;stride[1]&space;*&space;j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X(j)&space;=&space;[max(newweight-padding[1],0),&space;min(newweight-padding[1]&plus;Poolsize[1],&space;maxweight)]&space;newheight&space;=&space;stride[1]&space;*&space;j" title="X(j) = [max(newweight-padding[1],0), min(newweight-padding[1]+Poolsize[1], maxweight)]" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=newheight&space;=&space;stride[0]&space;&plus;&space;i,&space;newweight&space;=&space;stride[1]&space;*&space;j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?newheight&space;=&space;stride[0]&space;&plus;&space;i,&space;newweight&space;=&space;stride[1]&space;*&space;j" title="newheight = stride[0] + i, newweight = stride[1] * j" /></a>

设在反向图中前一层的某个点为 ( i, j ),
则其影响到的在后一层中的点的范围为：
<a href="https://www.codecogs.com/eqnedit.php?latex=Y(i)&space;=&space;[&space;max(&space;floor(&space;(&space;i&plus;padding[0]-Poolsize[0]&space;)&space;/&space;stride[0]&space;),&space;0&space;),&space;min&space;(&space;ceil(&space;(&space;i&space;&plus;&space;padding[0]&space;)&space;/&space;stride[0]&space;)&space;,&space;MAXPOSSIBLE(i))&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y(i)&space;=&space;[&space;max(&space;floor(&space;(&space;i&plus;padding[0]-Poolsize[0]&space;)&space;/&space;stride[0]&space;),&space;0&space;),&space;min&space;(&space;ceil(&space;(&space;i&space;&plus;&space;padding[0]&space;)&space;/&space;stride[0]&space;)&space;,&space;MAXPOSSIBLE(i))&space;]" title="Y(i) = [ max( floor( ( i+padding[0]-Poolsize[0] ) / stride[0] ), 0 ), min ( ceil( ( i + padding[0] ) / stride[0] ) , MAXPOSSIBLE(i)) ]" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=Y(j)&space;=&space;[&space;max(&space;floor(&space;(&space;j&plus;padding[1]-Poolsize[1]&space;)&space;/&space;stride[1]&space;),&space;0&space;),&space;min&space;(&space;ceil(&space;(&space;j&space;&plus;&space;padding[1]&space;)&space;/&space;stride[1]&space;)&space;,&space;MAXPOSSIBLE(j))&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y(j)&space;=&space;[&space;max(&space;floor(&space;(&space;j&plus;padding[1]-Poolsize[1]&space;)&space;/&space;stride[1]&space;),&space;0&space;),&space;min&space;(&space;ceil(&space;(&space;j&space;&plus;&space;padding[1]&space;)&space;/&space;stride[1]&space;)&space;,&space;MAXPOSSIBLE(j))&space;]" title="Y(j) = [ max( floor( ( j+padding[1]-Poolsize[1] ) / stride[1] ), 0 ), min ( ceil( ( j + padding[1] ) / stride[1] ) , MAXPOSSIBLE(j)) ]" /></a>

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


