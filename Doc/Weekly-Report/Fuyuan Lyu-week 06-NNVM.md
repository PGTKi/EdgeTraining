#### 本周工作

主要分为两个方面：

- NNVM/TVM 框架草图
- NNVM 反向图实现



##### 关于草图

请见OneNote



##### 官方关于NNVM反向图：

分两种情况：

1. 实现比较简单，如ReLU, Dense。直接在set_Attribute的cc文件中

   ````c++
   NNVM_REGISTER_OP("Some operations")
       ...
   .set_attr<FGradient>(
     // The way you compute your gradient with general operations
   })
   ````

2. 实现比较复杂，如Conv2d（这里，TVM team把conv2d作为一类操作，其中有大量的关于conv2d具体的实现方法，如spatial，winogard，element-wise conv2d）。会额外定义运算符。如最简单的spatial conv2d对应的是_conv2d_grad



##### 关于NNVM反向图：

- 目前已经实现ReLU。（使用的是general ops）

- NNVM本身已经集成了MaxPool和Dense，我们对这两者的实现主要卡在对NNVM中tensor等变量的定义上（这两者要求2-d input）

- 比较麻烦的是Conv2d，根据目前网络的报错：

  ````shell
  nnvm._base.NNVMError: [12:38:56] /home/firefly/Huawei/xiaoqiao.xu/tvm/nnvm/include/nnvm/op.h:532: Check failed: idx < data_.size() && data_[idx].second: Attribute FTVMCompute has not been registered for Operator _conv2d_grad
  ````

  因此，是_conv2d_grad并没有定义FTVMCompute属性，我们需要在nnvm/src/top/nn/convolution.cc中添加对应的backward逻辑。由于NNVM在这一块的缺失，猜想这也是tinyflow中需要自己单独构建back nn node的原因。




