### Week 4 report

#### 上周目标

1-2 week 内在TVM中实现以下的四个函数：

- conv2d
- fully-connected
- ReLU
- Pooling



#### 上周工作

##### 环境配置

周日下午拿到RK3399开发板后，我开始了搭建一些环境。我所参考的攻略主要来自于[官方文档](http://wiki.t-firefly.com/zh_CN/Firefly-RK3399/upgrade_firmware.html)和[另一份官方文档](http://www.t-firefly.com/doc/product/info/id/73.html)。此外，需要注意的是，由于RK3399的OS不同，需要依照[烧写要求](http://wiki.t-firefly.com/zh_CN/Firefly-RK3399/upgrade_table.html)选择对应的Android Tool版本。由于官方下载服务器的失灵，其中2.58版本需要从[这里](http://www.t-firefly.com/doc/download/page/id/4.html#windows_22)下载。全程使用windows OS进行烧写操作。

此后，又配置了一些端口映射、安装了pip, cmake等常用工具



##### TVM 环境搭建

周一与周二的主要工作在于跑通TVM相应的工作环境，在RK3399和本地PC都配置了相应的方案。

TVM团队本身采用交叉编译架构，为了：

1. RK3399 runtime很稳定，更改TVM代码本身不需要在RK3399上编译（费时间）
2. llvm/tvm等复杂编译不用在RK3399上跑



#### 本周目标

##### tinyflow

参考tinyflow的背景，取得以下成就：

- 已知一个简单的前向图，调用Gradient中的pass得到对应的反向图
- 根据前述反向图，完成training



#### TVM

在TVM中实现以下的四个函数：

- conv2d
- fully-connected
- ReLU
- Pooling



#### 本月目标及风险

在 rk3399 上用 GPU 运行基本的训练功能，可以容忍低性能。尽量完成优化

潜在风险：不熟悉代码，对于工作量估计较低。


#### 与老师讨论后下周的目标
1. 找到NNVM对于反向图的接口
2. 在TVM层面实现对于反向算子的支持
3. 下周之前列出整个网络的flow，可能有多个路径支持
4. （选择）考虑之后后期对tensorflow、pytorch等框架的model import支持
5. （选择）稀疏性等动态优化可能需要在算子层面实现优化
5. 帮助另外两外同学找到如果要加control gate需要在TVM哪个部分实现
