### Week 9 Report
#### 本周任务：
- [x] ROCm/MIOpen调研
- [x] 完成conv2d testing code撰写
- [ ] 完成end-to-end testing code(likely MNIST)
- [x] 完成 \_max_pool2d_gradient FTVMCompute，尽量按照in_grad做循环
- [x] 调研conv2d如何用已有的ops实现，写出伪代码
- [ ] 尽量完成 \_conv2d_gradient FTVMCompute
- [x] 初步调研对动态图的支持
- [x] 初步调研对结构性稀疏的支持

#### 下周任务：

- 继续完成end-to-end testing code
- 确保FTVMCompute正确性
- 尽量优化Schedule
- __进一步细化对结构性稀疏的支持__
  - __在考虑控制的粒度时候，应该以硬件的最低粒度为主__
  - __要详细了解GPU进行运算的粒度__
  - __控制粒度要考虑fusing的问题__
- 在周末前，完成end-to-end training

#### 潜在风险：
- 听怜悯分析，TVM对training的效果会很一般
- 可能需要手写顶端的部分工作，TVM对training没有留下很多的空间

-------------------------------------
#### ROCm/MIOpen调研
- [\_max_pool2d_gradient](https://shimo.im/docs/E4pBu1ZQn60bqpHJ)
- [\_conv2d_gradient](https://shimo.im/docs/e3sfCvxPWaY9J1a9)

#### \_max_pool2d_gradient FTVMCompute 
目前，已经测试通过了以下几种实现方式：
- tvm.hybrid 以out_grad为循环
- **tvm.compute 以in_grad为循环，reduce_axis一定要是编译器常量，但是可以加偏移量（这是目前最满意的实现）**
- tvm.hybrid 以in_grad为循环。已经放弃，问题在于没有办法在tvm.hybrid中创建mask tensor来记录in_grad和out_grad的对应关系

在虚拟机上运行时间略小于1s
 
之后会在tvm.schedule上进行优化，目前schedule只是默认的方式

#### conv2d testing code
[已经完成](https://github.com/acada-sjtu/EdgeTraining/blob/master/Code/edge-tvm/op-test-conv2d.py)

#### end-to-end testing code
暂缓完成，目前没有可预见的技术障碍。提高了Dynamic Graph可行性分析的优先级

#### \_conv2d_gradient 调研及伪代码
实现方法详见调研

#### \_conv2d_gradient FTVMCompute
目前已经初步完成FTVMCompute，暂不优化Schedule，正在测试

#### 对结构性稀疏的支持
采用control的方法，在计算图中添加control元素：
- 这需要涉及到对Op的重写
- 这可能是一个对方法很依赖的实现
- 结构性稀疏有可能与AutoTVM的特性（AI for System）产生冲突。AutoTVM在一个device上部署了最优解，但结构性稀疏可能导致部分没有算，因此效能未必会大幅提升。有可能需要与AutoTVM进行一定的衔接。

#### 对动态图的支持
1. 初步调研了TVM对现有的frameworks import数据的支持，感觉是一个非常case-by-case的工作
2. 如果考虑在TVM中实现动态特性，则需要把compiler load到edge端，负载过大


