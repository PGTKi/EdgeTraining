### Week 9 Report
#### 本周任务：
- [x] ROCm/MIOpen调研
- [ ] 完成conv2d testing code撰写
- [ ] 完成end-to-end testing code(likely MNIST)
- [x] 完成 \_max_pool2d_gradient FTVMCompute，尽量按照in_grad做循环
- [ ] 调研conv2d如何用已有的ops实现，写出伪代码
- [ ] 尽量完成 \_conv2d_gradient FTVMCompute

#### 下周任务：

#### 潜在风险：


-------------------------------------
#### ROCm/MIOpen调研
- [\_max_pool2d_gradient](https://shimo.im/docs/E4pBu1ZQn60bqpHJ)
- [\_conv2d_gradient](https://shimo.im/docs/e3sfCvxPWaY9J1a9)

#### \_max_pool2d_gradient FTVMCompute 
目前，已经测试通过了以下几种实现方式：
- tvm.hybrid 以out_grad为循环
- tvm.compute 以in_grad为循环，目前还只能让input中的点和output整个做对比，因此大量test的时候会fail
- tvm.hybrid 以in_grad为循环。已经失败，问题在于没有办法在tvm.hybrid中创建mask tensor来记录in_grad和out_grad的对应关系

在虚拟机上运行时间约为1s

#### conv2d testing code
预计周日完成

#### end-to-end testing code
预计周日完成

#### \_conv2d_gradient 调研及伪代码
预计周一完成

#### \_conv2d_gradient FTVMCompute
后期完成
