# Batch Norm

- [x] `_assign` 处理: 注册了一个改写 BatchNorm 的 Pass 更新 `moving_var` 和 `moving_mean`
- [x] BatchNorm 计算
- [ ] BatchNorm 调度: 对 GPU 的 `schedule_injective` 的尝试并不工作。
- [x] 重构了 Conv2D 的代码并初步通过测试。方便后续 DepthwiseConv2D 处理

# CO-DEV

- [x] 关于上周遇到的问题：TVM Tensor Expression 的表达力比较有限（因为它的 Bound Inference 等功能限制了表达能力). 解决方案是 `ir_builder`
- [x] 解决一个关于 workload 的报错
