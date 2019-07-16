# Batch Norm

- [x] `_assign` 处理: 注册了一个改写 BatchNorm 的 Pass 更新 `moving_var` 和 `moving_mean`
- [x] BatchNorm 计算
- [ ] BatchNorm 调度: 对 GPU 的 `schedule_injective` 的尝试并不工作。

# CO-DEV

- [ ] 关于上周遇到的问题：TVM Tensor Expression 的表达力比较有限（因为它的 Bound Inference 等功能限制了表达能力). 解决方案是 `ir_builder`
