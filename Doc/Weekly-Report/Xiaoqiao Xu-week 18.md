# Channel-wise 稀疏算子讨论

- 路线: explicit 地传递 data tensor 和 index tensor
- 需要讨论算子实现:
  - elementwise add: 由于 index tensor 不能在 compile time 确定，所以 **ewise add 的输出 channel 数不能 compile time 确定**。一个最直白的 workaround 是为worst case生成计算&存储。
    - 问题: 关于这个问题，算法上是否有协同设计的空间？
    - related: TVM 目前不支持 dynamic tensor shape, 但是有一个 [RFC - Relay - Dynamic Dimensions](https://github.com/dmlc/tvm/issues/3042) 讨论
      - 子问题1: code generation: 直接实现需要大改 halide. 一个绕过的方法是手写支持 dynamic shape 的算子, 并通过 tvm.extern 方式调用. 缺点为: 没有利用 tvm 的自动生成算子的设计哲学.
      - 子问题2: storage allocation: 直接按照最悲观的情况分配即可.
  - BatchNorm: BatchNorm 有一步是给输出加 `beta`, 但是对于稀疏(被 omit out)的单元, 应该怎么处理?
    - 初步讨论的看法: fine-tuning 的时候通常会 freeze 中间的层，所以这个问题暂时可以搁置 ---- 有点类似于把 conv + BN 退化成 conv + bias.
 
# 非稀疏的 BatchNorm 反向实现
 
- 路线: 禁用旧的 `simplify_inference` pass, 实现相应的 FTVMCompute 和 FGradient
- 特殊处理: 更新 `moving_var` / `moving_mean` 的时候需要在计算图层面增加 `_assign` 
 
# Resources on TVM
 
- 计算图层面: 
  - tqchen 的课 [Systems for ML](https://dlsys.cs.washington.edu/)
- Tensor Expression: 
  - (tvm 文档?)

# demo 准备

- BatchNorm 算子
- Depthwise Grad
- CPU 占用率?
