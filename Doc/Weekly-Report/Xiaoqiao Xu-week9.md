## Op 开发

### poolgrad

为了解决 gather/scatter 问题。上周用 tvm.compute() 重新实现了一遍。现在已经能通过 lfy 写的测试。可以认为 NNVM 的计算部分定义已经完工。

目前还差 scheduling. 鉴于 pooling 是一个低运算、高IO的算子, 比较合适的做法是让它和其他的 OP 做 fusion. 单独做 scheduling 能起到的作用比较有限.

### conv2d

* Compute 层面：
  * 原计划是复制 pool_grad 的代码，稍微改动一点点，变成 conv2d_grad.
  * 根据周末的新信息输入，conv2d_transpose 可以当作 conv2d_grad 用.
    * 大概的已经写了，还在看为什么测试不过.
    * 目前 tvm 的 conv2d_transpose 只支持 group == 1 的情况. 后续要增加 group != 1 的代码

## Mali 架构调查

RK3399 的 GPU (Mali Midgard uarch, T860 MP4) 是 SIMD / VLIW 的. 4 Core, 每个 Core 有 2个 ALU。

所以测试平台 T860MP4 的算力是 2 * 10(FLOP/cyc) * 4(core) * 800(MHz)= 64GFLOP/s
每个 ALU 每个 cycle 可以进行以下运算(10 FLOP, 或5个FMA): (信息来源: AnandTech)

* 4 vector adds
* 4 vector multiply
* 1 scalar add
* 1 scalar multiply

## 其他事项

本周看了 Halide IR `Expr` 和 `Stmt` / `tvm.hybrid`, 看了 `tvm.lower()`
