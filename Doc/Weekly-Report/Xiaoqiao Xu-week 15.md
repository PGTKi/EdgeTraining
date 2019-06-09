上周工作

1. graph runtime 上的一些 debug。包括：调查为什么之前有些情况下运行不起来的、怎么更好地计时、LFY 之前遇到的一些奇怪情况。
2. 支持迁移学习（CIFAR10➡️Fashion-MNIST)

关于第一点，TVM Graph Runtime 目前的逻辑是 `run()` 只管 push task to queue list, get_output 才会等待 queue 结束并读结果。所以在计时的时候要 了解这一点
比较推荐的做法是用 `module.time_evaluator`

关于第二点，现在挑战比较大的是 batchnorm 的反向。此前 BN 只支持前向，并且它的前向也是特殊处理的（在 `SimplifyInference` 中特殊处理，把 BN 替换成其他 NNVM OP
所以反向也要特殊处理。

## 关于 Sparsity

NNVM 中已经有 CSR sparsity 的支持，可以看 `tvm.contrib.sparse` 模块。我们的设计可以参考这一点。总的来说就是一个 Python 层面 thin wrapper。(在 Python  object 里面保存多个普通的 TVM Tensor)