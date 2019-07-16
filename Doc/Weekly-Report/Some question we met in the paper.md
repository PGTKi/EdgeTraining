### Some question we met during the process
1. TVM Tensor Expression 的表达受限，在实现conv2d_sparse过程中，需要用index tensor来控制data tensor，这点在TVM的默认表达中无法完成
2. LLVM Codegen过程中，return结果为tensor list的时候，会出现FCmpInst的报错。报错内容是：本来应该是fp类型的数据变成了pointer类型。猜测是tensor list比tensor函数多一个维度，在进行展开的时候，本来应该指向data的数据，指向了pointer/vector，导致了报错
3. 对Dynamic Shape的支持不好，Element-wise Add/Time和BN层需要prepare for the worst case,pruning ratio没有办法变化
4. CSR格式index数量两倍于data数量，控制粒度太细。

