徐涍荍

Contents:

* [Tutorial: Run TVM with Python/C++ debugger](#tutorial-run-tvm-with-pythonc-debugger)
* [以 mali 上的 `conv2d` 实现为例， 说明应该如何实现一个新的 Op](#user-content-以-mali-上的-conv2d-实现为例-说明应该如何实现一个新的-op)
* [下一步的计划](#user-content-下一步的计划)

# Tutorial: Run TVM with Python/C++ debugger

背景: 在 debugger 里观察 tvm 代码的运行要比干看代码更直观. TVM 的 Python 代码和 C++ 代码相互调用 (既有在 Python 里调用 C++ 函数的, 也有在 C++ 里调用 Python 函数的). 本文记录怎么设置 debugger, 目标是可以同时在 C++ 和 Python 代码里设置断点.

> 其实就是让 `gdb` attach 到 python 解释器进程上。然后 pdb/gdb 两开花...

首先把想运行的代码放到某个 py 文件里

```
cd tvm
mkdir playground
touch playground/hello.py
vim hello.py
```

```
# contents of hello.py

import nnvm
import tvm
from nnvm import symbol
from nnvm import graph
import os

if __name__ == "__main__":
    print(os.getpid())
    input("Press enter to continue.")

    x = symbol.Variable("x")
    conv1 = symbol.conv2d(x, channels=6, kernel_size=(3, 3), name="conv1", use_bias=True)
    out_sum = symbol.sum(conv1)
    ys = out_sum
    xs = x

    g = graph.create(out_sum)
    g._set_symbol_list_attr('grad_ys', ys)
    g._set_symbol_list_attr('grad_xs', xs)
    ny = len(ys.list_output_names())
    grad_ys = [symbol.ones_like(ys[i]) for i in range(ny)]
    g._set_symbol_list_attr('grad_ys_out_grad', grad_ys)
    sym = g.apply('Gradient').symbol
    nx = len(xs) if isinstance(xs, list) else len(xs.list_output_names())
    ret = [sym[i] for i in range(nx)]
    # print(len(ret))
    # print("---")
    # print(graph.create(ret[0]).json())
    # print("---")
    # print(graph.create(ret[0]).ir())

    g2 = graph.create(ret[0])
    deploy_graph, lib, params = nnvm.compiler.build(g2, target=tvm.target.mali(), shape={"x": (1,3,28,28)}, dtype="float32")
    print("exiting")
```

在 CLion 中, `if __name__ == "__main__":` 会显示三角形, 选择 "Debug `Hello`" 。

然后在 CLion 顶部菜单 "Run" ➡️ "Attach to Process" 输入 Python 解释器的 pid (已经 print 了), 此时同一个 pid 有两个选项, **选择 "Attach with LLDB/GDB to"**, 不要选 "Attach with Python Debugger To"。

所以现在你已经同时开了 lldb(gdb) 和 python debugger. 在两边设置的代码都可以 step by step 运行了.


# 以 mali 上的 `conv2d` 实现为例， 说明应该如何实现一个新的 Op

> 背景和需求: 分析 `conv2d` 的实现是为了模仿它，实现 `_conv2d_grad`

Call stack 就不分析了。之前已经有一个 doc. 另外, 在 debugger 里设置断点跑一遍是更加直观的。

我们直接看 `topi/python/topi/mali/conv2d.py` 中的函数. `conv2d_mali` ➡️ `_decl_spatial_pack`

实际的计算定义位于那几个 `tvm.compute()` 的函数中。 

cfg变量: 与 autotvm 模块相关，是用来寻找、存储最佳的 split 、最佳的 reorder 等参数的。

## 关于 tvm.compute() 函数、以及 tvm 下层是怎么搞的

tvm 框架内默认只支持密集的计算(note: 稀疏的相关内容可以通过 hybrid 相关的 API 实现, 将来补充). tvm.compute() 定义计算是什么, 而 schedule 相关的 API 定义顺序（比如先算 row 还是先算 column)

tvm.compute() 有三个参数. **它的输出是 Expression AST 的节点**. 它的输入是 shape、一个定义计算关系的 lambda 函数、名字.

```
dvshape = (N, OH // VH, OW // VW, CI, VH*HSTR + KH-1, VW*WSTR + KW-1) # 这是一个长度为6的tuple
data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw:
                        data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw],
                        name='data_vec')
```

第一个例子, data_vec 处的 tvm.compute()，这里 data_pad 是一个 4D `Tensor`类, (在本例子里是 [1,3,28,28], 分别是 NCHW). `Tensor` 类不会为向量分配实际的值, 只是作为 Expr AST 记录其尺寸等信息. 我们通过上面的函数调用, 声明了 data_vec 是一个 6D `Tensor`, 并且 data_vec的第[n, h, w, ci, vh, vw]个元素的定义是: data_pad 的第 `[n,ci,h*VH*HSTR+vh,w*VW*WSTR+vw]` 个元素. （所以这个计算关系是密集的）。

接下来看同一个函数里的另一个例子: 计算 conv 的

```
ovshape = (N, CO // VC, OH // VH, OW // VW, VH, VW, VC)  # 这是一个长度为6的tuple
conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
    tvm.sum(data_vec[n, h, w, ci, vh*HSTR+kh, vw*WSTR+kw].astype(out_dtype) *
            kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
            axis=[ci, kh, kw]), name='conv')
```

相似的, conv 是一个 6D `Tensor`. 这里定义的计算关系是:

conv[n, co, h, w, vh, vw, vc] 的值**定义为**：

```
tvm.sum(data_vec[n, h, w, ci, vh*HSTR+kh, vw*WSTR+kw].astype(out_dtype) *
            kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
            axis=[ci, kh, kw])
```

用 C++ 代码也可以构造出等价的 Expr AST Node。如果有找到合适的范例的话，我们可以考虑写一下 ---- 如果有需求的话. 不过目前我见到卷积相关的实现都是在 Python 里做的，依样画葫芦会更轻松.

## 关于 tvm.compute() 的具体实现

**本小节不重要, 可跳过**

tvm.compute() 的定义在 `python/tvm/api.py`

这里的 Python 代码会建立若干个 `_IterVar`, 把这些 `_IterVar` 作为参数传递给第二个参数(fcompute, 即之前的 lambda)。

回顾上一小节：

```
data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw:
                        data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw],
                        name='data_vec')
```

data_pad 的类型是 `Tensor`，方括号运算符对应的是 `python/tvm/tensor.py` 中的 `Tensor.__getitem__()`, 返回了 TensorSlice. 最终调用了 `Tensor.__call__`. 所以这个 lambda 会在 tvm.compute() 里面被呼叫, 输入参数是若干个 `_IterVar`, 返回值是 `Tensor.__call__()` 的返回值.

# 下一步的计划

1. 我们已经分析出了写一个新的运算符的做法. 下一步是写为 `_conv2d_grad` 写 tvm.compute() 的计算关系. 写完之后就可以验证结果是否正常了.
2. 在第一步完成之后，通过 autotvm 调优性能。这件事必须在 1 完成之后再做
3. 通过 tvm.hybrid ，准备后续稀疏的操作.
