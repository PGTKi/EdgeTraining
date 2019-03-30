# 本周进展

* 本周拿到了硬件。环境搞好了，可以编译与 rpc 调试
* NNVM 的调用大概看了一下, 整理了下层的调用关系链

TODO:
[ ] 确定技术路线图

---

下层调用的 API 结构仍旧更新在: https://shimo.im/docs/FOGmkWlh5xMr0ivd

---

NNVM 的反向图 pass 需要 `grad_ys` `grad_xs` `grad_ys_out_grad` 这三个 attr

一个 minimal working example 如下

```
import nnvm
import tvm
from nnvm import symbol
from nnvm import graph

x = symbol.Variable("x")
dense = symbol.dense(x, units=10)
out_sum = symbol.sum(dense)

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

print(graph.create(ret[0]).json())
print("---")
print(graph.create(ret[0]).ir())
```
