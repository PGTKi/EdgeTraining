### Week 3 Report

#### 路线图

请参考Xu Xiaoqiao的部分

#### 对于判断的支持（对应结构稀疏性）

初步结论：

- 不太可能在NNVM层面实现支持
- 更倾向于在TVM层面实现支持，即重新定义一种operator

tensorflow对图的支持:

tf.cond() operator: 利用placeholder来实现分支控制

但对intermediate result无法支持


tensorflow fold: 在tensorflow的基础上，对dynamic graph支持。上次更新为1年前，似乎已经放弃。

MXNet:

也是需要计算两者，较tf更差：

```python
a = mx.symbol.Variable(name='a')
b = mx.symbol.Variable(name='b')
a_minus_b = a - b
a_plus_b  = a + b

# gt = a > b
gt = a.__gt__(b)

result = mx.sym.where(condition=gt, x=a_minus_b, y=a_plus_b)

ex = result.bind(ctx=mx.cpu(), args={'a':mx.nd.array([1,2,3]), 'b':mx.nd.array([3,2,1])})
r = ex.forward()

print(r[0].asnumpy()) #result should be [1+3, 2+2, 3-1]
```


可以发现基本上Static Graph的DL framework对if的支持都是不好的。


不过对Approximate Random Dropout，有另外一种方法：

假设成pooling来做，理论上可以通过copy pooling层并改写来实现
