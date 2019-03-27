#### 本周任务

暂略

--------
#### Tinyflow中NNVM的调用
*以下部分主要记述在[tinyflow](https://github.com/tqchen/tinyflow)代码示例对NNVM的调用*

##### 首先，看一下tinyflow是什么？
tinyflow是一个展示NNVM的使用方法的教学示例。本质上是一个mini版的DL框架，且有以下特点：
- interface语法类似tensorflow
- Graph Level的调用和优化使用NNVM
- Ops Level的实现使用Torch7

##### 从examples中开始，尝试厘清调用关系
我们从example\mnist_mlp_auto_shape_interence.py这一最简单的例子开始向下说明在tinyflow中的调用关系
仅从代码上看，这一段代码和tensorflow并没有多大区别。以下是构建模型的部分:

``` python
x = tf.placeholder(tf.float32)
fc1 = tf.nn.linear(x, num_hidden=100, name="fc1", no_bias=False)
relu1 = tf.nn.relu(fc1)
fc2 = tf.nn.linear(relu1, num_hidden=10, name="fc2")
```

而python\tinyflow\nn.py中并不包含linear和relu操作，仅有conv2d和maxpool两个操作，且两个操作起地都是类似嵌套函数的作用。
```python
def max_pool(data,
             strides=[1, 1, 1, 1],
             padding='VALID',
             data_format='NCHW', **kwargs):
    return _sym.max_pool(data, strides=strides, padding=padding,
                         data_format=data_format, **kwargs)
```

在nnvm\python\nnvm\symbol.py中，我们并没有找到了对应的代码。*（对于python和c++在这个部分是如何interact的，我需要进一步了解）*

之后：
关于python和c++在TVM框架中的执行，我参考的是[Runtime](https://docs.tvm.ai/dev/runtime.html)。在Runtime中定义好了c++的function后，TVM在python层面需要通过TVM_REGISTER_GLOBAL和tvm.get_global_func来注册和调用对应的变量名。其中在注册过程中需要定义好输入和输出。

##### 从source开始，厘清tinyflow与Torch7的调用关系
根据README，我们了解到Ops Level的实现都是基于Torch7的。而我们在src\torch\下，我们发现了op_nn_torch.cc和op_tensor_torch.cc。在这两个文件中，我们发现了大量的NNVM注册调用：
```C
NNVM_REGISTER_OP(linear)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    local wshape = ishape[2]
    local m = nn.Linear(wshape[2], wshape[1])
    if #ishape == 2 then
      m = m:noBias()
    end
    return m
  end
)");
```
引号中的文字，就是对应操作用Lua语言在Torch7架构中的实现。
在这里，我们发现：NNVM的特点就是大量的Ops都需要通过NNVM_REGISTER_OP来注册。


在src\下，我们发现了op_nn.cc和op_tensor.cc等文件。其中包含了在对应src\torch\下的cc文件中注册的Ops:
```C
NNVM_REGISTER_OP(linear)
.describe("A linear transformation layer")
.set_attr_parser(ParamParser<LinearParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
    return (dmlc::get<LinearParam>(attrs.parsed).no_bias? 2 : 3);
  })
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    if (dmlc::get<LinearParam>(attrs.parsed).no_bias) {
      return std::vector<std::string>{"data", "weight"};
    } else {
      return std::vector<std::string>{"data", "weight", "bias"};
    }
  })
.include("nn_module")
.set_attr<FInferShape>("FInferShape", LinearShape);
```

可见，在src\torch\下的cc文件注册的是Ops在Torch7框架下如何实现，而src\下的cc文件则在定义对应Ops中的相关输入、输出等信息。


此外，在src\op_nn.cc中，作者还通过inline函数，构建反向图：
```C
// create a backward node
inline std::vector<NodeEntry> MakeNNBackwardNode(
    const NodePtr& n,
    const std::vector<NodeEntry>& ograds) {
  static auto& backward_need_inputs = Op::GetAttr<bool>("TBackwardNeedInputs");
  static auto& backward_need_outputs = Op::GetAttr<bool>("TBackwardNeedOutputs");
  static auto& backward_num_nograd = Op::GetAttr<int>("TBackwardNumNoGradInputs");
  nnvm::NodePtr p = nnvm::Node::Create();
  p->attrs.op = nnvm::Op::Get("_backward");
  p->attrs.name = n->attrs.name + "_backward";

  NNBackwardParam param;
  param.forward_readonly_inputs = static_cast<uint32_t>(n->inputs.size());
  param.need_inputs = backward_need_inputs[n->op()];
  param.need_outputs = backward_need_outputs[n->op()];
  param.num_no_grad_inputs = backward_num_nograd.get(n->op(), 0);
  CHECK_EQ(ograds.size(), 1);
  CHECK_EQ(param.forward_readonly_inputs + param.num_states,
           static_cast<uint32_t>(n->inputs.size()));

  p->attrs.parsed = param;
  p->control_deps.emplace_back(n);
  // layout [output_grad, inputs, states, output]
  p->inputs.push_back(ograds[0]);
  if (param.need_inputs) {
    for (index_t i = 0; i < param.forward_readonly_inputs; ++i) {
      p->inputs.push_back(n->inputs[i]);
    }
  }
  for (index_t i = 0; i < param.num_states; ++i) {
    p->inputs.push_back(n->inputs[i + param.forward_readonly_inputs]);
  }
  if (param.need_outputs) {
    for (uint32_t i = 0; i < n->num_outputs(); ++i) {
      p->inputs.emplace_back(nnvm::NodeEntry{n, i, 0});
    }
  }

  std::vector<nnvm::NodeEntry> ret;
  for (index_t i = 0; i < param.forward_readonly_inputs; ++i) {
    ret.emplace_back(nnvm::NodeEntry{p, i, 0});
  }
  if (param.num_states != 0 || param.num_no_grad_inputs != 0) {
    nnvm::NodePtr np = nnvm::Node::Create();
    np->attrs.op = nnvm::Op::Get("_no_gradient");
    for (uint32_t i = 0; i < param.num_no_grad_inputs; ++i) {
      ret.at(ret.size() - i - 1) = nnvm::NodeEntry{np, 0, 0};
    }
    for (index_t i = 0; i < param.num_states; ++i) {
      ret.emplace_back(nnvm::NodeEntry{np, 0, 0});
    }
  }
  return ret;
}
```


----
*之前我们讨论了tinyflow中的架构（使用旧版NNVM做graph和Torch7做op），可是我们怎么在NNVM/TVM中找到对应接口呢？*

@[XXQ](https://github.com/xuxiaoqiao)之前在[石墨文档](https://shimo.im/docs/FOGmkWlh5xMr0ivd/)中从NNVM代码层讨论了NNVM和TVM的交互路径。
我们承接做进一步的分解





