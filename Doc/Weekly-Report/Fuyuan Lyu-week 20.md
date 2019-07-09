## Conv2DSparse NNVM改写
- [x] FGetAttrDict
- [x] FListInputNames: 在List中加入data_index和weight_index
- [x] FInferShape: 更改所有的index、output的ndim，增加index的shape assignment
- [x] FInferType：更改所有的index、output的ndim，将else中的ElemWiseType更改为<-1,-1>以避开判定
- [x] FCorrectLayout：更改Layout属性，以便衔接后续的FInferShape和FInferType

## NNVM Compute和Schedule

---------------------------

## 一些长期的课题
### Two big assumptions
1. 边缘端需要支持训练
2. 边缘端的训练和推断可能需要用到pruning、quantization等efficient neural network手段


### Architecture 层面
#### 1. 在边缘端上的动态图问题
__需求__

在边缘端训练的情况下，pruning ratio若想变化，则需要重构整个network，代价不小。如何处理？

__可能的参考__

- PyTorch
- TensorFlow Eager execution 


#### 2. JIT与动态图的关系，为什么大量业界的高性能计算库大多使用JIT
__需求__

同上。需要调研JIT与动态图的区别。可能JIT是一种解决方案。

__可能的参考__

- [some discuss](https://news.ycombinator.com/item?id=16434634)
- [Tensor Comprehension](https://github.com/facebookresearch/TensorComprehensions)
- [XLA](https://www.tensorflow.org/xla)
- Others



### Algorithm 层面
#### 1. 对于BN层的处理
__需求__

目前几乎所有的pruning、quantization算法遇到BN都需要转换，代价大。


------

## 一些短期的课题
1. depth-wise的反向（TVM+少量NNVM）
2. ReLU, maxpool, avgpool对sparsity index的支持（NNVM）
3. 良好的import问题（NNVM）
4. Dense对sparsity index的支持（NNVM+TVM，输入稀疏、输出不稀疏）


